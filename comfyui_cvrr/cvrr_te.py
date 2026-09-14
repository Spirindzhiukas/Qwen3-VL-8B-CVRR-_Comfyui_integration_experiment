"""ComfyUI text-encoder classes that run CVRR and emit FLUX.2 Klein conditioning.

The integration deliberately re-uses ComfyUI's own modules:

* ``comfy.text_encoders.qwen3vl.Qwen3VL``      -- the Qwen3-VL-8B LM + vision tower
* ``comfy.text_encoders.qwen3vl.Qwen3VLTokenizer`` -- Qwen3-VL chat template with
  ``<|vision_start|><|image_pad|><|vision_end|>`` image placeholders
* ``comfy.text_encoders.flux.Flux2TEModel``    -- the 3-layer-tap -> ``[B, L, 12288]``
  Klein conditioning layout
* ``comfy.text_encoders.flux.KleinTokenizer8B`` -- Klein's prompt template

Only the *decoder sweep* is replaced: instead of a single forward pass,
:class:`CVRRTextModel` runs CVRR's split-branch recurrence
(:mod:`comfyui_cvrr.cvrr_core`) and hands the resulting taps to the very same
Klein stacking code, so downstream nodes see an ordinary ``CONDITIONING``.

Runtime geometry (from the released ``config.json``)::

    ell_star = 22   recurrent layer = 23   upper decoder = 24..35
    T = 4           beta = 0.33            merged fp32 transition (7 projections)
    Klein taps = layers 9, 18, 27          output = [B, tokens, 3 * 4096]
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

import comfy.text_encoders.flux
import comfy.text_encoders.qwen3vl
from comfy import sd1_clip
from comfy.ldm.modules.attention import optimized_attention_for_device

from .cvrr_core import (
    KLEIN_9B_TAPS,
    CVRREncodeResult,
    CVRRSpec,
    CVRRTextEncoderCore,
    MergedTransition,
    install_merged_transition,
    klein_stack_taps,
    merged_transition_keys,
)

log = logging.getLogger("comfyui_cvrr")

#: Geometry of ``dmis-lab/Qwen3-VL-8B-CVRR`` (read from its ``config.json``).
RELEASE_SPEC = CVRRSpec(
    ell_star=22,
    num_recurrent_steps=4,
    beta=0.33,
    taps=KLEIN_9B_TAPS,
    name="CVRR-Qwen3-VL-8B",
)

RELEASE_CONFIG_NAME = "cvrr_release_config.json"
TRANSITION_FILE_NAME = "merged_transition.safetensors"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class CVRROptions:
    """Per-run CVRR settings attached to a text encoder."""

    spec: CVRRSpec = RELEASE_SPEC
    mode: str = "aligned"
    enabled: bool = True
    tap_gains: Tuple[float, ...] = field(default_factory=tuple)
    block_visual_access: bool = False

    def resolved_gains(self) -> Tuple[float, ...]:
        if not self.tap_gains:
            return tuple(1.0 for _ in self.spec.taps)
        if len(self.tap_gains) != len(self.spec.taps):
            raise ValueError(
                f"tap_gains has {len(self.tap_gains)} entries but the spec has "
                f"{len(self.spec.taps)} taps"
            )
        return tuple(float(g) for g in self.tap_gains)

    def with_changes(self, **kwargs: Any) -> "CVRROptions":
        return replace(self, **kwargs)


def load_release_spec(directory: str, taps: Sequence[int] = KLEIN_9B_TAPS) -> CVRRSpec:
    """Read ``cvrr_release_config.json`` (or the ``release`` block of ``config.json``)."""
    candidates = [
        os.path.join(directory, RELEASE_CONFIG_NAME),
        os.path.join(directory, "config.json"),
        os.path.join(directory, "native_backbone", "config.json"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        release = data.get("release", data)
        if "ell_star" not in release:
            continue
        spec = CVRRSpec.from_release(release, taps=taps)
        log.info("comfyui_cvrr: loaded CVRR geometry from %s: %s", path, spec)
        return spec
    log.info("comfyui_cvrr: no release metadata found in %s, using defaults", directory)
    return replace(RELEASE_SPEC, taps=tuple(int(t) for t in taps))


# ---------------------------------------------------------------------------
# Attention plumbing
# ---------------------------------------------------------------------------


class _AttentionResolver:
    """Resolve ComfyUI's attention kernel lazily, per device and per mask use."""

    def __call__(self, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
        op = optimized_attention_for_device(
            q.device, mask=mask is not None, small_input=True
        )
        return op(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)


_ATTENTION = _AttentionResolver()


# ---------------------------------------------------------------------------
# The text model
# ---------------------------------------------------------------------------


class CVRRTextModel(comfy.text_encoders.qwen3vl.Qwen3VL):
    """Qwen3-VL-8B decoder + vision tower with CVRR's recurrent visual path.

    Behaves exactly like the stock ComfyUI class until :meth:`configure_cvrr` is
    called; text-only prompts and disabled CVRR fall through to the parent
    implementation, so a single loader can serve both paths.
    """

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__(config_dict, dtype, device, operations)
        self.cvrr_options: Optional[CVRROptions] = None
        self.cvrr_core: Optional[CVRRTextEncoderCore] = None
        self.cvrr_transition: Optional[MergedTransition] = None
        self.last_cvrr_result: Optional[CVRREncodeResult] = None

    # -- guards ------------------------------------------------------------

    def preprocess_embed(self, embed, device):
        """Refuse image encodes on a vision-less (text-only export) encoder.

        A checkpoint shipped without the vision tower (``--drop-vision``, or an
        LM-only safetensors file) leaves the visual modules' parameters on the
        meta device forever (ComfyUI stages only parameters it actually has).
        The first image encode then dies deep inside the vision forward with
        ``NotImplementedError: Cannot copy out of meta tensor`` -- catch it here
        with a message that says what to do.  Text-only encodes are untouched.
        """
        if isinstance(embed, dict) and embed.get("type") == "image":
            visual = getattr(self, "visual", None)
            if visual is None or any(p.is_meta for p in visual.parameters()):
                raise RuntimeError(
                    "comfyui_cvrr: this text encoder has no usable vision tower "
                    "(its visual.* weights are missing -- e.g. a text-only or "
                    "--drop-vision export), so it cannot process images. Load the "
                    "full Qwen3-VL(-8B) encoder file (the converted CVRR backbone "
                    "or a vision-complete finetune). Text-only prompts keep working."
                )
        return super().preprocess_embed(embed, device)

    # -- configuration ----------------------------------------------------

    def configure_cvrr(
        self,
        spec: Optional[CVRRSpec] = None,
        mode: str = "aligned",
        tap_gains: Sequence[float] = (),
        block_visual_access: bool = False,
        enabled: bool = True,
    ) -> CVRROptions:
        spec = spec or RELEASE_SPEC
        spec.validate(self.num_layers)
        options = CVRROptions(
            spec=spec,
            mode=mode,
            enabled=enabled,
            tap_gains=tuple(tap_gains),
            block_visual_access=block_visual_access,
        )
        self.cvrr_options = options
        self.cvrr_core = CVRRTextEncoderCore(
            self.model,
            spec,
            transition=self.cvrr_transition,
            attention_op=_ATTENTION,
        )
        return options

    def attach_transition(self, state_dict, spec: Optional[CVRRSpec] = None) -> MergedTransition:
        """Install the merged fp32 transition weights on the recurrent layer."""
        spec = spec or (self.cvrr_options.spec if self.cvrr_options else RELEASE_SPEC)
        spec.validate(self.num_layers)
        weights = merged_transition_keys(state_dict)
        self.cvrr_transition = install_merged_transition(
            self.model.layers[spec.recurrent_layer], weights
        )
        if self.cvrr_core is not None:
            self.cvrr_core.transition = self.cvrr_transition
        return self.cvrr_transition

    @property
    def cvrr_ready(self) -> bool:
        """True when the recurrent path can actually run (weights present)."""
        return (
            self.cvrr_options is not None
            and self.cvrr_options.enabled
            and self.cvrr_core is not None
            and self.cvrr_transition is not None
        )

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        input_ids,
        attention_mask=None,
        embeds=None,
        num_tokens=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
        embeds_info=[],
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        visual_pos_masks = kwargs.pop("visual_pos_masks", None)
        deepstack_embeds = kwargs.pop("deepstack_embeds", None)

        if embeds is not None and position_ids is None:
            position_ids, visual_pos_masks, deepstack_embeds = self.build_image_inputs(
                embeds, embeds_info
            )

        has_image = visual_pos_masks is not None and bool(visual_pos_masks.any())
        if not (self.cvrr_ready and embeds is not None and has_image):
            # Stock behaviour: text-only prompts, plain multimodal encoding, or
            # CVRR switched off.
            return super().forward(
                input_ids,
                attention_mask=attention_mask,
                embeds=embeds,
                num_tokens=num_tokens,
                intermediate_output=intermediate_output,
                final_layer_norm_intermediate=final_layer_norm_intermediate,
                dtype=dtype,
                embeds_info=embeds_info,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_embeds=deepstack_embeds,
                **kwargs,
            )

        options = self.cvrr_options
        result = self.cvrr_core.encode(
            # ComfyUI's decoder layers write their output back into the input
            # tensor, and CVRR runs several passes over this buffer, so work on a
            # copy rather than on the caller's embeddings.
            mm_embeds=embeds.to(torch.float32).clone(),
            mm_position_ids=position_ids,
            mm_attention_mask=attention_mask,
            visual_mask=visual_pos_masks,
            deepstack=deepstack_embeds,
            taps=options.spec.taps,
            mode=options.mode,
            block_visual_access=options.block_visual_access,
        )
        gains = options.resolved_gains()
        taps = [tap * gain for tap, gain in zip(result.taps, gains)]
        stacked = torch.stack(taps, dim=1)  # [B, n_taps, L, D], ComfyUI's list-tap layout
        self.last_cvrr_result = result
        return None, stacked, None, None


def make_cvrr_qwen3vl(model_type: str):
    """Factory matching ``comfy.text_encoders.qwen3vl._make_qwen3vl_model``."""

    class CVRRQwen3VL_(CVRRTextModel):
        pass

    CVRRQwen3VL_.model_type = model_type
    return CVRRQwen3VL_


class CVRRQwen3VLClipModel(sd1_clip.SDClipModel):
    """``Qwen3VLClipModel`` variant that builds :class:`CVRRTextModel`.

    Mirrors ``comfy.text_encoders.qwen3vl.Qwen3VLClipModel.__init__`` but injects
    the CVRR-capable model class.  (The parent hard-codes its own factory, so the
    constructor is repeated here; ``tests/test_comfy_integration.py`` exercises it
    against the real ComfyUI source.)
    """

    def __init__(self, device="cpu", layer=None, layer_idx=None, dtype=None,
                 attention_mask=True, model_options={}, model_type="qwen3vl_8b",
                 taps: Sequence[int] = KLEIN_9B_TAPS):
        if layer is None:
            layer = list(taps)
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config={},
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=make_cvrr_qwen3vl(model_type),
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )

    @property
    def text_model(self) -> CVRRTextModel:
        return self.transformer

    def configure_cvrr(self, **kwargs) -> CVRROptions:
        return self.transformer.configure_cvrr(**kwargs)

    def attach_transition(self, state_dict, spec: Optional[CVRRSpec] = None):
        return self.transformer.attach_transition(state_dict, spec=spec)

    def encode_token_weights(self, token_weight_pairs):
        """Repair the reported attention mask for the CVRR token layout.

        ``SDClipModel.forward`` reports the mask of the *tokenised* sequence, but
        in ``strict`` / ``aligned`` mode CVRR emits one row per text token and the
        image tokens are gone, so that mask would not match the conditioning the
        diffusion model receives.
        """
        self.transformer.last_cvrr_result = None
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        result = self.transformer.last_cvrr_result
        if result is not None and self.return_attention_masks:
            mask = result.attention_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            extra = dict(extra or {})
            extra["attention_mask"] = mask.to(torch.long)
        return out, pooled, extra

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p,
                 repetition_penalty, seed, presence_penalty=0.0):
        # Text generation is not part of the diffusion path; keep the parent
        # behaviour by delegating through the stock implementation.
        return super().generate(
            tokens, do_sample, max_length, temperature, top_k, top_p, min_p,
            repetition_penalty, seed, presence_penalty=presence_penalty,
        )


class CVRRTE(comfy.text_encoders.flux.Flux2TEModel):
    """FLUX.2 Klein text encoder backed by the CVRR Qwen3-VL model.

    ``Flux2TEModel.encode_token_weights`` stacks the taps into Klein's
    ``[B, tokens, 3 * 4096]`` layout; this subclass additionally reports the
    *emitted* token mask, because CVRR conditions on a different number of tokens
    than the tokenizer produced (image tokens are dropped in ``strict`` /
    ``aligned`` mode).
    """

    def __init__(self, device="cpu", dtype=None, model_options={},
                 model_type="qwen3vl_8b", name: Optional[str] = None,
                 taps: Sequence[int] = KLEIN_9B_TAPS):
        # ``name`` doubles as the tokenizer key (``SD1ClipModel.clip_name``), so it
        # has to match the tokenizer's ``model_type``.
        clip_model = lambda **kw: CVRRQwen3VLClipModel(**kw, model_type=model_type, taps=taps)
        super().__init__(device=device, dtype=dtype, model_options=model_options,
                         name=name or model_type, clip_model=clip_model)

    # -- passthroughs ------------------------------------------------------

    @property
    def clip_model(self) -> CVRRQwen3VLClipModel:
        return getattr(self, self.clip)

    @property
    def text_model(self) -> CVRRTextModel:
        return self.clip_model.text_model

    def configure_cvrr(self, **kwargs) -> CVRROptions:
        return self.clip_model.configure_cvrr(**kwargs)

    def attach_transition(self, state_dict, spec: Optional[CVRRSpec] = None):
        return self.clip_model.attach_transition(state_dict, spec=spec)

    def set_cvrr_options(self, options: Optional[CVRROptions]) -> None:
        """Swap the whole option set for the next encode (``None`` = stock path)."""
        self.text_model.cvrr_options = options

    # -- encode -----------------------------------------------------------

    def encode_token_weights(self, token_weight_pairs):
        self.text_model.last_cvrr_result = None
        # Only enforce the no-per-token-weights rule while a CVRR encode is
        # configured.  Encoder stacks retrofitted via ``apply_transition`` are
        # re-typed to this class but shared with non-CVRR consumers (e.g. a
        # stock CLIP Text Encode using weighted prompts), and those must keep
        # working when no CVRR encode is in flight.
        options = getattr(self.text_model, "cvrr_options", None)
        if options is not None and options.enabled:
            pairs = token_weight_pairs[self.clip_name]
            for token in pairs:
                for entry in token:
                    if len(entry) > 1 and entry[1] != 1.0:
                        raise ValueError(
                            "comfyui_cvrr: per-token prompt weights are not supported by the "
                            "CVRR encoder because the emitted token count differs from the "
                            "tokenised prompt (image tokens are removed)."
                        )
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)

        result = self.text_model.last_cvrr_result
        if result is not None:
            extra = dict(extra or {})
            mask = result.attention_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            extra["attention_mask"] = mask.to(torch.long)
        return out, pooled, extra


# ---------------------------------------------------------------------------
# Loader helpers (used by nodes.py)
# ---------------------------------------------------------------------------


def build_clip(
    clip_path: str,
    embedding_directory=None,
    device: str = "default",
    spec: Optional[CVRRSpec] = None,
    mode: str = "aligned",
    transition_path: Optional[str] = None,
    model_type: str = "qwen3vl_8b",
):
    """Load a converted CVRR text encoder into a ComfyUI ``CLIP`` object.

    Mirrors the single-file branch of ``comfy.sd.load_text_encoder_state_dicts``
    but selects :class:`CVRRTE` instead of the stock Klein encoder.
    """
    import comfy.sd
    import comfy.supported_models_base
    import comfy.utils

    spec = spec or RELEASE_SPEC
    state_dict, _metadata = comfy.utils.load_torch_file(
        clip_path, safe_load=True, return_metadata=True
    )

    tokenizer_class = comfy.text_encoders.qwen3vl.tokenizer(model_type=model_type)
    target = comfy.supported_models_base.ClipTarget(tokenizer_class, _make_te_factory(spec, model_type))
    model_options = {}
    if device == "cpu":
        cpu = torch.device("cpu")
        model_options["load_device"] = model_options["offload_device"] = cpu

    clip = comfy.sd.CLIP(
        target,
        embedding_directory=embedding_directory,
        parameters=comfy.utils.calculate_parameters(state_dict),
        state_dict=[state_dict],
        model_options=model_options,
    )
    te = clip.cond_stage_model
    if not isinstance(te, CVRRTE):
        raise RuntimeError(f"comfyui_cvrr: unexpected text encoder class {type(te)!r}")

    # Vision coverage: a --drop-vision / LM-only export leaves the visual
    # modules on meta device and only explodes at the first image encode, so
    # flag it here (warning + marker for the loader's info string).
    te.cvrr_no_vision = not any("visual." in key for key in state_dict)
    if te.cvrr_no_vision:
        log.warning(
            "comfyui_cvrr: %s contains no vision tower weights (no visual.* "
            "keys); the CLIP is text-only and image encodes will refuse to run. "
            "Convert the full Qwen3-VL backbone (without --drop-vision).",
            clip_path,
        )

    te.configure_cvrr(spec=spec, mode=mode)
    if transition_path:
        transition_sd = comfy.utils.load_torch_file(transition_path, safe_load=True)
        te.attach_transition(transition_sd, spec=spec)
        te.configure_cvrr(spec=spec, mode=mode)
    return clip


def _make_te_factory(spec: CVRRSpec, model_type: str):
    class CVRRTE_(CVRRTE):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            super().__init__(device=device, dtype=dtype, model_options=model_options,
                             model_type=model_type, taps=spec.taps)

    return CVRRTE_


def _cvrr_state_defaults(text_model) -> None:
    """Instance attributes ``CVRRTextModel.__init__`` would have set."""
    if not hasattr(text_model, "cvrr_options"):
        text_model.cvrr_options: Optional[CVRROptions] = None
        text_model.cvrr_core: Optional[CVRRTextEncoderCore] = None
        text_model.cvrr_transition: Optional[MergedTransition] = None
        text_model.last_cvrr_result: Optional[CVRREncodeResult] = None


def ensure_cvrr_module(clip, spec: Optional[CVRRSpec] = None):
    """Upgrade a *stock* Qwen3-VL text encoder CLIP to the CVRR classes in place.

    This is the counterpart of :func:`build_clip`: instead of loading a
    dedicated file, an existing ``Flux2TEModel`` + ``Qwen3VLClipModel`` +
    ``Qwen3VL`` stack (stock ``CLIPLoader`` output, or any finetune that kept
    the architecture) is re-typed to the CVRR subclasses.  All attributes the
    subclasses rely on are plain ``__dict__`` entries, so a ``__class__`` swap
    plus defaults is sufficient.

    Returns the cond-stage text encoder object.  Raises a descriptive
    ``ValueError`` for encoders that cannot host CVRR (non-Qwen3-VL
    architectures, or the *text-only* Qwen3 encoders such as Klein's stock
    ``qwen_3_8b``: CVRR needs the vision tower).
    """
    spec = spec or RELEASE_SPEC
    te = getattr(clip, "cond_stage_model", None)
    if te is None:
        raise ValueError("comfyui_cvrr: object has no cond_stage_model; expected a ComfyUI CLIP")
    if isinstance(te, CVRRTE):  # built by build_clip(); already CVRR-capable
        return te
    model = getattr(te, getattr(te, "clip", ""), None)
    if model is None or not hasattr(model, "transformer"):
        raise ValueError(
            "comfyui_cvrr: unsupported text encoder stack "
            f"({type(te).__name__}); CVRR needs a Qwen3-VL (Flux2/Klein-style) "
            "text encoder CLIP"
        )
    text_model = model.transformer
    if not hasattr(text_model, "model") or not hasattr(text_model, "num_layers"):
        raise ValueError(
            "comfyui_cvrr: unsupported text encoder "
            f"({type(text_model).__name__}); CVRR needs the Qwen3-VL decoder stack"
        )
    if not hasattr(text_model, "visual"):
        raise ValueError(
            "comfyui_cvrr: CVRR needs the Qwen3-*VL* architecture with a vision "
            "tower; this CLIP is a text-only Qwen3 encoder (e.g. Klein's stock "
            "qwen_3_8b). Load a Qwen3-VL-8B-based file (the converted CVRR "
            "release, or any Qwen3-VL-8B finetune) with CLIPLoader type 'flux2' "
            "and attach the transition again."
        )
    spec.validate(text_model.num_layers)
    text_model.__class__ = CVRRTextModel
    _cvrr_state_defaults(text_model)
    model.__class__ = CVRRQwen3VLClipModel
    te.__class__ = CVRRTE
    if isinstance(getattr(model, "layer", None), (list, tuple)):
        # keep the stock fallback path's intermediate taps aligned with the spec
        model.layer = list(spec.taps)
    return te


def apply_transition(clip, transition_state_dict, spec: Optional[CVRRSpec] = None):
    """Install merged transition weights on any Qwen3-VL CLIP, LoRA-style.

    The 772 MB ``merged_transition.safetensors`` is the *only* CVRR-specific
    weight artifact, so it can be attached to any Qwen3-VL-8B-family encoder
    (the converted CVRR release, or a finetune with the same shapes) instead of
    shipping a merged checkpoint per finetune.  The weights live as fp32
    non-persistent buffers wrapped around the recurrent layer's seven
    projections (:class:`~comfyui_cvrr.cvrr_core.MergedProjection`) and stay
    disabled unless an encode is explicitly configured, so attaching them does
    not change stock (non-CVRR) encodes of the same encoder.

    Note: the underlying encoder object is shared between CLIP handles cloned
    from the same loader output, so attaching a *different* transition file
    replaces the previous one for all handles (same caveat as LoRA-relevant
    mutations of a cached model; attaching the same file twice is a refresh).
    """
    te = ensure_cvrr_module(clip, spec)
    model = te.clip_model
    model.attach_transition(transition_state_dict, spec=spec)
    if model.text_model.cvrr_options is not None:
        model.text_model.configure_cvrr(spec=spec or model.text_model.cvrr_options.spec,
                                        mode=model.text_model.cvrr_options.mode)
    return te


def is_cvrr_clip(clip) -> bool:
    return isinstance(getattr(clip, "cond_stage_model", None), CVRRTE)
