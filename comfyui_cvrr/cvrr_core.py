"""CVRR (Causal Visual Recurrent Reasoning) implemented on the *text-encoder* side.

This module is deliberately framework-agnostic: it only needs an object that
behaves like ComfyUI's ``comfy.text_encoders.llama.Llama2_`` text model, i.e. it
exposes

* ``text_model.layers``      -- ``ModuleList`` of decoder blocks callable as
                                ``layer(x=..., attention_mask=..., freqs_cis=...,
                                         optimized_attention=..., past_key_value=None)
                                -> (x, present_key_value)``
* ``text_model.compute_freqs_cis(position_ids, device) -> freqs_cis``
* ``text_model.norm``        -- optional final RMSNorm

That is exactly the interface ComfyUI's Qwen3-VL / Qwen3 text encoders use, so
the driver below runs unchanged on real ComfyUI modules and on the tiny
synthetic models used by the test-suite.

Why the algorithm looks like this
---------------------------------

CVRR (dmis-lab, "Reason Through the Latent!") splits a VLM decoder into three
sections given ``ell_star``:

    layers 0 .. ell_star          frozen lower branch ("native multimodal
                                  initialization")
    layer  ell_star + 1           the *shared recurrent transition*
    layers ell_star + 2 .. N-1    the upper ("answer") decoder

For the released ``Qwen3-VL-8B-CVRR`` checkpoint ``ell_star = 22`` on a
36-layer decoder, so the recurrent layer is 23 and the upper decoder is
24..35.

The reference inference procedure (see ``modeling_cvrr_merged.py`` and
``cvrr/modeling_cvrr.py`` upstream) is:

1. **Multimodal branch.** Run image + text tokens through layers ``0..ell_star``
   to obtain the "persistent visual scaffold".
2. **Text-only branch.** Run the same prompt *without* image tokens through
   ``0..ell_star`` and then once (with the recurrent adapters disabled) through
   layer ``ell_star + 1``.  This is the context the upper decoder may attend to.
3. **Initial question state.** One pass of layer ``ell_star + 1`` over the
   multimodal scaffold, adapters *disabled*; keep only the non-visual ("question")
   rows.
4. **Recurrence.** ``T - 1`` times: rebuild the scaffold with the current
   question state written into the question rows, evaluate layer
   ``ell_star + 1`` with the *merged* transition weights enabled, and take
   ``state <- lerp(state, proposal, beta)``.
5. **Strict interface.** Feed the final question state into layers
   ``ell_star + 2 .. N-1`` on the *text-only* sequence.  Visual rows never reach
   the upper decoder.

Everything below reproduces 1-5 for the case that matters to a diffusion model:
we do not generate tokens, we only need the *hidden states* that come out of the
upper decoder, because a text encoder's job is to hand per-token embeddings to
the diffusion transformer.

Two output modes are provided:

``strict``  (CVRR as released)
    The emitted token sequence has exactly the same token layout as the
    text-only prompt.  It is a drop-in replacement for a normal text-encoder
    output, so it can be fed straight into a model whose text conditioning is
    ``[B, tokens, 3 * 4096]`` (FLUX.2 Klein taps layers 9/18/27 of Qwen3-8B).

``vl``      (CVRR with the visual barrier removed)
    The visual rows stay in the sequence, so the emitted token sequence contains
    the image tokens too.  This is the "force a VLM into a non-VL diffusion
    model" variant: the diffusion transformer simply sees more context tokens.

The implementation is intentionally written as full-sequence layer calls (rather
than CVRR's question-rows-only optimisation).  ``tests/test_cvrr_core.py``
verifies numerically that the two are algebraically equivalent, which is also
what the released code claims.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "CVRRSpec",
    "MergedProjection",
    "MergedTransition",
    "install_merged_transition",
    "merged_transition_keys",
    "MERGED_PROJECTION_PATHS",
    "CVRREncodeResult",
    "CVRRTextEncoderCore",
    "klein_stack_taps",
    "sdpa_attention",
]


# ---------------------------------------------------------------------------
# Release geometry
# ---------------------------------------------------------------------------

#: Projections that the released ``merged_transition.safetensors`` file replaces.
MERGED_PROJECTION_PATHS: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

#: Layer taps FLUX.2 Klein 9B reads out of its Qwen3-8B text encoder.
KLEIN_9B_TAPS: Tuple[int, ...] = (9, 18, 27)


@dataclass(frozen=True)
class CVRRSpec:
    """Hyper-parameters of a released CVRR checkpoint.

    ``ell_star`` is the *inclusive* index of the last frozen lower layer, exactly
    as in the release metadata.  For ``Qwen3-VL-8B-CVRR``:
    ``ell_star=22``, ``recurrent_layer=23``, ``upper_decoder_start=24``.
    """

    ell_star: int = 22
    num_recurrent_steps: int = 4
    beta: float = 0.33
    taps: Tuple[int, ...] = KLEIN_9B_TAPS
    #: Name of the released model this spec was read from (informational).
    name: str = "Qwen3-VL-8B-CVRR"

    @property
    def recurrent_layer(self) -> int:
        return self.ell_star + 1

    @property
    def upper_decoder_start(self) -> int:
        return self.ell_star + 2

    @classmethod
    def from_release(cls, release: Mapping[str, Any], taps: Sequence[int] = KLEIN_9B_TAPS) -> "CVRRSpec":
        """Build a spec from the ``release`` block of a CVRR ``config.json``."""
        ell_star = int(release.get("ell_star", 22))
        recurrent = release.get("recurrent_layer")
        if recurrent is not None and int(recurrent) != ell_star + 1:
            raise ValueError(
                "inconsistent release metadata: recurrent_layer != ell_star + 1 "
                f"({recurrent} != {ell_star + 1})"
            )
        return cls(
            ell_star=ell_star,
            num_recurrent_steps=int(release.get("inference_T", 4)),
            beta=float(release.get("inference_beta", 0.33)),
            taps=tuple(int(t) for t in taps),
            name=str(release.get("name", "unknown-cvrr")),
        )

    def validate(self, num_layers: int) -> None:
        if not 0 <= self.ell_star < num_layers - 1:
            raise ValueError(
                f"ell_star={self.ell_star} is out of range for {num_layers} decoder layers"
            )
        if self.num_recurrent_steps < 1:
            raise ValueError("num_recurrent_steps must be >= 1")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        for tap in self.taps:
            if not 0 <= tap <= num_layers:
                raise ValueError(f"tap layer {tap} outside [0, {num_layers}]")


# ---------------------------------------------------------------------------
# Merged recurrent transition
# ---------------------------------------------------------------------------


class MergedProjection(nn.Module):
    """A native projection plus a switchable merged (base + LoRA) weight.

    Mirrors ``NativeMergedLinear`` from the released ``modeling_cvrr_merged.py``:
    the merged weight is kept in float32 and never truncated by a cast of the
    surrounding module.

    The wrapped ``base`` may be *any* module with a ``forward`` that computes
    ``x @ W.T`` for a logical weight of the merged weight's shape, in whatever
    storage format it likes -- plain ``nn.Linear`` (fp32/bf16/fp16), ComfyUI's
    ``fp8_ops``/``manual_cast`` Linears, or the mixed-precision quant modules
    whose ``weight`` is a ``QuantizedTensor`` (``float8_e4m3fn``/``e5m2``,
    ``int8_tensorwise`` with or without convrot, NVFP4, ...).  ``QuantizedTensor``
    subclasses report the **logical** shape via ``.shape``, so the check below
    sees through them.
    """

    def __init__(self, base: nn.Module, merged_weight: torch.Tensor):
        super().__init__()
        weight = getattr(base, "weight", None)
        if weight is None:
            quantized = getattr(base, "quant_format", None)
            if quantized is not None:
                raise ValueError(
                    "merged projection on a quantized layer is not possible yet: the "
                    f"{quantized}-quantized weights have not finished loading "
                    "(module has no .weight). Load/prepare the base checkpoint "
                    "before attaching merged_transition.safetensors."
                )
            # Weightless module (e.g. tensors still streaming in): validate
            # against the fused geometry if it is discoverable.
            in_f = getattr(base, "in_features", None)
            out_f = getattr(base, "out_features", None)
            if in_f is not None and out_f is not None and (
                (int(in_f), int(out_f)) != (int(merged_weight.shape[1]), int(merged_weight.shape[0]))
            ):
                raise ValueError(
                    "merged projection shape mismatch: base declares "
                    f"{out_f}x{in_f} but merged weight is {tuple(merged_weight.shape)}"
                )
        elif tuple(weight.shape) != tuple(merged_weight.shape):
            raise ValueError(
                "merged projection shape mismatch: "
                f"{tuple(weight.shape)} (storage dtype {weight.dtype}) != "
                f"{tuple(merged_weight.shape)}"
            )
        self.base = base
        self.enabled = False
        self.register_buffer(
            "merged_weight", merged_weight.detach().to(dtype=torch.float32).clone(), persistent=False
        )

    def _apply(self, fn, recurse=True):  # noqa: D102
        # A caller's dtype cast must not truncate the fp32 merged weights (and
        # then merely upcast already-lost precision).  Mirrors the released
        # ``NativeMergedLinear._apply``.
        original = self.merged_weight
        super()._apply(fn, recurse=recurse)
        self.merged_weight = original.to(device=self.merged_weight.device, dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if not self.enabled:
            return self.base(x)
        bias = None if self.base.bias is None else self.base.bias.float()
        return torch.nn.functional.linear(x.float(), self.merged_weight, bias).to(x.dtype)

    def extra_repr(self) -> str:  # noqa: D102
        return f"enabled={self.enabled}, shape={tuple(self.merged_weight.shape)}"


def merged_transition_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalise a ``merged_transition.safetensors`` mapping.

    The released file stores ``self_attn.q_proj.weight`` (and 6 more) as float32.
    Some conversions name the tensors ``...weight.merged`` or ``...merged_weight``;
    all variants are accepted and normalised to ``<path>.weight``.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        path = key
        for suffix in (".merged_weight", ".merged", "_merged"):
            if path.endswith(suffix):
                path = path[: -len(suffix)] + ".weight"
                break
        if not path.endswith(".weight"):
            raise ValueError(f"unexpected key in merged transition file: {key!r}")
        out[path] = value
    missing = [f"{p}.weight" for p in MERGED_PROJECTION_PATHS if f"{p}.weight" not in out]
    if missing:
        raise ValueError(f"merged transition file is incomplete, missing: {missing}")
    return out


class MergedTransition:
    """Handle for switching the merged recurrent projections on and off."""

    def __init__(self, layer: nn.Module, projections: Sequence[MergedProjection]):
        self.layer = layer
        self.projections = list(projections)

    @property
    def enabled(self) -> bool:
        return all(p.enabled for p in self.projections)

    @contextlib.contextmanager
    def active(self, enabled: bool = True):
        previous = [p.enabled for p in self.projections]
        for projection in self.projections:
            projection.enabled = bool(enabled)
        try:
            yield self
        finally:
            for projection, was in zip(self.projections, previous):
                projection.enabled = was

    def set(self, enabled: bool) -> None:
        for projection in self.projections:
            projection.enabled = bool(enabled)


def install_merged_transition(
    layer: nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> MergedTransition:
    """Replace the seven projections of ``layer`` with :class:`MergedProjection`.

    ``layer`` is a ComfyUI ``TransformerBlock`` (``layer.self_attn.q_proj``,
    ``layer.mlp.gate_proj``, ...); the released file uses the same sub-paths.
    """
    weights = merged_transition_keys(state_dict)
    resolved = []
    projections = []
    for path in MERGED_PROJECTION_PATHS:
        parent = layer
        parts = path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        name = parts[-1]
        base = getattr(parent, name)
        resolved.append((parent, name, base, path))
        if isinstance(base, MergedProjection):  # already installed, refresh weights
            base.merged_weight.copy_(weights[f"{path}.weight"].float())
            projections.append(base)
            continue
        try:
            wrapped = MergedProjection(base, weights[f"{path}.weight"])
        except ValueError as exc:
            raise ValueError(f"merged transition unusable at {path!r}: {exc}") from exc
        projections.append(wrapped)
    # all seven validated; only now rewire the modules (atomic install)
    for (parent, name, base, path), wrapped in zip(resolved, projections):
        if base is not wrapped:
            setattr(parent, name, wrapped)
    return MergedTransition(layer, projections)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None,
    skip_reshape: bool = False,
    enable_gqa: bool = False,
    **_kwargs: Any,
) -> torch.Tensor:
    """Minimal stand-in for ``comfy.ldm.modules.attention.optimized_attention``.

    Used by the test-suite and as a fallback when ComfyUI is not importable.
    ComfyUI's blocks call attention with ``skip_reshape=True`` and tensors shaped
    ``[B, H, L, D]``.
    """
    if not skip_reshape:
        b, l, _ = q.shape
        q = q.view(b, l, heads, -1).transpose(1, 2)
        k = k.view(b, k.shape[1], heads, -1).transpose(1, 2)
        v = v.view(b, v.shape[1], heads, -1).transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=enable_gqa
    )


def text_row_mask(visual_mask: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor],
                  batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[B, L]`` mask of "question" rows: non-visual and non-padding."""
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    if visual_mask is not None:
        mask &= ~visual_mask.bool()
    if attention_mask is not None:
        mask &= attention_mask.bool().reshape(batch_size, -1)[:, :seq_len]
    return mask


def gather_rows(x: torch.Tensor, row_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad ``x`` rows selected by ``row_mask`` into a dense ``[B, W, D]`` batch."""
    counts = row_mask.sum(dim=1)
    width = int(counts.max()) if counts.numel() else 0
    out = x.new_zeros((x.shape[0], max(width, 1), x.shape[-1]))
    padding = torch.ones((x.shape[0], max(width, 1)), dtype=torch.bool, device=x.device)
    for row, count in enumerate(counts.tolist()):
        if count:
            out[row, :count] = x[row][row_mask[row]]
            padding[row, :count] = False
    return out, padding


def scatter_rows(full: torch.Tensor, row_mask: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`gather_rows` (``rows`` may carry padding on the right)."""
    out = full.clone()
    for row in range(full.shape[0]):
        count = int(row_mask[row].sum())
        if count:
            out[row][row_mask[row]] = rows[row, :count]
    return out


def _causal_additive_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(seq_len, seq_len, dtype=dtype, device=device).fill_(
        torch.finfo(dtype).min / 4
    ).triu_(1)


def visual_key_block_bias(
    question_mask: torch.Tensor, visual_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Additive mask blocking visual rows as *keys* for question-row queries.

    Implements CVRR's ``block_visual_access`` ablation exactly: outputs at every
    row are still computed, but the recurrent transition can no longer read the
    persistent visual rows.
    """
    neg = torch.finfo(dtype).min / 4
    block = question_mask[:, None, :, None] & visual_mask[:, None, None, :]
    return torch.zeros(block.shape, dtype=dtype, device=block.device).masked_fill(block, neg)


def build_layer_mask(
    x: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Reproduce ComfyUI's additive attention mask (padding + causal), ``[B,1,L,L]``."""
    seq_len = x.shape[1]
    if seq_len <= 1:
        return None
    causal = _causal_additive_mask(seq_len, x.dtype, x.device)
    if attention_mask is None:
        return causal
    mask = 1.0 - attention_mask.to(x.dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
    )[:, :, :seq_len, :seq_len].expand(x.shape[0], 1, seq_len, seq_len)
    mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)
    return mask + causal


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclass
class CVRREncodeResult:
    """Hidden states produced by :meth:`CVRRTextEncoderCore.encode`."""

    #: One tensor ``[B, L_out, D]`` per requested tap layer, in the order given.
    taps: Tuple[torch.Tensor, ...]
    #: Attention mask matching ``taps`` ([B, L_out]); all-ones when unpadded.
    attention_mask: torch.Tensor
    #: Which rows of the multimodal sequence the emitted tokens correspond to
    #: (``strict`` mode: the question rows; ``vl`` mode: everything).
    row_mask: torch.Tensor
    #: Final recurrent question state ``[B, Lq, D]`` (zeroed on padding).
    state: Optional[torch.Tensor] = None
    #: Diagnostics (recurrent state norms, whether visual access was blocked...).
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return int(self.taps[0].shape[1])

    def stacked(self) -> torch.Tensor:
        """``[B, L, len(taps) * D]`` -- the layout FLUX.2 Klein's TE consumes."""
        return torch.cat(self.taps, dim=-1)


class CVRRTextEncoderCore:
    """Runs CVRR over a ComfyUI-style decoder stack and returns layer taps.

    Parameters
    ----------
    text_model:
        ComfyUI ``Llama2_`` instance (``Qwen3VL.model`` / ``Qwen3_8B.model``).
    spec:
        Release geometry (:class:`CVRRSpec`).
    transition:
        Optional :class:`MergedTransition` installed on the recurrent layer.
        When omitted the recurrent layer runs with its native weights, which is
        the correct behaviour only for debugging / ablation.
    attention_op:
        Attention implementation handed to the blocks.  Defaults to
        :func:`sdpa_attention`.
    """

    def __init__(
        self,
        text_model: nn.Module,
        spec: CVRRSpec,
        transition: Optional[MergedTransition] = None,
        attention_op: Optional[Callable[..., torch.Tensor]] = None,
    ):
        self.text_model = text_model
        self.spec = spec
        self.transition = transition
        self.attention_op = attention_op or sdpa_attention

    # -- plumbing ---------------------------------------------------------

    @property
    def _layers(self) -> nn.ModuleList:
        return self.text_model.layers

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def _freqs(self, position_ids: torch.Tensor, device: torch.device) -> Any:
        return self.text_model.compute_freqs_cis(position_ids, device)

    def _run_layers(
        self,
        x: torch.Tensor,
        start: int,
        stop: int,
        *,
        freqs: Any,
        attention_mask: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        deepstack: Optional[Sequence[torch.Tensor]] = None,
        visual_mask: Optional[torch.Tensor] = None,
        capture: Iterable[int] = (),
        taps: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run ``layers[start:stop]``, mirroring ComfyUI's ``Llama2_.forward``."""
        capture = set(capture)
        attention_op = self.attention_op
        for index in range(start, stop):
            if index in capture:
                if taps is None:
                    raise RuntimeError("capture requested without a tap dictionary")
                taps[index] = x.clone()
            layer = self._layers[index]
            x, _ = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs,
                optimized_attention=attention_op,
                past_key_value=None,
            )
            if deepstack is not None and index < len(deepstack) and visual_mask is not None:
                x = x.clone()
                x[visual_mask] = x[visual_mask] + deepstack[index].to(x)
        # "entering layer `stop`" is the state after the last executed layer.
        if stop in capture and taps is not None and stop not in taps:
            taps[stop] = x.clone()
        return x

    # -- main entry point --------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        *,
        mm_embeds: torch.Tensor,
        mm_position_ids: torch.Tensor,
        mm_attention_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        deepstack: Optional[Sequence[torch.Tensor]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        text_position_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        taps: Optional[Sequence[int]] = None,
        mode: str = "aligned",
        block_visual_access: bool = False,
    ) -> CVRREncodeResult:
        """Encode one batch through CVRR and return the requested layer taps.

        ``mm_*`` describe the image-conditioned sequence (image tokens already
        spliced in); ``text_*`` describe the same prompt with the image tokens
        removed.  When ``text_embeds`` is omitted it is derived from ``mm_embeds``
        by dropping the rows selected by ``visual_mask``, which is what
        guarantees the token alignment the reference implementation asserts.

        Modes
        -----
        ``strict``
            CVRR as released: every tap below the upper decoder is taken from the
            text-only branch.  Those taps therefore carry no image information at
            all -- only taps at or above ``ell_star + 2`` do.  For FLUX.2 Klein
            (taps 9/18/27, recurrent layer 23) that means 1 of 3 taps is
            image-aware.
        ``aligned`` (default)
            The same token layout as ``strict`` -- one output token per *text*
            token, so it stays a drop-in text-encoder replacement -- but every tap
            is read from the multimodal branch, so the whole conditioning is
            image-aware.  The strict answer-path barrier is what is dropped here;
            the recurrence is unchanged.
        ``vl``
            Keeps the visual rows in the output, i.e. the diffusion model sees the
            image tokens as extra context tokens.
        """
        if mode not in ("strict", "aligned", "vl"):
            raise ValueError(f"unknown mode {mode!r}")
        spec = self.spec
        spec.validate(self.num_layers)
        taps = tuple(int(t) for t in (taps if taps is not None else spec.taps))
        device = mm_embeds.device
        batch, seq_len, _ = mm_embeds.shape

        q_mask = text_row_mask(visual_mask, mm_attention_mask, batch, seq_len, device)
        if int(q_mask.sum(dim=1).min()) == 0:
            raise ValueError("no text (question) rows found in the multimodal sequence")

        # ---- 0. text-only branch inputs -------------------------------
        if text_embeds is None:
            text_embeds, text_padding = gather_rows(mm_embeds, q_mask)
            if text_attention_mask is None:
                text_attention_mask = (~text_padding).to(torch.long)
            if text_position_ids is None:
                # ComfyUI's M-RoPE position layout is (3, seq) for one sequence
                # (see ``qwen2vl_mrope_position_ids``); a text-only branch is just
                # sequential positions on all three axes.
                positions = torch.arange(text_embeds.shape[1], device=device)
                text_position_ids = positions.view(1, -1).expand(3, -1).contiguous()
        if text_attention_mask is None:
            text_attention_mask = torch.ones(
                text_embeds.shape[0], text_embeds.shape[1], dtype=torch.long, device=device
            )
        if text_position_ids is None:
            positions = torch.arange(text_embeds.shape[1], device=device)
            text_position_ids = positions.view(1, -1).expand(3, -1).contiguous()

        mm_mask = build_layer_mask(mm_embeds, mm_attention_mask)
        text_mask = build_layer_mask(text_embeds, text_attention_mask)
        mm_freqs = self._freqs(mm_position_ids, device)
        text_freqs = self._freqs(text_position_ids, device)

        collected: Dict[int, torch.Tensor] = {}
        lower_taps = tuple(t for t in taps if t < spec.upper_decoder_start)
        upper_taps = tuple(t for t in taps if t >= spec.upper_decoder_start)
        mm_taps = mode in ("aligned", "vl")

        # ---- 1. multimodal scaffold: layers 0 .. ell_star --------------
        scaffold = self._run_layers(
            mm_embeds,
            0,
            spec.recurrent_layer,
            freqs=mm_freqs,
            attention_mask=mm_attention_mask,
            mask=mm_mask,
            deepstack=deepstack,
            visual_mask=visual_mask,
            capture=lower_taps if mm_taps else (),
            taps=collected,
        )

        # ---- 2. text-only branch below the upper decoder ---------------
        # Only used by `strict`, and only for its taps: the upper decoder builds
        # its own K/V from the recurrent state (verified against the released
        # cache handling), so there is no reason to run this branch further than
        # the deepest requested tap.
        if not mm_taps and lower_taps:
            self._run_layers(
                text_embeds,
                0,
                max(lower_taps) + 1,
                freqs=text_freqs,
                attention_mask=text_attention_mask,
                mask=text_mask,
                capture=lower_taps,
                taps=collected,
            )

        # ---- 3. initial question state (adapters off) -------------------
        with (self.transition.active(False) if self.transition else contextlib.nullcontext()):
            first_full = self._run_layers(
                scaffold,
                spec.recurrent_layer,
                spec.recurrent_layer + 1,
                freqs=mm_freqs,
                attention_mask=mm_attention_mask,
                mask=mm_mask,
            )
        state, state_padding = gather_rows(first_full, q_mask)
        state = state.masked_fill(state_padding.unsqueeze(-1), 0).clone()

        stats: Dict[str, Any] = {
            "mode": mode,
            "recurrent_layer": spec.recurrent_layer,
            "upper_decoder_start": spec.upper_decoder_start,
            "num_recurrent_steps": spec.num_recurrent_steps,
            "beta": spec.beta,
            "merged_transition": self.transition is not None,
            "block_visual_access": bool(block_visual_access),
        }

        # ---- 4. recurrence ---------------------------------------------
        if spec.num_recurrent_steps > 1:
            if self.transition is None:
                raise RuntimeError(
                    "CVRR recurrence needs the merged transition weights; load "
                    "merged_transition.safetensors and pass it as `transition`."
                )
            recurrent_mask = mm_mask
            if block_visual_access:
                if visual_mask is None:
                    raise ValueError("block_visual_access requires visual_mask")
                recurrent_mask = visual_key_block_bias(q_mask, visual_mask, mm_embeds.dtype)
                recurrent_mask = recurrent_mask if mm_mask is None else recurrent_mask + mm_mask
            with self.transition.active(True):
                for step in range(1, spec.num_recurrent_steps):
                    scaffold_in = scatter_rows(first_full, q_mask, state)
                    proposal_full = self._run_layers(
                        scaffold_in,
                        spec.recurrent_layer,
                        spec.recurrent_layer + 1,
                        freqs=mm_freqs,
                        attention_mask=mm_attention_mask,
                        mask=recurrent_mask,
                    )
                    proposal, proposal_padding = gather_rows(proposal_full, q_mask)
                    state = torch.lerp(state, proposal, spec.beta)
                    state = state.masked_fill(proposal_padding.unsqueeze(-1), 0)
                    stats[f"state_norm_step_{step}"] = float(
                        state.float().norm(dim=-1).mean()
                    )

        stats["state_norm_final"] = float(state.float().norm(dim=-1).mean())

        # ---- 5. upper decoder ------------------------------------------
        if mode == "strict":
            if upper_taps:
                self._run_layers(
                    state,
                    spec.upper_decoder_start,
                    max(upper_taps) + 1,
                    freqs=text_freqs,
                    attention_mask=text_attention_mask,
                    mask=text_mask,
                    capture=upper_taps,
                    taps=collected,
                )
            row_mask = q_mask
            out_mask = text_attention_mask
        else:
            # The visual rows stay available to the question rows; the upper
            # decoder runs over the full multimodal sequence and only the
            # question rows are kept for `aligned`.
            upper_in = scatter_rows(first_full, q_mask, state)
            if upper_taps:
                self._run_layers(
                    upper_in,
                    spec.upper_decoder_start,
                    max(upper_taps) + 1,
                    freqs=mm_freqs,
                    attention_mask=mm_attention_mask,
                    mask=mm_mask,
                    deepstack=deepstack,
                    visual_mask=visual_mask,
                    capture=upper_taps,
                    taps=collected,
                )
            if mode == "vl":
                row_mask = torch.ones_like(q_mask)
                if mm_attention_mask is not None:
                    row_mask &= mm_attention_mask.bool()
                out_mask = (
                    mm_attention_mask
                    if mm_attention_mask is not None
                    else torch.ones(batch, seq_len, dtype=torch.long, device=device)
                )
            else:
                row_mask = q_mask
                out_mask = text_attention_mask

        ordered = []
        for tap in taps:
            if tap not in collected:
                raise RuntimeError(f"tap layer {tap} was not captured")
            value = collected[tap]
            if mode == "aligned":
                value, value_padding = gather_rows(value, q_mask)
                out_mask = (~value_padding).to(torch.long)
            ordered.append(value)

        return CVRREncodeResult(
            taps=tuple(ordered),
            attention_mask=out_mask,
            row_mask=row_mask,
            state=state,
            stats=stats,
        )


def klein_stack_taps(taps: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack taps the way ComfyUI's ``Flux2TEModel`` does (``[B, L, n*D]``)."""
    if len(taps) != 3:
        raise ValueError(f"FLUX.2 Klein expects 3 taps, got {len(taps)}")
    return torch.cat(taps, dim=-1)
