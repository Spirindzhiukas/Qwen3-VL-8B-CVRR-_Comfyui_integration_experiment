"""ComfyUI nodes for the CVRR Qwen3-VL text encoder.

Node set
--------
``CVRRTextEncoderLoader``   load a converted CVRR text encoder (+ optional merged
                            transition) with ComfyUI's own ``CLIP`` object
``CVRRApplyTransition``     LoRA-style attach of ``merged_transition.safetensors``
                            to *any* loaded Qwen3-VL CLIP (stock CLIPLoader output
                            or a finetune) -- no dedicated converted file needed
``CVRRTextEncode``          CLIP Text Encode with an optional reference image: the
                            conditioning is produced by CVRR's visual recurrence
``CVRRTextEncodePlain``     plain text-prompt encode (no image, no recurrence) for
                            probing VL-native conditioning on other models (e.g.
                            ideogram4's Qwen3-VL TE)
``CVRREditTextEncode``      convenience node in the spirit of community
                            "Flux2 Klein Edit Text Encode" nodes: prompt + image +
                            VAE -> CVRR conditioning *and* FLUX.2 reference latents
``CVRRSetReferenceLatent``  "Set Reference Latent+" that VAE-encodes the image for
                            you and can chain multiple references
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import folder_paths
import node_helpers

log = logging.getLogger("comfyui_cvrr")

from .cvrr_te import (
    RELEASE_CONFIG_NAME,
    RELEASE_SPEC,
    TRANSITION_FILE_NAME,
    CVRROptions,
    CVRRSpec,
    apply_transition,
    build_clip,
    is_cvrr_clip,
    load_release_spec,
)

CATEGORY = "model/conditioning/cvrr"
TEXT_ENCODER_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin")
MODE_OPTIONS = ["aligned", "strict", "vl"]

#: Longest-edge guard for images fed to the Qwen3-VL vision tower (patch16 x
#: merge2 => 32-px tokens).  2048 px ~= 4096 visual tokens on a square image;
#: the released prepare_inputs() allows up to 8192 tokens (~2896 px), but
#: vision-tower self-attention is quadratic, so the node caps lower by default.
VL_EDGE_CAP = 2048

MODE_TOOLTIP = (
    "CVRR's token layout: 'aligned' (default; only the highest image-aware tap is "
    "fused into the text rows), 'strict' (the release's causal interface), 'vl' "
    "(raw VL rows, image tokens kept)."
)
BLOCKSIZE_TOOLTIP = (
    "ell* (CVRR's visual-read boundary): the decoder layer index whose adapter-off "
    "pass anchors the recurrence. 0 = release value 22. Advanced; only change for "
    "retrained variants."
)
STEPS_TOOLTIP = (
    "T, the recurrence count: how many times CVRR's recurrence re-reads the prompt "
    "(nothing to do with KSampler/diffusion steps). 0 = release value 4. The "
    "adapter was trained at T=4; higher values change state statistics."
)
BETA_TOOLTIP = (
    "beta in state = (1-beta)*state + beta*proposal. -1 = release value 0.33. "
    "1.0 = always take the newest proposal, 0.0 = freeze the first-pass state."
)
REF_METHOD_TOOLTIP = (
    "How reference latents are concatenated by FLUX.2/Klein (same values as "
    "ComfyUI's reference-latent-method node); 'index' is the Klein/Kontext default."
)
TAPGAIN_TOOLTIP = (
    "Per-tap multipliers applied to the 3 stacked hidden-state taps, e.g. "
    "'1,1,1.2'. Scale-tunes the CVRR conditioning."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _encoder_files() -> List[str]:
    """Everything in ``models/text_encoders`` (recursively, subfolder-prefixed).

    Identical to the list the stock ``CLIPLoader`` combo shows, so the merged
    transition can live next to the encoder checkpoints under *any* file name,
    exactly like every other text-encoder-side asset ComfyUI loads.
    """
    try:
        return folder_paths.get_filename_list("text_encoders")
    except Exception:  # pragma: no cover - folder not registered
        return []


def _resolve_transition(name: str) -> str:
    """Resolve a combo selection the way ``CLIPLoader`` resolves ``clip_name``:
    inside ``models/text_encoders`` first, with a back-compat fallback to the
    other model folders for older layouts."""
    for folder in ("text_encoders", "diffusion_models", "unet", "clip"):
        path = folder_paths.get_full_path(folder, name)
        if path:
            return path
    raise ValueError(f"comfyui_cvrr: cannot find {name!r} in models/text_encoders")


def _spec_from_encoder_file(encoder_path: str, taps: Sequence[int]) -> CVRRSpec:
    """Look for CVRR metadata next to the encoder file."""
    directory = os.path.dirname(encoder_path)
    return load_release_spec(directory, taps=taps)


def _encode_with_options(clip, tokens, options: CVRROptions, enabled: bool = True):
    """Run ``clip.encode_from_tokens_scheduled`` under a CVRR option set."""
    model = clip.cond_stage_model
    if not is_cvrr_clip(clip):
        raise ValueError(
            "comfyui_cvrr: this CLIP is not CVRR-enabled. Either use the 'CVRR "
            "Qwen3-VL Text Encoder Loader' node with a converted CVRR file, or "
            "load any Qwen3-VL-8B CLIP (CLIPLoader type 'flux2') and run it "
            "through the 'CVRR Apply Transition' node with "
            "merged_transition.safetensors -- the latter is the LoRA-style attach "
            "path and works with Qwen3-VL-8B finetunes too."
        )
    clip_model = getattr(model, getattr(model, "clip", ""), None)
    text_model = getattr(clip_model, "transformer", None)
    previous = getattr(text_model, "cvrr_options", None)
    spec = previous.spec if previous is not None else options.spec
    text_model.configure_cvrr(
        spec=spec,
        mode=options.mode,
        tap_gains=options.tap_gains,
        block_visual_access=options.block_visual_access,
        enabled=enabled,
    )
    try:
        return clip.encode_from_tokens_scheduled(tokens)
    finally:
        if previous is None:
            text_model.cvrr_options = None
        else:
            text_model.configure_cvrr(
                spec=previous.spec,
                mode=previous.mode,
                tap_gains=previous.tap_gains,
                block_visual_access=previous.block_visual_access,
                enabled=previous.enabled,
            )


def _clamp_to_edge_cap(width: int, height: int, multiple: int = 32,
                       cap: int = VL_EDGE_CAP) -> Tuple[int, int]:
    """Shrink (width, height) so the longest edge never exceeds ``cap`` pixels."""
    longest = max(width, height)
    if longest <= cap:
        return width, height
    scale = cap / float(longest)
    return (max(multiple, int(math.floor(width * scale / multiple)) * multiple),
            max(multiple, int(math.floor(height * scale / multiple)) * multiple))


def _resize_for_vl(image: torch.Tensor, megapixels: float, multiple: int = 32) -> torch.Tensor:
    """Scale an image batch to roughly ``megapixels`` before feeding the VLM
    (never above the vision tower's safe longest edge)."""
    import comfy.utils

    if megapixels <= 0:
        return image
    samples = image.movedim(-1, 1)
    total = float(samples.shape[2] * samples.shape[3])
    target = max(1.0, megapixels * 1024.0 * 1024.0)
    scale = math.sqrt(target / total)
    width = max(multiple, int(round(samples.shape[3] * scale / multiple)) * multiple)
    height = max(multiple, int(round(samples.shape[2] * scale / multiple)) * multiple)
    width, height = _clamp_to_edge_cap(width, height, multiple)
    out = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return out.movedim(1, -1)[:, :, :, :3]


def _ref_target_size(image: torch.Tensor, target_edge: int,
                     multiple: int = 32) -> Tuple[int, int]:
    """Processed reference size for a target longest edge.

    Rule (user-facing): never upscale the input, never exceed the vision
    tower's safe edge, round down to a multiple of 32 (satisfies both the
    tower's 32-px tokens and Flux2 VAE's /16).
    """
    height, width = int(image.shape[1]), int(image.shape[2])
    natural = max(height, width)
    edge = max(multiple, (min(int(target_edge), natural, VL_EDGE_CAP) // multiple) * multiple)
    scale = edge / float(natural)
    width = max(multiple, int(round(width * scale / multiple)) * multiple)
    height = max(multiple, int(round(height * scale / multiple)) * multiple)
    width, height = _clamp_to_edge_cap(width, height, multiple)
    return width, height


def _resize_to_size(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import comfy.utils

    samples = image.movedim(-1, 1)
    out = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return out.movedim(1, -1)[:, :, :, :3]


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


class CVRRTextEncoderLoader:
    """Load a converted CVRR Qwen3-VL text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        encoders = _encoder_files()
        transitions = [""] + encoders  # name-agnostic, same list as CLIPLoader
        return {
            "required": {
                "clip_name": (encoders,),
                "type": (["qwen3vl_8b", "qwen3vl_4b", "qwen3vl_32b"],
                         {"tooltip": "Backbone family of the converted file: selects the "
                                     "tokenizer and the Qwen3-VL config (8B is the CVRR release)."}),
                "mode": (MODE_OPTIONS, {"tooltip": MODE_TOOLTIP}),
                "merged_transition": (transitions,),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
                "blocksize": ("INT", {"default": 0, "min": 0, "max": 512, "advanced": True,
                                      "tooltip": BLOCKSIZE_TOOLTIP}),
                "inference_steps": ("INT", {"default": 0, "min": 0, "max": 32, "advanced": True,
                                            "tooltip": STEPS_TOOLTIP}),
                "beta": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                   "advanced": True,
                                   "tooltip": BETA_TOOLTIP}),
                "tap_gains": ("STRING", {"default": "", "advanced": True,
                                         "tooltip": TAPGAIN_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "info")
    FUNCTION = "load"
    CATEGORY = "model/loaders"

    def load(self, clip_name, type="qwen3vl_8b", mode="aligned", merged_transition="",
             device="default", blocksize=0, inference_steps=0, beta=-1.0, tap_gains=""):
        encoder_path = _resolve_transition(clip_name)
        spec = _spec_from_encoder_file(encoder_path, RELEASE_SPEC.taps)
        if blocksize:
            spec = CVRRSpec(ell_star=int(blocksize), num_recurrent_steps=spec.num_recurrent_steps,
                            beta=spec.beta, taps=spec.taps, name=spec.name)
        if inference_steps:
            spec = CVRRSpec(ell_star=spec.ell_star, num_recurrent_steps=int(inference_steps),
                            beta=spec.beta, taps=spec.taps, name=spec.name)
        if beta >= 0:
            spec = CVRRSpec(ell_star=spec.ell_star, num_recurrent_steps=spec.num_recurrent_steps,
                            beta=float(beta), taps=spec.taps, name=spec.name)

        transition_path = None
        if merged_transition:
            transition_path = _resolve_transition(merged_transition)
        else:
            # convenience only: a converted release bundles the canonical file
            # next to the encoder; picking the file in the combo always wins
            sibling = os.path.join(os.path.dirname(encoder_path), TRANSITION_FILE_NAME)
            transition_path = sibling if os.path.isfile(sibling) else None

        clip = build_clip(
            encoder_path,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            device=device,
            spec=spec,
            mode=mode,
            transition_path=transition_path,
            model_type=type,
        )
        no_vision = getattr(getattr(clip, "cond_stage_model", None), "cvrr_no_vision", False)
        info = (
            f"CVRR {spec.name}: ell_star={spec.ell_star} recurrent_layer={spec.recurrent_layer} "
            f"upper_decoder={spec.upper_decoder_start} T={spec.num_recurrent_steps} "
            f"beta={spec.beta} mode={mode} merged_transition="
            f"{'yes' if transition_path else 'NO (recurrence disabled)'} "
            f"vision_tower={'MISSING (text-only file, images disabled)' if no_vision else 'yes'}"
        )
        return (clip, info)


def _spec_with_overrides(directory: str, blocksize: int, inference_steps: int,
                         beta: float) -> CVRRSpec:
    """Release-rack spec from ``directory`` plus the optional advanced overrides."""
    spec = load_release_spec(directory, taps=RELEASE_SPEC.taps)
    if blocksize:
        spec = CVRRSpec(ell_star=int(blocksize), num_recurrent_steps=spec.num_recurrent_steps,
                        beta=spec.beta, taps=spec.taps, name=spec.name)
    if inference_steps:
        spec = CVRRSpec(ell_star=spec.ell_star, num_recurrent_steps=int(inference_steps),
                        beta=spec.beta, taps=spec.taps, name=spec.name)
    if beta >= 0:
        spec = CVRRSpec(ell_star=spec.ell_star, num_recurrent_steps=spec.num_recurrent_steps,
                        beta=float(beta), taps=spec.taps, name=spec.name)
    return spec


class CVRRApplyTransition:
    """Attach a CVRR merged transition to any Qwen3-VL CLIP, LoRA-style.

    The merged transition is the only CVRR-specific weight artifact, so instead
    of loading a dedicated converted file this node patches an already-loaded
    CLIP: stock ``CLIPLoader`` output (type ``flux2``), a converted CVRR file,
    or any Qwen3-VL-8B finetune that kept the architecture's shapes.  The
    transition file is selected from ``models/text_encoders`` exactly like a
    ``CLIPLoader``-style combo -- any name, any subfolder -- and is validated
    by contents (7 projection tensors), not by its filename.  The underlying
    encoder object is shared between cloned CLIP handles, but the transition
    weights stay disabled for regular (non-CVRR) encodes -- they are gated
    per encode by the CVRR encode nodes, not permanently merged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _encoder_files()
        return {
            "required": {
                "clip": ("CLIP",),
                "merged_transition": (files, {"tooltip":
                    "The CVRR merged transition safetensors (7 fp32 projection "
                    "tensors of the recurrent layer). Any file name works; it is "
                    "validated by contents, not by name."}),
            },
            "optional": {
                "blocksize": ("INT", {"default": 0, "min": 0, "max": 512, "advanced": True,
                                      "tooltip": BLOCKSIZE_TOOLTIP}),
                "inference_steps": ("INT", {"default": 0, "min": 0, "max": 32, "advanced": True,
                                            "tooltip": STEPS_TOOLTIP}),
                "beta": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                   "advanced": True,
                                   "tooltip": BETA_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "info")
    FUNCTION = "apply"
    CATEGORY = "model/conditioning/cvrr"

    def apply(self, clip, merged_transition, blocksize=0, inference_steps=0, beta=-1.0):
        if not merged_transition:
            raise ValueError("comfyui_cvrr: select a merged transition file")
        path = _resolve_transition(merged_transition)
        spec = _spec_with_overrides(os.path.dirname(path), blocksize, inference_steps, beta)
        import comfy.utils

        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        if hasattr(clip, "clone"):
            patched = clip.clone()
        else:  # pragma: no cover - non-standard CLIP handle
            log.warning("comfyui_cvrr: input CLIP has no clone(); patching in place")
            patched = clip
        try:
            apply_transition(patched, state_dict, spec=spec)
        except ValueError as exc:
            raise ValueError(f"comfyui_cvrr: attaching {merged_transition!r} failed: {exc}") from exc
        patched.cvrr_attachment = spec
        info = (
            f"CVRR {spec.name}: ell_star={spec.ell_star} recurrent_layer={spec.recurrent_layer} "
            f"upper_decoder={spec.upper_decoder_start} T={spec.num_recurrent_steps} "
            f"beta={spec.beta} transition={os.path.basename(path)}"
        )
        log.info("comfyui_cvrr: attached %s -> %s", info, os.path.basename(path))
        return (patched, info)


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------


class CVRRTextEncode:
    """CLIP Text Encode whose conditioning is produced by CVRR's visual recurrence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "mode": (MODE_OPTIONS, {"tooltip": MODE_TOOLTIP}),
                "vl_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0,
                                            "step": 0.01, "tooltip": "Resolution handed to the "
                                            "Qwen3-VL vision tower (the image itself is *not* "
                                            "VAE-encoded here). Automatically clamped to a "
                                            "2048 px longest edge (~4096 visual tokens)."}),
                "tap_gains": ("STRING", {"default": "", "advanced": True,
                                         "tooltip": TAPGAIN_TOOLTIP}),
                "block_visual_access": ("BOOLEAN", {"default": False, "advanced": True,
                                                    "tooltip": "CVRR's causal ablation: the "
                                                    "recurrence may not re-read visual rows."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, prompt, mode, vl_megapixels, tap_gains="",
               block_visual_access=False, image=None, image2=None):
        spec = _cvrr_spec_of(clip)
        gains = _parse_gains(tap_gains, spec)
        options = CVRROptions(spec=spec, mode=mode, tap_gains=gains,
                              block_visual_access=block_visual_access)

        images = []
        for candidate in (image, image2):
            if candidate is not None:
                images.append(_resize_for_vl(candidate, vl_megapixels, multiple=32)[:1])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = _encode_with_options(clip, tokens, options)
        info = _describe(clip, options, has_image=bool(images))
        return conditioning, info


class CVRRTextEncodePlain:
    """Plain `CLIP Text Encode` for the CVRR stack.

    No image inputs, no visual recurrence: the conditioning is produced by the
    stock Qwen3-VL language pass and shaped by the TE wrapper the CLIP carries
    (Klein's 3-tap stack, ideogram4's 13-tap permute, ...).  Use it to test
    that the conditioning our stack produces is ingested correctly by models
    that natively speak a Qwen3-VL text encoder, without the edit/reference
    machinery in the loop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, prompt):
        spec = _cvrr_spec_of(clip)
        options = CVRROptions(spec=spec, mode="aligned")
        tokens = clip.tokenize(prompt, images=[])
        conditioning = _encode_with_options(clip, tokens, options)
        info = _describe(clip, options, has_image=False)
        return conditioning, info


class CVRREditTextEncode:
    """CVRR conditioning + FLUX.2 reference latents in one node.

    A CVRR-capable take on EditUtils' ``Flux2Klein Edit Text Encode``: one
    ``ref_longest_edge`` value drives the reference image for *both* the
    Qwen3-VL vision tower and the reference-latent VAE encode, and the node
    reports the processed reference size on its ``width``/``height`` outputs
    for chaining (e.g. into ``Empty Flux 2 Latent``).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "image": ("IMAGE",),
                "mode": (MODE_OPTIONS, {"tooltip": MODE_TOOLTIP}),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 64, "max": 4096,
                                             "step": 16,
                                             "tooltip": "Processed longest edge of the reference "
                                             "image (mirrors EditUtils' ref_longest_edge): drives "
                                             "both the VL encoder input AND the reference-latent "
                                             "VAE encode. Never upscales the source, caps at 2048 "
                                             "px (~4096 visual tokens), rounds down to /32."}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                               "default": ""}),
                "reference_latents_method": (["index", "offset", "uxo", "index_timestep_zero"],
                                             {"advanced": True, "default": "index",
                                              "tooltip": REF_METHOD_TOOLTIP}),
                "tap_gains": ("STRING", {"default": "", "advanced": True,
                                         "tooltip": TAPGAIN_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "negative", "width", "height", "info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, vae, prompt, image, mode, ref_longest_edge,
               image2=None, negative_prompt="", reference_latents_method="index", tap_gains=""):
        spec = _cvrr_spec_of(clip)
        gains = _parse_gains(tap_gains, spec)
        options = CVRROptions(spec=spec, mode=mode, tap_gains=gains)

        width, height = _ref_target_size(image, ref_longest_edge)
        prepared = [_resize_to_size(image, width, height)[:1]]
        if image2 is not None:
            width2, height2 = _ref_target_size(image2, ref_longest_edge)
            prepared.append(_resize_to_size(image2, width2, height2)[:1])

        tokens = clip.tokenize(prompt, images=prepared)
        positive = _encode_with_options(clip, tokens, options)

        negative_tokens = clip.tokenize(negative_prompt, images=prepared)
        negative = _encode_with_options(clip, negative_tokens, options)
        negative = node_helpers.conditioning_set_values(negative, {"pooled_output": None})

        refs = [vae.encode(candidate) for candidate in prepared]

        if refs:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": refs}, append=True
            )
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents_method": reference_latents_method}
        )
        stats = _tap_stats(clip)
        info = (_describe(clip, options, has_image=True, references=len(refs))
                + f" ref={width}x{height} method={reference_latents_method}"
                + (f" taps[{stats}]" if stats else ""))
        return positive, negative, width, height, info


class CVRRSetReferenceLatent:
    """"Set Reference Latent+" -- VAE-encodes images and chains them onto conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "reference_latents_method": (["index", "offset", "uxo", "index_timestep_zero"],
                                             {"advanced": True, "default": "index"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "reference_latent")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, conditioning, vae, image, megapixels, image2=None,
              reference_latents_method="index"):
        import comfy.utils

        refs = []
        for candidate in (image, image2):
            if candidate is None:
                continue
            samples = candidate.movedim(-1, 1)
            total = float(samples.shape[2] * samples.shape[3])
            target = max(1.0, megapixels * 1024.0 * 1024.0)
            scale = math.sqrt(target / total)
            width = max(16, int(round(samples.shape[3] * scale / 16)) * 16)
            height = max(16, int(round(samples.shape[2] * scale / 16)) * 16)
            scaled = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
            refs.append(vae.encode(scaled.movedim(1, -1)[:, :, :, :3]))

        if not refs:
            raise ValueError("comfyui_cvrr: no image supplied")
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": refs}, append=True
        )
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents_method": reference_latents_method}
        )
        return conditioning, {"samples": refs[0]}


# ---------------------------------------------------------------------------
# small utilities shared by the nodes
# ---------------------------------------------------------------------------


def _cvrr_spec_of(clip) -> CVRRSpec:
    attachment = getattr(clip, "cvrr_attachment", None)
    if isinstance(attachment, CVRRSpec):
        return attachment
    model = getattr(clip, "cond_stage_model", None)
    clip_model = getattr(model, getattr(model, "clip", ""), None)
    options = getattr(getattr(clip_model, "transformer", None), "cvrr_options", None)
    if options is not None:
        return options.spec
    return RELEASE_SPEC


def _parse_gains(text: str, spec: CVRRSpec) -> Tuple[float, ...]:
    if not text or not text.strip():
        return ()
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    values = tuple(float(p) for p in parts)
    if len(values) != len(spec.taps):
        raise ValueError(
            f"comfyui_cvrr: tap_gains expects {len(spec.taps)} values for taps "
            f"{spec.taps}, got {len(values)}"
        )
    return values


def _tap_stats(clip) -> str:
    """Concise per-tap magnitude summary of the last CVRR encode (diagnostics)."""
    model = getattr(clip, "cond_stage_model", None)
    clip_model = getattr(model, getattr(model, "clip", ""), None)
    result = getattr(getattr(clip_model, "transformer", None), "last_cvrr_result", None)
    if result is None:
        return ""
    parts = []
    for tap in result.taps:
        t = tap.detach().float()
        parts.append(f"mu={t.mean().item():+.3g} sd={t.std().item():.3g} |max|={t.abs().max().item():.3g}")
    return "; ".join(parts)


def _describe(clip, options: CVRROptions, has_image: bool, references: int = 0) -> str:
    model = getattr(clip, "cond_stage_model", None)
    ready = bool(getattr(getattr(model, "text_model", None), "cvrr_ready", False))
    spec = options.spec
    return (
        f"CVRR mode={options.mode} ready={ready} has_image={has_image} refs={references} | "
        f"ell_star={spec.ell_star} T={spec.num_recurrent_steps} beta={spec.beta} "
        f"taps={spec.taps} gains={options.resolved_gains()}"
    )


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "CVRRTextEncoderLoader": CVRRTextEncoderLoader,
    "CVRRApplyTransition": CVRRApplyTransition,
    "CVRRTextEncode": CVRRTextEncode,
    "CVRRTextEncodePlain": CVRRTextEncodePlain,
    "CVRREditTextEncode": CVRREditTextEncode,
    "CVRRSetReferenceLatent": CVRRSetReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "CVRRTextEncoderLoader": "CVRR Qwen3-VL Text Encoder Loader",
    "CVRRApplyTransition": "CVRR Apply Merged Transition",
    "CVRRTextEncode": "CVRR Text Encode (Qwen3-VL / Klein)",
    "CVRRTextEncodePlain": "CVRR Text Encode (text-only prompt)",
    "CVRREditTextEncode": "CVRR Edit Text Encode (Klein ref-latent)",
    "CVRRSetReferenceLatent": "CVRR Set Reference Latent+",
}
