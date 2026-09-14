"""ComfyUI nodes for the CVRR Qwen3-VL text encoder.

Node set
--------
``CVRRTextEncoderLoader``   load a converted CVRR text encoder (+ optional merged
                            transition) with ComfyUI's own ``CLIP`` object
``CVRRApplyTransition``     attach ``merged_transition.safetensors`` to an already
                            loaded CVRR encoder
``CVRRTextEncode``          CLIP Text Encode with an optional reference image: the
                            conditioning is produced by CVRR's visual recurrence
``CVRREditTextEncode``      convenience node in the spirit of community
                            "Flux2 Klein Edit Text Encode" nodes: prompt + image +
                            VAE -> CVRR conditioning *and* FLUX.2 reference latents
``CVRRSetReferenceLatent``  "Set Reference Latent+" that VAE-encodes the image for
                            you and can chain multiple references
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import folder_paths
import node_helpers

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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _encoder_files() -> List[str]:
    try:
        return folder_paths.get_filename_list("text_encoders")
    except Exception:  # pragma: no cover - folder not registered
        return []


def _transition_files() -> List[str]:
    """Merged-transition files: ``merged_transition.safetensors`` in any model
    folder, or anything matching ``*transition*.safetensors`` next to a text
    encoder."""
    found: List[str] = []
    for folder in ("text_encoders", "clip", "diffusion_models", "unet"):
        try:
            names = folder_paths.get_filename_list(folder)
        except Exception:  # pragma: no cover
            continue
        for name in names:
            base = os.path.basename(name).lower()
            if "transition" in base or "cvrr" in base:
                if name not in found:
                    found.append(name)
    return found


def _resolve(folder: str, name: str) -> str:
    path = folder_paths.get_full_path(folder, name)
    if path is None:  # pragma: no cover - defensive
        raise ValueError(f"comfyui_cvrr: could not resolve {folder}/{name}")
    return path


def _spec_from_encoder_file(encoder_path: str, taps: Sequence[int]) -> CVRRSpec:
    """Look for CVRR metadata next to the encoder file."""
    directory = os.path.dirname(encoder_path)
    return load_release_spec(directory, taps=taps)


def _encode_with_options(clip, tokens, options: CVRROptions, enabled: bool = True):
    """Run ``clip.encode_from_tokens_scheduled`` under a CVRR option set."""
    model = clip.cond_stage_model
    if not is_cvrr_clip(clip):
        raise ValueError(
            "comfyui_cvrr: this CLIP was not loaded as a CVRR text encoder. Use the "
            "'CVRR Qwen3-VL Text Encoder Loader' node (recommended) or convert the "
            "checkpoint and load it with CLIPLoader type 'flux2' -- the latter runs "
            "the stock Qwen3-VL language model *without* the CVRR recurrence."
        )
    previous = getattr(getattr(model, "text_model", None), "cvrr_options", None)
    spec = previous.spec if previous is not None else options.spec
    model.configure_cvrr(
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
            model.text_model.cvrr_options = None
        else:
            model.configure_cvrr(
                spec=previous.spec,
                mode=previous.mode,
                tap_gains=previous.tap_gains,
                block_visual_access=previous.block_visual_access,
                enabled=previous.enabled,
            )


def _resize_for_vl(image: torch.Tensor, megapixels: float, multiple: int = 32) -> torch.Tensor:
    """Scale an image batch to roughly ``megapixels`` before feeding the VLM."""
    import comfy.utils

    if megapixels <= 0:
        return image
    samples = image.movedim(-1, 1)
    total = float(samples.shape[2] * samples.shape[3])
    target = max(1.0, megapixels * 1024.0 * 1024.0)
    scale = math.sqrt(target / total)
    width = max(multiple, int(round(samples.shape[3] * scale / multiple)) * multiple)
    height = max(multiple, int(round(samples.shape[2] * scale / multiple)) * multiple)
    out = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return out.movedim(1, -1)[:, :, :, :3]


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


class CVRRTextEncoderLoader:
    """Load a converted CVRR Qwen3-VL text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        encoders = _encoder_files()
        transitions = [""] + _transition_files()
        return {
            "required": {
                "clip_name": (encoders,),
                "mode": (MODE_OPTIONS,),
                "merged_transition": (transitions,),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
                "blocksize": ("INT", {"default": 0, "min": 0, "max": 512, "advanced": True,
                                      "tooltip": "0 = use the release value (ell_star)."}),
                "inference_steps": ("INT", {"default": 0, "min": 0, "max": 32, "advanced": True,
                                            "tooltip": "0 = use the release value (T)."}),
                "beta": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                   "advanced": True,
                                   "tooltip": "-1 = use the release value (0.33)."}),
                "tap_gains": ("STRING", {"default": "", "advanced": True,
                                         "tooltip": "Optional per-tap multipliers, e.g. '1,1,1.2'."}),
            },
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "info")
    FUNCTION = "load"
    CATEGORY = "model/loaders"

    def load(self, clip_name, mode="aligned", merged_transition="", device="default",
             blocksize=0, inference_steps=0, beta=-1.0, tap_gains=""):
        encoder_path = _resolve("text_encoders", clip_name)
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
            folder = "text_encoders"
            transition_path = folder_paths.get_full_path(folder, merged_transition)
            if transition_path is None:
                for candidate in ("diffusion_models", "unet", "clip"):
                    transition_path = folder_paths.get_full_path(candidate, merged_transition)
                    if transition_path is not None:
                        break
        if transition_path is None:
            # look for a sibling file with the canonical release name
            sibling = os.path.join(os.path.dirname(encoder_path), TRANSITION_FILE_NAME)
            transition_path = sibling if os.path.isfile(sibling) else None

        clip = build_clip(
            encoder_path,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            device=device,
            spec=spec,
            mode=mode,
            transition_path=transition_path,
        )
        info = (
            f"CVRR {spec.name}: ell_star={spec.ell_star} recurrent_layer={spec.recurrent_layer} "
            f"upper_decoder={spec.upper_decoder_start} T={spec.num_recurrent_steps} "
            f"beta={spec.beta} mode={mode} merged_transition="
            f"{'yes' if transition_path else 'NO (recurrence disabled)'}"
        )
        return (clip, info)


class CVRRApplyTransition:
    """Install ``merged_transition.safetensors`` on a loaded CVRR text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "merged_transition": ([""] + _transition_files(),),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "model/conditioning/cvrr"

    def apply(self, clip, merged_transition):
        path = None
        for folder in ("text_encoders", "diffusion_models", "unet", "clip"):
            path = folder_paths.get_full_path(folder, merged_transition)
            if path:
                break
        if path is None:
            raise ValueError(f"comfyui_cvrr: cannot find {merged_transition}")
        import comfy.utils

        state_dict, _ = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        apply_transition(clip, state_dict)
        return (clip,)


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
                "mode": (MODE_OPTIONS,),
                "vl_megapixels": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 8.0,
                                            "step": 0.01, "tooltip": "Resolution handed to the "
                                            "Qwen3-VL vision tower (the image itself is *not* "
                                            "VAE-encoded here)."}),
                "tap_gains": ("STRING", {"default": "", "advanced": True}),
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


class CVRREditTextEncode:
    """CVRR conditioning + FLUX.2 reference latents in one node.

    Equivalent to ``CVRRTextEncode`` followed by ``VAEEncode`` +
    ``ReferenceLatent`` (a.k.a. the "Set Reference Latent" pattern), which is the
    shape most community Klein edit nodes use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "image": ("IMAGE",),
                "mode": (MODE_OPTIONS,),
                "vl_megapixels": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 8.0, "step": 0.01}),
                "reference_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0,
                                                   "step": 0.05,
                                                   "tooltip": "Resolution of the image that is "
                                                   "VAE-encoded into the reference latent."}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                               "default": ""}),
                "reference_latents_method": (["index", "offset", "uxo", "index_timestep_zero"],
                                             {"advanced": True, "default": "index"}),
                "tap_gains": ("STRING", {"default": "", "advanced": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, vae, prompt, image, mode, vl_megapixels, reference_megapixels,
               image2=None, negative_prompt="", reference_latents_method="index", tap_gains=""):
        import comfy.utils

        spec = _cvrr_spec_of(clip)
        gains = _parse_gains(tap_gains, spec)
        options = CVRROptions(spec=spec, mode=mode, tap_gains=gains)

        vl_images = [_resize_for_vl(image, vl_megapixels, multiple=32)[:1]]
        if image2 is not None:
            vl_images.append(_resize_for_vl(image2, vl_megapixels, multiple=32)[:1])

        tokens = clip.tokenize(prompt, images=vl_images)
        positive = _encode_with_options(clip, tokens, options)

        negative_tokens = clip.tokenize(negative_prompt, images=vl_images)
        negative = _encode_with_options(clip, negative_tokens, options)
        negative = node_helpers.conditioning_set_values(negative, {"pooled_output": None})

        refs = []
        for candidate in (image, image2):
            if candidate is None:
                continue
            samples = candidate.movedim(-1, 1)
            total = float(samples.shape[2] * samples.shape[3])
            target = max(1.0, reference_megapixels * 1024.0 * 1024.0)
            scale = math.sqrt(target / total)
            width = max(16, int(round(samples.shape[3] * scale / 16)) * 16)
            height = max(16, int(round(samples.shape[2] * scale / 16)) * 16)
            scaled = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
            refs.append(vae.encode(scaled.movedim(1, -1)[:, :, :, :3]))

        if refs:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": refs}, append=True
            )
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents_method": reference_latents_method}
        )
        info = _describe(clip, options, has_image=True, references=len(refs))
        return positive, negative, info


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
    model = getattr(clip, "cond_stage_model", None)
    options = getattr(getattr(model, "text_model", None), "cvrr_options", None)
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
    "CVRREditTextEncode": CVRREditTextEncode,
    "CVRRSetReferenceLatent": CVRRSetReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "CVRRTextEncoderLoader": "CVRR Qwen3-VL Text Encoder Loader",
    "CVRRApplyTransition": "CVRR Apply Merged Transition",
    "CVRRTextEncode": "CVRR Text Encode (Qwen3-VL / Klein)",
    "CVRREditTextEncode": "CVRR Edit Text Encode (Klein ref-latent)",
    "CVRRSetReferenceLatent": "CVRR Set Reference Latent+",
}
