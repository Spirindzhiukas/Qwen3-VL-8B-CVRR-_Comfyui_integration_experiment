"""The loader path: a converted file -> ``comfy.sd.CLIP`` -> conditioning.

This is the same route a real workflow takes (``comfy.sd.CLIP`` with a
``ClipTarget``), on a tiny checkpoint, so it proves the host classes are wired
the way ComfyUI expects: parameter names, ``load_sd``, tokenizer, patcher and
``encode_from_tokens_scheduled``.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import TINY_DIM, TINY_MODEL_TYPE, TINY_TAPS, random_transition

pytestmark = pytest.mark.comfy


@pytest.fixture(scope="module")
def tiny_checkpoint(comfy, tmp_path_factory):
    """Write a tiny CVRR text encoder + merged transition to disk."""
    import comfyui_cvrr.cvrr_te as cvrr_te
    from safetensors.torch import save_file

    model = cvrr_te.make_cvrr_qwen3vl(TINY_MODEL_TYPE)({}, torch.float32, torch.device("cpu"),
                                                      torch.nn)
    directory = tmp_path_factory.mktemp("cvrr_tiny")
    state = {key: value.detach().clone().contiguous()
             for key, value in model.state_dict().items()}
    encoder_path = directory / "cvrr_tiny.safetensors"
    save_file(state, str(encoder_path))

    # merged transition: 7 fp32 projections of the recurrent layer
    class _Holder:
        transformer = model

    spec = __import__("comfyui_cvrr.cvrr_core", fromlist=["x"]).CVRRSpec(
        ell_star=2, num_recurrent_steps=4, beta=0.33, taps=TINY_TAPS
    )
    transition_path = directory / "merged_transition.safetensors"
    save_file({k: v.contiguous().float() for k, v in random_transition(_Holder, spec).items()},
              str(transition_path))
    return encoder_path, transition_path, spec


def test_build_clip_and_encode(comfy, tiny_checkpoint):
    import comfyui_cvrr.cvrr_te as cvrr_te

    encoder_path, transition_path, spec = tiny_checkpoint
    clip = cvrr_te.build_clip(
        str(encoder_path), device="cpu", spec=spec, mode="aligned",
        model_type=TINY_MODEL_TYPE, transition_path=str(transition_path),
    )
    assert cvrr_te.is_cvrr_clip(clip)
    assert clip.cond_stage_model.text_model.cvrr_ready

    tokens = clip.tokenize("Make it look like a watercolour.",
                           images=[torch.rand(1, 64, 64, 3)])
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    assert len(conditioning) == 1
    cond, meta = conditioning[0]
    assert cond.shape[0] == 1
    assert cond.shape[-1] == 3 * TINY_DIM       # Klein's 3 x 4096 layout
    mask = meta["attention_mask"]
    assert mask.shape == (1, cond.shape[1])
    assert int(mask.sum()) == cond.shape[1]


def test_loaded_weights_are_the_converted_ones(comfy, tiny_checkpoint):
    """The loader must actually install the file's weights, not a fresh model."""
    import comfyui_cvrr.cvrr_te as cvrr_te

    encoder_path, _transition, spec = tiny_checkpoint
    clip = cvrr_te.build_clip(str(encoder_path), device="cpu", spec=spec,
                              model_type=TINY_MODEL_TYPE)
    text_model = clip.cond_stage_model.text_model
    assert text_model.model.embed_tokens.weight[1, :3].abs().sum() > 0

    # the merged transition is NOT part of the text encoder file: the recurrence
    # stays disabled until it is attached.
    assert not text_model.cvrr_ready


def test_apply_transition_enables_recurrence(comfy, tiny_checkpoint, spec):
    import comfyui_cvrr.cvrr_te as cvrr_te
    from safetensors.torch import load_file

    encoder_path, transition_path, spec = tiny_checkpoint
    clip = cvrr_te.build_clip(str(encoder_path), device="cpu", spec=spec,
                              model_type=TINY_MODEL_TYPE)
    cvrr_te.apply_transition(clip, load_file(str(transition_path)), spec=spec)
    text_model = clip.cond_stage_model.text_model
    assert text_model.cvrr_ready
    # mode/step settings are still applied after the weights land
    options = clip.cond_stage_model.text_model.cvrr_options
    assert options is not None and options.spec.num_recurrent_steps == 4
