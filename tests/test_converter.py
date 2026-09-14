"""The converter: released HF layout in, loadable ComfyUI file out.

Builds a fake ``dmis-lab/Qwen3-VL-8B-CVRR``-shaped release directory (HF-style
keys: ``model.language_model.*``, ``model.visual.*``, ``lm_head.*``) from the tiny
model, converts it, checks the key mapping against ComfyUI's own parameter names
and finally loads the result through :func:`comfyui_cvrr.cvrr_te.build_clip`.
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.conftest import TINY_MODEL_TYPE, TINY_TAPS, random_transition

pytestmark = pytest.mark.comfy

#: ComfyUI (converted) prefix -> released-checkpoint prefix
INVERSE_PREFIX = (
    ("visual.", "model.visual."),
    ("model.lm_head.", "lm_head."),
)


def hf_key_for(comfy_key: str) -> str:
    for comfy_prefix, hf_prefix in INVERSE_PREFIX:
        if comfy_key.startswith(comfy_prefix):
            return hf_prefix + comfy_key[len(comfy_prefix):]
    if comfy_key.startswith("model."):
        return "model.language_model." + comfy_key[len("model."):]
    return comfy_key


@pytest.fixture(scope="module")
def release_dir(comfy, tmp_path_factory):
    """A miniature released checkpoint directory."""
    import comfyui_cvrr.cvrr_te as cvrr_te
    from safetensors.torch import save_file

    directory = tmp_path_factory.mktemp("cvrr_release")
    model = cvrr_te.make_cvrr_qwen3vl(TINY_MODEL_TYPE)({}, torch.float32, torch.device("cpu"),
                                                       torch.nn)
    state = {hf_key_for(key): value.detach().clone().contiguous()
             for key, value in model.state_dict().items()}

    native = directory / "native_backbone"
    native.mkdir()
    keys = sorted(state)
    half = len(keys) // 2
    shards = {"native-00001.safetensors": keys[:half], "native-00002.safetensors": keys[half:]}
    weight_map = {}
    for name, shard_keys in shards.items():
        save_file({key: state[key] for key in shard_keys}, str(native / name))
        weight_map.update({key: name for key in shard_keys})
    (native / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(v.numel() * v.element_size()
                                                   for v in state.values())},
                    "weight_map": weight_map}),
        encoding="utf-8",
    )
    (directory / "cvrr_release_config.json").write_text(
        json.dumps({"name": "CVRR-tiny", "ell_star": 2, "recurrent_layer": 3,
                    "upper_decoder_start": 4, "inference_T": 4, "inference_beta": 0.33}),
        encoding="utf-8",
    )

    spec = __import__("comfyui_cvrr.cvrr_core", fromlist=["x"]).CVRRSpec(
        ell_star=2, num_recurrent_steps=4, beta=0.33, taps=TINY_TAPS
    )

    class _Holder:
        transformer = model

    save_file({k: v.float().contiguous() for k, v in random_transition(_Holder, spec).items()},
              str(directory / "merged_transition.safetensors"))
    return directory


def test_convert_maps_every_parameter(comfy, release_dir, tmp_path, capsys):
    from comfyui_cvrr.convert_cvrr_to_comfy import convert, verify

    out = tmp_path / "comfy_te"
    convert(str(release_dir), str(out), dtype="keep")
    produced = {p.name for p in out.iterdir()}
    assert "cvrr_merged_transition.safetensors" in produced
    assert any(name.endswith(".safetensors") and "transition" not in name for name in produced)
    assert "cvrr_release_config.json" in produced

    assert verify(str(out), model_type=TINY_MODEL_TYPE) == 0
    assert "OK: every converted tensor maps onto a ComfyUI Qwen3-VL parameter" in capsys.readouterr().out


def test_converted_file_loads_and_encodes(comfy, release_dir, tmp_path, spec, tokenizer):
    import comfyui_cvrr.cvrr_te as cvrr_te
    from comfyui_cvrr.convert_cvrr_to_comfy import convert

    out = tmp_path / "comfy_te"
    convert(str(release_dir), str(out), dtype="keep")
    encoder = next(str(p) for p in sorted(out.iterdir())
                   if p.suffix == ".safetensors" and "transition" not in p.name)

    clip = cvrr_te.build_clip(
        encoder, device="cpu", spec=spec, model_type=TINY_MODEL_TYPE,
        transition_path=str(out / "cvrr_merged_transition.safetensors"),
    )
    assert cvrr_te.is_cvrr_clip(clip)
    assert clip.cond_stage_model.text_model.cvrr_ready
    conditioning = clip.encode_from_tokens_scheduled(
        clip.tokenize("Turn the sketch into a photograph.",
                      images=[torch.rand(1, 64, 64, 3)])
    )
    cond = conditioning[0][0]
    assert cond.shape[0] == 1 and cond.shape[-1] == 3 * 32


def test_convert_transition_only(release_dir, tmp_path):
    from comfyui_cvrr.convert_cvrr_to_comfy import convert
    from safetensors.torch import load_file

    out = tmp_path / "transition"
    convert(str(release_dir), str(out), transition_only=True, dtype="fp16")
    weights = load_file(str(out / "cvrr_merged_transition.safetensors"))
    assert len(weights) == 7
    assert all(value.dtype == torch.float16 for value in weights.values())


def test_map_key_rejects_unrelated_keys():
    from comfyui_cvrr.convert_cvrr_to_comfy import map_key

    assert map_key("lm_head.weight") == "model.lm_head.weight"
    assert map_key("model.language_model.norm.weight") == "model.norm.weight"
    assert map_key("model.visual.merger.linear_fc2.weight") == "visual.merger.linear_fc2.weight"
    assert map_key("model.visual.rotary.inv_freq") is None
    assert map_key("model.language_model.norm.weight", keep_vision=False) == "model.norm.weight"
