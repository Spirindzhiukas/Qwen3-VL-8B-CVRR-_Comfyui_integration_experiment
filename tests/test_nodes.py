"""Node-definition smoke tests.

The nodes are the user-facing surface and ComfyUI imports them inside a real
ComfyUI process, which this test-suite cannot start.  Instead, the two ComfyUI
modules the node file needs (``folder_paths``, ``node_helpers``) are stubbed, so
the node classes can be imported, their ``INPUT_TYPES`` evaluated and their
declarations checked for consistency.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from tests.conftest import TINY_DIM, TINY_MODEL_TYPE

pytestmark = pytest.mark.comfy


@pytest.fixture()
def nodes_module(monkeypatch, comfy):
    folder_paths = types.ModuleType("folder_paths")
    # Plausible contents so that combo-type widget values can be validated.
    model_files = {
        "text_encoders": [
            "qwen_3_8b_fp8mixed.safetensors",
            "cvrr/cvrr_qwen3vl_8b-00001.safetensors",
            "cvrr/cvrr_merged_transition.safetensors",
        ],
    }
    folder_paths.get_filename_list = lambda folder: list(model_files.get(folder, []))
    folder_paths.get_full_path = lambda folder, name: None
    folder_paths.get_folder_paths = lambda folder: []
    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values, append=False):
        out = []
        for tensor, meta in conditioning:
            meta = dict(meta)
            for key, value in values.items():
                if append and key in meta:
                    meta[key] = list(meta[key]) + list(value)
                else:
                    meta[key] = value
            out.append([tensor, meta])
        return out

    node_helpers.conditioning_set_values = conditioning_set_values
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "node_helpers", node_helpers)
    monkeypatch.delitem(sys.modules, "comfyui_cvrr.nodes", raising=False)

    import comfyui_cvrr.nodes as nodes

    return nodes


def test_mappings_are_consistent(nodes_module):
    nodes = nodes_module
    assert nodes.NODE_CLASS_MAPPINGS, "no nodes registered"
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert cls.CATEGORY
        assert isinstance(cls.RETURN_TYPES, tuple) and cls.RETURN_TYPES
        if hasattr(cls, "RETURN_NAMES"):
            assert len(cls.RETURN_NAMES) == len(cls.RETURN_TYPES), name
        assert callable(getattr(cls, cls.FUNCTION, None)), f"{name}.{cls.FUNCTION} is missing"


def test_input_types_evaluate(nodes_module):
    nodes = nodes_module
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        inputs = cls.INPUT_TYPES()
        assert "required" in inputs, name
        for section in ("required", "optional"):
            for key, spec in inputs.get(section, {}).items():
                # ComfyUI allows (list,) and (type, options) -- and (list, options)
                assert isinstance(spec, tuple) and 1 <= len(spec) <= 2, f"{name}.{key}"
                assert isinstance(spec[0], (list, str)), f"{name}.{key}"
                if len(spec) == 2:
                    assert isinstance(spec[1], dict), f"{name}.{key}"


def test_static_choices_are_populated(nodes_module):
    """Combos that do not come from folder_paths must list their choices."""
    nodes = nodes_module
    loader = nodes.NODE_CLASS_MAPPINGS["CVRRTextEncoderLoader"].INPUT_TYPES()
    assert loader["required"]["mode"][0] == ["aligned", "strict", "vl"]
    edit = nodes.NODE_CLASS_MAPPINGS["CVRREditTextEncode"].INPUT_TYPES()
    assert edit["optional"]["reference_latents_method"][0] == \
        ["index", "offset", "uxo", "index_timestep_zero"]


def test_encode_nodes_accept_images_and_report_info(nodes_module):
    nodes = nodes_module
    for name in ("CVRRTextEncode", "CVRREditTextEncode"):
        inputs = nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()
        assert inputs["required"]["clip"][0] == "CLIP"
        optional = inputs.get("optional", {})
        assert "image" in optional or "image" in inputs["required"]
    inputs = nodes.NODE_CLASS_MAPPINGS["CVRREditTextEncode"].INPUT_TYPES()
    assert inputs["required"]["vae"][0] == "VAE"
    assert "reference_latents_method" in inputs["optional"]


def test_parse_tap_gains(nodes_module):
    from tests.conftest import TINY_TAPS

    spec = __import__("comfyui_cvrr.cvrr_core", fromlist=["x"]).CVRRSpec(taps=TINY_TAPS)
    assert nodes_module._parse_gains("", spec) == ()
    assert nodes_module._parse_gains("1, 1.5,2", spec) == (1.0, 1.5, 2.0)
    with pytest.raises(ValueError):
        nodes_module._parse_gains("1,2", spec)


def test_example_workflow_matches_node_schemas(nodes_module):
    """The shipped example graph must feed every node the way ComfyUI will.

    ComfyUI assigns widget values positionally (required first, then optional),
    so a stale example graph is a silent mis-wiring rather than an error.
    """
    import json
    import pathlib

    nodes = nodes_module
    path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "klein9b_cvrr_edit.json"
    graph = json.loads(path.read_text())

    link_types = {"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK"}
    checked = 0
    for node in graph["nodes"]:
        cls = nodes.NODE_CLASS_MAPPINGS.get(node["type"])
        if cls is None:
            continue  # a core ComfyUI node
        checked += 1
        schema = cls.INPUT_TYPES()
        declared = {}
        for section in ("required", "optional"):
            declared.update(schema.get(section, {}))

        for entry in node.get("inputs", []):
            assert entry["name"] in declared, f"{node['type']}: unknown input {entry['name']}"

        widgets = [name for name, spec in declared.items()
                   if not (isinstance(spec[0], str) and spec[0] in link_types)]
        assert len(node["widgets_values"]) == len(widgets), (
            f"{node['type']}: {len(node['widgets_values'])} widget values for {widgets}"
        )
        for value, name in zip(node["widgets_values"], widgets):
            options = declared[name][0]
            if isinstance(options, list):
                assert value in options, f"{node['type']}.{name}: {value!r} not in {options}"
            elif options == "INT":
                assert isinstance(value, int), f"{node['type']}.{name}"
            elif options == "FLOAT":
                assert isinstance(value, (int, float)), f"{node['type']}.{name}"
            elif options == "STRING":
                assert isinstance(value, str), f"{node['type']}.{name}"

    assert checked >= 2, "example workflow no longer references any CVRR node"


# ---------------------------------------------------------------------------
# LoRA-style attach: CVRRApplyTransition on a *stock* CLIPLoader-produced CLIP
# ---------------------------------------------------------------------------


def _stock_tiny_clip(comfy, seed=1234):
    """A stock ``Flux2TEModel`` + ``Qwen3VLClipModel`` inside a real ``comfy.sd.CLIP``.

    This is the same object ``CLIPLoader(type='flux2')`` returns for a small
    Qwen3-VL checkpoint: none of the CVRR classes are involved.
    """
    import comfy.sd
    import comfy.supported_models_base
    import comfy.text_encoders.flux as flux
    import comfy.text_encoders.qwen3vl as qwen3vl

    def _clip_model(**kw):
        return qwen3vl.Qwen3VLClipModel(**kw, model_type=TINY_MODEL_TYPE)

    class _StockTE(flux.Flux2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            super().__init__(device=device, dtype=dtype, model_options=model_options,
                             name=TINY_MODEL_TYPE, clip_model=_clip_model)

    target = comfy.supported_models_base.ClipTarget(
        qwen3vl.tokenizer(model_type=TINY_MODEL_TYPE), _StockTE)
    cpu = torch.device("cpu")
    clip = comfy.sd.CLIP(target, model_options={"load_device": cpu, "offload_device": cpu})
    torch.manual_seed(seed)
    state = {k: torch.randn(v.shape, dtype=torch.float32) * 0.05
             for k, v in clip.cond_stage_model.state_dict().items()
             if v.is_floating_point()}
    clip.cond_stage_model.load_state_dict(state, strict=False)
    return clip


def _write_transition(clip, spec, tmp_path):
    """Stand-in transition file (plus tiny release geometry) for the given clip."""
    import json

    from safetensors.torch import save_file

    from tests.conftest import random_transition

    model = getattr(clip.cond_stage_model, clip.cond_stage_model.clip)
    sd = random_transition(model, spec)
    path = tmp_path / "cvrr" / "cvrr_merged_transition.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))
    # The apply node reads the release geometry from this file; the tiny
    # stand-in model needs its own geometry (taps come from the patched
    # nodes_module.RELEASE_SPEC inside the tiny_spec_in_nodes fixture).
    (path.parent / "cvrr_release_config.json").write_text(json.dumps({
        "ell_star": spec.ell_star,
        "inference_T": spec.num_recurrent_steps,
        "inference_beta": spec.beta,
        "name": "tiny-cvrr-test",
    }))
    return path


def _serve_file(monkeypatch, nodes_module, name, path):
    files = {("text_encoders", name): str(path),
             ("diffusion_models", name): str(path),
             ("unet", name): str(path),
             ("clip", name): str(path)}
    monkeypatch.setattr(nodes_module.folder_paths, "get_full_path",
                        lambda folder, filename: files.get((folder, filename)))


@pytest.fixture()
def tiny_spec_in_nodes(monkeypatch, nodes_module, spec):
    # The apply node reads the release tarp geometry from disk; for the tiny
    # stand-in model, swap the release defaults for the tiny spec (taps and
    # ell_star both must fit 6 layers).
    monkeypatch.setattr(nodes_module, "RELEASE_SPEC", spec)
    return spec


def test_apply_transition_retrofits_a_stock_clip(comfy, nodes_module, monkeypatch,
                                                 tmp_path, tiny_spec_in_nodes):
    nodes = nodes_module
    spec = tiny_spec_in_nodes
    clip = _stock_tiny_clip(comfy)
    path = _write_transition(clip, spec, tmp_path)
    _serve_file(monkeypatch, nodes_module, "cvrr/cvrr_merged_transition.safetensors", path)

    assert not nodes.is_cvrr_clip(clip), "stock loader output must not start CVRR-capable"
    image = torch.rand(1, 64, 64, 3)
    tokens = clip.tokenize("a small robot", images=[image])
    with torch.no_grad():
        baseline = clip.encode_from_tokens_scheduled(tokens)

    patched, info = nodes.CVRRApplyTransition().apply(
        clip=clip, merged_transition="cvrr/cvrr_merged_transition.safetensors")
    assert patched is not clip, "LoRA-style nodes return a new CLIP handle"
    assert nodes.is_cvrr_clip(patched)
    attachment = getattr(patched, "cvrr_attachment", None)
    assert attachment is not None and attachment.taps == spec.taps
    assert (attachment.ell_star, attachment.num_recurrent_steps, attachment.beta) \
        == (spec.ell_star, spec.num_recurrent_steps, spec.beta)
    assert f"ell_star={spec.ell_star}" in info

    # The original handle's stock encode is unchanged: the merged weights are
    # installed but gated per encode, so attaching them is inert by itself.
    with torch.no_grad():
        stock_again = clip.encode_from_tokens_scheduled(tokens)
    assert torch.allclose(stock_again[0][0], baseline[0][0])

    cond, _ = nodes.CVRRTextEncode().encode(clip=patched, prompt="a small robot",
                                            mode="aligned", vl_megapixels=0.0, image=image)
    tensor = cond[0][0]
    assert tensor.shape[0] == 1 and tensor.shape[2] == 3 * TINY_DIM
    assert not torch.equal(tensor, baseline[0][0])

    # Weighted prompts through the stock path still work on the same encoder:
    # the no-per-token-weights rule only applies while a CVRR encode is active.
    text_only = clip.tokenize("a large metal cube")
    key = next(iter(text_only))
    for row in text_only[key]:
        for i, entry in enumerate(row):
            if isinstance(entry, tuple) and len(entry) == 2 and entry[1] == 1.0:
                row[i] = (entry[0], 1.3)
                break
        break
    with torch.no_grad():
        clip.encode_from_tokens_scheduled(text_only)

    # Deterministic across runs, and the encode restored the stock state.
    cond2, _ = nodes.CVRRTextEncode().encode(clip=patched, prompt="a small robot",
                                             mode="aligned", vl_megapixels=0.0, image=image)
    assert torch.equal(cond2[0][0], tensor)
    with torch.no_grad():
        assert torch.equal(clip.encode_from_tokens_scheduled(tokens)[0][0], baseline[0][0])

    # Re-attaching refreshes in place instead of wrapping twice.
    patched2, _ = nodes.CVRRApplyTransition().apply(
        clip=patched, merged_transition="cvrr/cvrr_merged_transition.safetensors")
    assert nodes.is_cvrr_clip(patched2)


def test_apply_transition_requires_a_vision_tower(comfy, nodes_module, monkeypatch,
                                                  tmp_path, tiny_spec_in_nodes):
    spec = tiny_spec_in_nodes
    clip = _stock_tiny_clip(comfy)
    path = _write_transition(clip, spec, tmp_path)
    _serve_file(monkeypatch, nodes_module, "cvrr/cvrr_merged_transition.safetensors", path)

    model = getattr(clip.cond_stage_model, clip.cond_stage_model.clip)
    del model.transformer.visual  # simulate a text-only Qwen3 encoder (Klein's stock TE)

    with pytest.raises(ValueError, match="vision"):
        nodes_module.CVRRApplyTransition().apply(
            clip=clip, merged_transition="cvrr/cvrr_merged_transition.safetensors")


def test_apply_transition_rejects_shape_mismatched_weights(comfy, nodes_module,
                                                           monkeypatch, tmp_path,
                                                           tiny_spec_in_nodes):
    from safetensors.torch import load_file, save_file

    spec = tiny_spec_in_nodes
    clip = _stock_tiny_clip(comfy)
    path = _write_transition(clip, spec, tmp_path)
    sd = load_file(str(path))
    first_key = sorted(sd)[0]
    sd[first_key] = sd[first_key].narrow(0, 0, max(1, sd[first_key].shape[0] - 1)).clone()
    save_file(sd, str(path))
    _serve_file(monkeypatch, nodes_module, "cvrr/cvrr_merged_transition.safetensors", path)

    with pytest.raises(ValueError, match="mismatch"):
        nodes_module.CVRRApplyTransition().apply(
            clip=clip, merged_transition="cvrr/cvrr_merged_transition.safetensors")
