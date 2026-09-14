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
