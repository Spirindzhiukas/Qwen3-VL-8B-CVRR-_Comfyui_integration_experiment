"""Shared fixtures for the comfyui_cvrr test-suite.

The tests run against a *real* ComfyUI checkout (``COMFYUI_PATH``, default
``../ComfyUI_src``) in which a reduced Qwen3-VL-8B configuration is registered
under a private ``model_type``.  Everything else -- the layer stack, the vision
tower with DeepStack, the tokenizer, the text-encoder wrapper classes, the
sampler-facing ``CLIP`` object -- is genuine ComfyUI code, so these tests
exercise the real integration points on a CPU-sized model.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

COMFY_PATH = os.environ.get(
    "COMFYUI_PATH", os.path.abspath(os.path.join(REPO, "..", "ComfyUI_src"))
)
HAS_COMFY = os.path.isdir(COMFY_PATH)

#: A Qwen3-VL-8B-shaped model small enough for CPU tests.  ``hidden_size``,
#: ``num_attention_heads`` and ``head_dim`` must stay consistent: the vision
#: tower derives its rotary dimension from ``hidden_size // num_heads // 2``.
TINY_MODEL_TYPE = "cvrr_tiny_test"
TINY_TAPS = (1, 3, 5)
TINY_ELL_STAR = 2          # -> recurrent layer 3, upper decoder 4
TINY_LAYERS = 6
TINY_DIM = 32


def pytest_collection_modifyitems(config, items):
    if HAS_COMFY:
        return
    skip = pytest.mark.skip(reason=f"no ComfyUI checkout at {COMFY_PATH} (set COMFYUI_PATH)")
    for item in items:
        if "comfy" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def comfy():
    """Import ComfyUI and register the tiny Qwen3-VL config."""
    if not HAS_COMFY:
        pytest.skip(f"no ComfyUI checkout at {COMFY_PATH} (set COMFYUI_PATH)")
    if COMFY_PATH not in sys.path:
        sys.path.insert(0, COMFY_PATH)
    import comfy.options

    # ComfyUI parses sys.argv the moment comfy.cli_args is imported; keep
    # pytest's own arguments out of its parser.
    comfy.options.args_parsing = False
    import comfy.cli_args

    comfy.cli_args.args.cpu = True  # do not initialise CUDA on CPU-only machines
    import comfy.text_encoders.qwen3vl as qwen3vl

    if TINY_MODEL_TYPE in qwen3vl.QWEN3VL_CONFIGS:
        return qwen3vl

    # NOTE: @dataclass is required -- without it the parent dataclass __init__
    # would re-apply the 8B defaults and silently build a 4096-wide model.
    @dataclass
    class TinyVLConfig(qwen3vl.Qwen3VL_8BConfig):
        vocab_size: int = 151936  # keep the real tokenizer's token ids valid
        hidden_size: int = TINY_DIM
        intermediate_size: int = 64
        num_hidden_layers: int = TINY_LAYERS
        num_attention_heads: int = 4
        num_key_value_heads: int = 2
        head_dim: int = 8
        max_position_embeddings: int = 4096

    # NB: assign *after* the class body.  Inside the body this name would become
    # a dataclass field whose default (None) shadows the class attribute, and
    # ``precompute_freqs_cis`` would then skip its interleaved-M-RoPE path and
    # produce frequencies with a leading MRoPE-axis dimension.
    TinyVLConfig.rope_dims = [2, 1, 1]

    qwen3vl.QWEN3VL_CONFIGS[TINY_MODEL_TYPE] = TinyVLConfig
    qwen3vl.QWEN3VL_VISION[TINY_MODEL_TYPE] = dict(
        hidden_size=64, num_heads=4, intermediate_size=128, depth=2,
        deepstack_visual_indexes=[0],
    )
    return qwen3vl


@pytest.fixture(scope="session")
def spec(comfy):
    from comfyui_cvrr.cvrr_core import CVRRSpec

    return CVRRSpec(
        ell_star=TINY_ELL_STAR, num_recurrent_steps=4, beta=0.33, taps=TINY_TAPS
    )


@pytest.fixture(scope="session")
def tokenizer(comfy):
    return comfy.tokenizer(TINY_MODEL_TYPE)(embedding_directory=None, tokenizer_data={})


def stock_te_class(comfy, model_type: str = TINY_MODEL_TYPE):
    """ComfyUI's own ``Qwen3VL`` bound to the tiny config.

    ``model_type`` is a *class* attribute read inside ``Qwen3VL.__init__``, so it
    has to be set on the class -- assigning it on an instance would build the
    full 4096-wide 8B model.
    """
    return type("TinyStockQwen3VL", (comfy.Qwen3VL,), {"model_type": model_type})


def random_transition(model, spec, seed: int = 0):
    """Stand-in ``merged_transition.safetensors`` contents for the tiny model."""
    import torch

    from comfyui_cvrr.cvrr_core import MERGED_PROJECTION_PATHS

    torch.manual_seed(seed)
    layer = model.transformer.model.layers[spec.recurrent_layer]
    weights = {}
    for path in MERGED_PROJECTION_PATHS:
        parent = layer
        parts = path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        weights[f"{path}.weight"] = torch.randn_like(getattr(parent, parts[-1]).weight) * 0.2
    return weights
