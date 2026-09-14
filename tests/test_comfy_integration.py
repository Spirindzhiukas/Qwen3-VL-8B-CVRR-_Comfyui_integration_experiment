"""Integration tests against the *real* ComfyUI text-encoder code.

These tests import ComfyUI from ``COMFYUI_PATH`` (default ``../ComfyUI_src``) and
build a tiny Qwen3-VL-8B-shaped encoder by registering a reduced config under a
private ``model_type``.  Everything else is genuine ComfyUI code:

* ``Qwen3VL`` (LM + vision tower + DeepStack)
* ``Qwen3VLTokenizer`` (Qwen3-VL chat template with image placeholders)
* ``SDClipModel.process_tokens`` -> vision tower -> ``embeds``/``embeds_info``
* ``Flux2TEModel`` tap stacking into ``[B, tokens, 3 * D]``

So what is exercised is precisely the code path a real workflow would take, minus
the 8B weights.  The tests skip when ComfyUI (or torch) is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import (
    TINY_DIM,
    TINY_MODEL_TYPE,
    TINY_TAPS,
    random_transition,
    stock_te_class,
)

pytestmark = pytest.mark.comfy


def make_clip_model(spec):
    """A ``CVRRQwen3VLClipModel`` with the merged transition installed."""
    import comfyui_cvrr.cvrr_te as cvrr_te

    model = cvrr_te.CVRRQwen3VLClipModel(
        device="cpu", dtype=torch.float32, attention_mask=True,
        model_type=TINY_MODEL_TYPE, taps=TINY_TAPS,
    )
    model.configure_cvrr(spec=spec, mode="aligned")
    model.attach_transition(random_transition(model, spec))
    model.configure_cvrr(spec=spec, mode="aligned")
    return model


def _same(a, b, atol: float = 1e-6):
    """Structural comparison that treats matching NaNs as equal."""
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, atol) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return torch.allclose(a, b, atol=atol, equal_nan=True)


def test_forward_delegates_to_stock_encoder_without_cvrr(comfy):
    """An unconfigured CVRRTextModel must behave exactly like ComfyUI's class."""
    import comfyui_cvrr.cvrr_te as cvrr_te

    torch.manual_seed(0)
    stock = stock_te_class(comfy)({}, torch.float32, torch.device("cpu"), torch.nn)
    custom = cvrr_te.make_cvrr_qwen3vl(TINY_MODEL_TYPE)(
        {}, torch.float32, torch.device("cpu"), torch.nn
    )
    custom.load_state_dict(stock.state_dict())
    # Give both models the *same* finite weights: a freshly constructed ComfyUI
    # encoder contains uninitialised parameters, which would otherwise make the
    # comparison NaN against NaN.
    state = {key: (torch.randn_like(value) * 0.05 if value.is_floating_point() else value.clone())
             for key, value in stock.state_dict().items()}
    stock.load_state_dict(state, strict=False)
    custom.load_state_dict(state, strict=False)

    embeds = torch.randn(1, 6, TINY_DIM)
    mask = torch.ones(1, 6, dtype=torch.long)
    with torch.no_grad():
        # NB: ComfyUI's TransformerBlock writes its output into the tensor it is
        # given (``torch.add(residual, x, out=output)`` with ``output = x``), so
        # every forward needs its own copy of ``embeds``.
        a = stock(None, attention_mask=mask, embeds=embeds.clone(), intermediate_output=[1, 3, 5])
        b = custom(None, attention_mask=mask, embeds=embeds.clone(), intermediate_output=[1, 3, 5])
    assert not torch.isnan(a[0]).any()
    assert _same(a, b)


def test_end_to_end_encode_through_comfy_text_encoder(comfy, spec, tokenizer):
    """Tokenize (with an image) -> vision tower -> CVRR -> Klein-shaped taps."""
    clip_model = make_clip_model(spec)

    image = torch.rand(1, 64, 64, 3)
    prompt = "Turn the photograph into an oil painting."
    tokens = tokenizer.tokenize_with_weights(prompt, images=[image])[TINY_MODEL_TYPE]
    with torch.no_grad():
        cond, _pooled, extra = clip_model.encode_token_weights(tokens[:1])

    assert cond.shape[0] == 1
    assert cond.shape[1] == len(TINY_TAPS)      # comfy list-tap layout [B, taps, L, D]
    assert cond.shape[-1] == TINY_DIM
    emitted = cond.shape[2]
    # the mask follows the *emitted* tokens, not the tokenised prompt
    assert extra["attention_mask"].shape == (1, emitted)
    assert int(extra["attention_mask"].sum()) == emitted
    # image tokens were dropped: fewer rows than the tokeniser produced
    # (``tokens`` is a list of prompts, each a list of token entries)
    assert emitted < len(tokens[0])


def test_conditioning_depends_on_the_image_content(comfy, spec, tokenizer):
    """End-to-end causality check: same prompt, different image, different cond."""
    clip_model = make_clip_model(spec)
    prompt = "Describe the scene."

    def encode(image):
        tokens = tokenizer.tokenize_with_weights(prompt, images=[image])[TINY_MODEL_TYPE]
        with torch.no_grad():
            cond, _pooled, _extra = clip_model.encode_token_weights(tokens[:1])
        return cond

    first = encode(torch.rand(1, 64, 64, 3))
    second = encode(torch.rand(1, 64, 64, 3))
    assert first.shape == second.shape
    assert not torch.allclose(first, second, atol=1e-5)


def test_cvrrte_produces_klein_layout(comfy, spec, tokenizer):
    """The outer TE stacks taps into Klein's ``[B, tokens, 3 * D]`` layout."""
    import comfyui_cvrr.cvrr_te as cvrr_te

    te = cvrr_te.CVRRTE(device="cpu", dtype=torch.float32, model_type=TINY_MODEL_TYPE,
                        name=TINY_MODEL_TYPE, taps=TINY_TAPS)
    te.configure_cvrr(spec=spec, mode="aligned")
    te.attach_transition(random_transition(te.clip_model, spec))
    te.configure_cvrr(spec=spec, mode="aligned")

    tokens = tokenizer.tokenize_with_weights(
        "Restyle this photo.", images=[torch.rand(1, 64, 64, 3)]
    )
    with torch.no_grad():
        cond, _pooled, extra = te.encode_token_weights(tokens)
    assert cond.shape == (1, cond.shape[1], 3 * TINY_DIM)
    assert extra["attention_mask"].shape == (1, cond.shape[1])


def test_modes_and_ablation_run(comfy, spec, tokenizer):
    clip_model = make_clip_model(spec)
    tokens = tokenizer.tokenize_with_weights(
        "Edit this.", images=[torch.rand(1, 64, 64, 3)]
    )[TINY_MODEL_TYPE]

    shapes = {}
    for mode in ("strict", "aligned", "vl"):
        clip_model.transformer.configure_cvrr(spec=spec, mode=mode)
        with torch.no_grad():
            cond, _pooled, extra = clip_model.encode_token_weights(tokens[:1])
        shapes[mode] = cond.shape[2]
        assert extra["attention_mask"].shape == (1, cond.shape[2])

    assert shapes["strict"] == shapes["aligned"]
    assert shapes["vl"] > shapes["strict"]

    clip_model.transformer.configure_cvrr(spec=spec, mode="aligned", block_visual_access=True)
    with torch.no_grad():
        clip_model.encode_token_weights(tokens[:1])


def test_text_only_prompt_uses_the_stock_path(comfy, spec, tokenizer):
    """Without an image CVRR stays out of the way (Klein still gets its 3 taps)."""
    clip_model = make_clip_model(spec)
    tokens = tokenizer.tokenize_with_weights("a red cube", images=[])[TINY_MODEL_TYPE]
    with torch.no_grad():
        cond, _pooled, _extra = clip_model.encode_token_weights(tokens[:1])
    assert cond.shape[1] == len(TINY_TAPS)
    assert clip_model.transformer.last_cvrr_result is None
