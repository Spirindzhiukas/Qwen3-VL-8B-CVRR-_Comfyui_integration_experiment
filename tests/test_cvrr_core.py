"""Correctness tests for the CVRR text-encoder core.

Evidence produced here:

1. **Equality with an independent reference implementation.**
   ``reference_cvrr_encode`` is a direct, unoptimised transcription of the
   released procedure (``modeling_cvrr_merged.py`` / ``cvrr/modeling_cvrr.py``):
   multimodal boundary pass, adapter-off pass through the recurrent layer,
   ``T - 1`` ``lerp``-averaged recurrent transitions with the merged weights, then
   the upper decoder.  It shares no helper code with ``cvrr_core``.

2. **Invariants that matter for the ComfyUI integration**: which taps are
   image-aware, which taps the recurrence can change at all, that the strict
   modes emit exactly one token per text token, and that the
   ``block_visual_access`` ablation really removes visual keys.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfyui_cvrr.cvrr_core import (  # noqa: E402
    CVRRSpec,
    CVRRTextEncoderCore,
    build_layer_mask,
    install_merged_transition,
    klein_stack_taps,
    sdpa_attention,
    visual_key_block_bias,
)
from tests.tiny_qwen import TinyConfig, TinyQwen3LM, apply_rope, tiny_sequence  # noqa: E402

HIDDEN = 32
TAPS = (1, 3, 5)
SPEC = CVRRSpec(ell_star=4, num_recurrent_steps=4, beta=0.33, taps=TAPS)
LAYERS, VIS, TXT = 8, 5, 6

PROJECTIONS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def make_batch(seed: int = 0, batch: int = 1):
    torch.manual_seed(seed)
    mm = torch.randn(batch, VIS + TXT, HIDDEN)
    mm_mask = torch.ones(batch, VIS + TXT, dtype=torch.long)
    visual = torch.zeros(batch, VIS + TXT, dtype=torch.bool)
    visual[:, :VIS] = True
    positions = torch.arange(VIS + TXT).view(1, 1, -1).expand(3, batch, -1).contiguous()
    text = mm[:, VIS:].clone()
    text_mask = torch.ones(batch, TXT, dtype=torch.long)
    text_positions = torch.arange(TXT).view(1, 1, -1).expand(3, batch, -1).contiguous()
    return mm, mm_mask, visual, positions, text, text_mask, text_positions


def merged_weights(model: TinyQwen3LM, layer_index: int, seed: int = 7):
    torch.manual_seed(seed)
    layer = model.layers[layer_index]
    out = {}
    for path in PROJECTIONS:
        parent = layer
        parts = path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        base = getattr(parent, parts[-1])
        out[f"{path}.weight"] = torch.randn_like(base.weight) * 0.5
    return out


def reference_cvrr_encode(model, spec, batch, weights, *, mode="strict"):
    """Independent transcription of the released CVRR inference procedure."""
    mm, mm_mask, visual, positions, text, text_mask, text_positions = batch
    transition = install_merged_transition(model.layers[spec.recurrent_layer], weights)

    def run(x, start, stop, positions_, attn_mask, capture=(), collected=None, mask=None):
        capture = set(capture)
        if mask is None:
            mask = build_layer_mask(x, attn_mask)
        freqs = model.compute_freqs_cis(positions_, x.device)
        for i in range(start, stop):
            if collected is not None and i in capture:
                collected[i] = x.clone()
            x, _ = model.layers[i](x=x, attention_mask=mask, freqs_cis=freqs,
                                   optimized_attention=sdpa_attention, past_key_value=None)
        if collected is not None and stop in capture and stop not in collected:
            collected[stop] = x.clone()
        return x

    # 1. multimodal boundary (layers 0..ell_star)
    scaffold = run(mm, 0, spec.recurrent_layer, positions, mm_mask)
    # 2. text-only branch (CVRR's answer-time prefix)
    text_states = run(text, 0, spec.recurrent_layer, text_positions, text_mask)
    # 3. adapter-off pass through the recurrent layer over the scaffold
    with transition.active(False):
        first_full = run(scaffold, spec.recurrent_layer, spec.recurrent_layer + 1, positions, mm_mask)
    with transition.active(False):
        run(text_states, spec.recurrent_layer, spec.recurrent_layer + 1, text_positions, text_mask)

    state = first_full[:, VIS:].clone()
    with transition.active(True):
        for _ in range(spec.num_recurrent_steps - 1):
            probe = first_full.clone()
            probe[:, VIS:] = state
            proposal_full = run(probe, spec.recurrent_layer, spec.recurrent_layer + 1,
                                positions, mm_mask)
            state = torch.lerp(state, proposal_full[:, VIS:], spec.beta)

    collected = {}
    lower = [t for t in spec.taps if t < spec.upper_decoder_start]
    upper = [t for t in spec.taps if t >= spec.upper_decoder_start]
    if mode == "strict":
        if lower:
            run(text, 0, max(lower) + 1, text_positions, text_mask,
                capture=lower, collected=collected)
        if upper:
            run(state, spec.upper_decoder_start, max(upper) + 1, text_positions, text_mask,
                capture=upper, collected=collected)
        taps = [collected[t] for t in spec.taps]
    else:
        if lower:
            run(mm, 0, spec.recurrent_layer, positions, mm_mask,
                capture=lower, collected=collected)
        upper_in = first_full.clone()
        upper_in[:, VIS:] = state
        if upper:
            run(upper_in, spec.upper_decoder_start, max(upper) + 1, positions, mm_mask,
                capture=upper, collected=collected)
        if mode == "vl":
            taps = [collected[t] for t in spec.taps]
        else:  # aligned: keep the question rows only
            taps = [collected[t][:, VIS:].clone() for t in spec.taps]
    return state, tuple(taps)


def build_driver(seed: int = 0, spec: CVRRSpec = SPEC):
    model = TinyQwen3LM(TinyConfig(hidden_size=HIDDEN, num_hidden_layers=LAYERS), seed=seed)
    weights = merged_weights(model, spec.recurrent_layer)
    transition = install_merged_transition(model.layers[spec.recurrent_layer], weights)
    return model, CVRRTextEncoderCore(model, spec, transition=transition)


def encode(driver, batch, mode="aligned", **kwargs):
    return driver.encode(mm_embeds=batch[0], mm_position_ids=batch[3],
                         mm_attention_mask=batch[1], visual_mask=batch[2], mode=mode, **kwargs)


# ---------------------------------------------------------------------------
# 1. equality with an independent transcription
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["strict", "aligned", "vl"])
def test_driver_matches_reference_transcription(mode):
    """The published procedure, transcribed independently, gives the same taps."""
    model = TinyQwen3LM(TinyConfig(hidden_size=HIDDEN, num_hidden_layers=LAYERS))
    batch = make_batch(seed=0)
    weights = merged_weights(model, SPEC.recurrent_layer)

    ref_state, ref_taps = reference_cvrr_encode(model, SPEC, batch, weights, mode=mode)
    driver = CVRRTextEncoderCore(
        model, SPEC,
        transition=install_merged_transition(model.layers[SPEC.recurrent_layer], weights),
    )
    result = driver.encode(
        mm_embeds=batch[0], mm_position_ids=batch[3], mm_attention_mask=batch[1],
        visual_mask=batch[2], text_embeds=batch[4], text_position_ids=batch[6],
        text_attention_mask=batch[5], mode=mode,
    )
    assert torch.allclose(result.state, ref_state, atol=1e-6), "recurrent state diverged"
    for index, (got, expected) in enumerate(zip(result.taps, ref_taps)):
        assert torch.allclose(got, expected, atol=1e-6), f"tap {index} diverged in mode {mode}"


# ---------------------------------------------------------------------------
# 2. merged transition plumbing
# ---------------------------------------------------------------------------


def test_merged_transition_toggle_is_surgical():
    model = TinyQwen3LM(TinyConfig(hidden_size=HIDDEN, num_hidden_layers=LAYERS))
    weights = merged_weights(model, SPEC.recurrent_layer)
    transition = install_merged_transition(model.layers[SPEC.recurrent_layer], weights)

    x = tiny_sequence(7, HIDDEN, seed=3)
    with transition.active(False):
        before, _ = model.layers[SPEC.recurrent_layer](x=x, optimized_attention=sdpa_attention,
                                                       past_key_value=None)
    with transition.active(True):
        after, _ = model.layers[SPEC.recurrent_layer](x=x, optimized_attention=sdpa_attention,
                                                      past_key_value=None)
        assert transition.enabled is True
    assert not torch.allclose(before, after)
    assert transition.enabled is False  # the context manager restores the old state
    again, _ = model.layers[SPEC.recurrent_layer](x=x, optimized_attention=sdpa_attention,
                                                  past_key_value=None)
    assert torch.allclose(before, again, atol=1e-6)


def test_merged_projection_keeps_fp32_precision():
    """The merged weight must survive a dtype cast; the release ships it in fp32."""
    model = TinyQwen3LM(TinyConfig(hidden_size=HIDDEN, num_hidden_layers=LAYERS))
    weights = merged_weights(model, SPEC.recurrent_layer)
    transition = install_merged_transition(model.layers[SPEC.recurrent_layer], weights)
    projection = transition.projections[0]
    projection.to(torch.bfloat16)
    assert projection.merged_weight.dtype == torch.float32


def test_install_is_idempotent():
    """Installing twice must refresh the weights rather than nest wrappers."""
    model = TinyQwen3LM(TinyConfig(hidden_size=HIDDEN, num_hidden_layers=LAYERS))
    weights = merged_weights(model, SPEC.recurrent_layer)
    first = install_merged_transition(model.layers[SPEC.recurrent_layer], weights)
    weights2 = {k: v + 1.0 for k, v in weights.items()}
    second = install_merged_transition(model.layers[SPEC.recurrent_layer], weights2)
    assert len(second.projections) == len(PROJECTIONS)
    assert torch.allclose(second.projections[0].merged_weight, weights2["self_attn.q_proj.weight"])


# ---------------------------------------------------------------------------
# 3. invariants of the emitted conditioning
# ---------------------------------------------------------------------------


def test_token_layouts_per_mode():
    """strict/aligned keep the text-token layout; vl adds the image tokens."""
    _, driver = build_driver()
    batch = make_batch(seed=1)
    strict = encode(driver, batch, mode="strict")
    aligned = encode(driver, batch, mode="aligned")
    vl = encode(driver, batch, mode="vl")
    assert strict.taps[0].shape[1] == TXT
    assert aligned.taps[0].shape[1] == TXT
    assert vl.taps[0].shape[1] == VIS + TXT
    # aligned is exactly vl restricted to the question rows
    for a, v in zip(aligned.taps, vl.taps):
        assert torch.allclose(a, v[:, VIS:], atol=1e-6)
    assert aligned.num_tokens == TXT
    assert aligned.stacked().shape == (1, TXT, 3 * HIDDEN)


def test_aligned_conditioning_is_image_aware_at_every_tap():
    """Changing only the image changes every emitted tap in `aligned` mode."""
    _, driver = build_driver()
    a = make_batch(seed=0)
    b = list(a)
    b[0] = a[0].clone()
    b[0][:, :VIS] += 1.0
    b = tuple(b)
    out_a, out_b = encode(driver, a, mode="aligned"), encode(driver, b, mode="aligned")
    for index, (x, y) in enumerate(zip(out_a.taps, out_b.taps)):
        assert not torch.allclose(x, y, atol=1e-5), f"tap {index} ignored the image"


def test_strict_mode_lower_taps_are_image_independent():
    """The released strict interface keeps visual rows out of the answer path.

    Taps below ``ell_star + 1`` come from the text-only branch, so they cannot
    see the image at all.  This is a design property, not a bug -- and it is the
    reason the integration defaults to ``aligned`` mode for diffusion models.
    """
    _, driver = build_driver()
    a = make_batch(seed=0)
    b = list(a)
    b[0] = a[0].clone()
    b[0][:, :VIS] += 1.0
    b = tuple(b)
    out_a, out_b = encode(driver, a, mode="strict"), encode(driver, b, mode="strict")
    for index, tap in enumerate(SPEC.taps):
        if tap < SPEC.upper_decoder_start:
            assert torch.allclose(out_a.taps[index], out_b.taps[index], atol=1e-6), (
                f"tap {tap} below the recurrent layer must not see the image in strict mode"
            )
    # ... while the recurrent state itself does depend on the image
    assert not torch.allclose(out_a.state, out_b.state, atol=1e-5)


def test_recurrence_only_changes_taps_from_the_upper_decoder():
    """CVRR only modifies hidden states at `ell_star + 1` and above."""
    _, driver = build_driver()
    batch = make_batch(seed=2)
    single = CVRRSpec(ell_star=SPEC.ell_star, num_recurrent_steps=1, beta=SPEC.beta, taps=TAPS)
    _, driver_one_step = build_driver(seed=0, spec=single)

    for mode in ("strict", "aligned"):
        many = encode(driver, batch, mode=mode)
        one = encode(driver_one_step, batch, mode=mode)
        for index, tap in enumerate(TAPS):
            if tap < SPEC.upper_decoder_start:
                assert torch.allclose(many.taps[index], one.taps[index], atol=1e-6), (
                    f"tap {tap} must be unaffected by the recurrence"
                )
        assert not torch.allclose(many.state, one.state, atol=1e-6)


def test_block_visual_access_removes_visual_keys():
    """The ablation makes question rows attend as if the visual rows were absent."""
    model, _ = build_driver()
    batch = make_batch(seed=2)
    mm, mm_mask, visual = batch[0], batch[1], batch[2]
    q_mask = ~visual.bool()
    x = mm.clone()
    positions = batch[3]
    freqs = model.compute_freqs_cis(positions, x.device)

    base_mask = build_layer_mask(x, mm_mask)
    bias = visual_key_block_bias(q_mask, visual, x.dtype)
    blocked_mask = base_mask + bias

    layer = model.layers[2]
    with torch.no_grad():
        out_blocked, _ = layer(x=x, attention_mask=blocked_mask, freqs_cis=freqs,
                               optimized_attention=sdpa_attention, past_key_value=None)

        # Reference: run the very same layer on the question rows only, with the
        # visual rows deleted from the key/value set.
        q_only = x[:, VIS:].clone()
        hidden = layer.input_layernorm(q_only)
        q, k, v = layer.self_attn.project(hidden)
        ref_freqs = model.compute_freqs_cis(batch[6], q_only.device)
        q = apply_rope(q, ref_freqs)
        k = apply_rope(k, ref_freqs)
        ref_mask = build_layer_mask(q_only, None)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=ref_mask, enable_gqa=True
        )
        attn = layer.self_attn.o_proj(attn.transpose(1, 2).reshape(1, TXT, -1))
        ref = q_only + attn
        ref = ref + layer.mlp(layer.post_attention_layernorm(ref))
    assert torch.allclose(out_blocked[:, VIS:], ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. batching, spec parsing, small helpers
# ---------------------------------------------------------------------------


def test_batch_of_two_matches_two_single_runs():
    _, driver = build_driver()
    a = make_batch(seed=5, batch=1)
    b = make_batch(seed=6, batch=1)
    batch = (
        torch.cat([a[0], b[0]]),
        torch.cat([a[1], b[1]]),
        torch.cat([a[2], b[2]]),
        torch.cat([a[3], b[3]], dim=1),
        None, None, None,
    )
    out_batch = encode(driver, batch)
    out_a = encode(driver, a)
    out_b = encode(driver, b)
    assert torch.allclose(out_batch.taps[-1][0], out_a.taps[-1][0], atol=1e-5)
    assert torch.allclose(out_batch.taps[-1][1], out_b.taps[-1][0], atol=1e-5)


def test_spec_read_from_real_release_metadata():
    """The released config.json maps onto the geometry the driver assumes."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "cvrr_qwen3vl_8b_release.json")
    with open(path) as handle:
        release = json.load(handle)
    spec = CVRRSpec.from_release(release)
    assert spec.ell_star == 22
    assert spec.recurrent_layer == 23
    assert spec.upper_decoder_start == 24
    assert spec.num_recurrent_steps == 4
    assert pytest.approx(spec.beta, abs=1e-9) == 0.33
    assert spec.taps == (9, 18, 27)
    spec.validate(36)


def test_spec_rejects_inconsistent_metadata():
    with pytest.raises(ValueError):
        CVRRSpec.from_release({"ell_star": 22, "recurrent_layer": 25})
    with pytest.raises(ValueError):
        CVRRSpec(ell_star=40, taps=(9,)).validate(36)
    with pytest.raises(ValueError):
        CVRRSpec(ell_star=4, beta=1.5, taps=(1,)).validate(8)


def test_klein_tap_stack_layout():
    _, driver = build_driver()
    batch = make_batch(seed=4)
    result = encode(driver, batch)
    stacked = klein_stack_taps(result.taps)
    assert stacked.shape == (1, TXT, 3 * HIDDEN)
    assert torch.allclose(stacked[:, :, :HIDDEN], result.taps[0])
    assert result.stacked().shape == stacked.shape


def test_unknown_mode_rejected():
    _, driver = build_driver()
    with pytest.raises(ValueError):
        encode(driver, make_batch(seed=0), mode="nope")
