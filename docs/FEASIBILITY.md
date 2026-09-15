# CVRR in ComfyUI — feasibility assessment and implementation plan

*Assessment of `dmis-lab/Qwen3-VL-8B-CVRR` ("CVRR", *Reason Through the Latent!*)
as a text encoder / reference encoder for ComfyUI, with the concrete goal of
giving FLUX.2 Klein 9B a vision-aware conditioning path.*

Everything below was established against real sources: the released weights and
config files of `dmis-lab/Qwen3-VL-8B-CVRR`, the `dmis-lab/CVRR` reference
implementation, and a checkout of ComfyUI (`b0058496`).

---

## 1. Verdict

**Yes — it is feasible, and the hard part is smaller than it looks.**

| Question | Answer |
|---|---|
| Can the released weights be loaded by ComfyUI? | **Yes.** The released `native_backbone/` is an unmodified Qwen3-VL-8B; its text config is *field-for-field identical* to ComfyUI's `Qwen3VL_8BConfig` and its vision config to `QWEN3VL_VISION["qwen3vl_8b"]`. The only genuinely new weights are `merged_transition.safetensors` (772 MB, FP32) belonging to **one** decoder layer (23). |
| Can it go through the regular `CLIPLoader`? | **Not the release folder as-is** — it declares a custom `model_type: "cvrr_merged"` and an inference-only `from_pretrained`. But a *converted* file is an ordinary Qwen3-VL text encoder, so the regular loader can be made to accept it with a small upstream patch (option A). |
| Do we need custom nodes? | **Yes, for the recurrence.** No stock ComfyUI code path runs two aligned encoder branches and T recurrent passes. Option B (node pack) delivers this today; option A (upstream) needs a new `CLIPType`. |
| Do we need new reference-encode nodes? | **Yes, and they are cheap**: they are `CLIP Text Encode (Prompt)` + `Set Reference Latent` with a CVRR-aware `clip` behind them (implemented). |
| Does it make Klein 9B "understand" reference images better? | **Mechanically yes, empirically unknown.** See §6 — this is the honest open question. |

---

## 2. What CVRR is, in three sentences

CVRR keeps the released Qwen3-VL-8B backbone frozen and adds a **recurrent
visual transition** after layer `ell_star = 22`: the decoder layer at index
`ell_star + 1 = 23` is reused as a shared transition whose weights are the
native weights *plus* a merged rank-32 LoRA (the 772 MB file). A question
embedding is run through the multimodal prompt scaffold for `T = 4` steps, each
step proposing a new question state which is blended with
`state ← lerp(state, proposal, β=0.33)`, and only the **final recurrent
question state** is decoded by layers 24…35. In the released checkpoint the
transition is already merged into dense FP32 weights, so loading it is a weight
swap, not a PEFT/LoRA integration.

Consequences that matter for ComfyUI:

* It needs **two encodings of the same prompt** — one multimodal (image + text)
  and one text-only (for `question_ids` / `question_attention_mask`) — and the
  text-only pass drives answer decoding.
* It is **inference-only**: no `save_pretrained()`, no vLLM / SGLang, no
  `device_map="auto"`, greedy only, and one full replica per GPU.
* Its `forward()` is not implemented at all — the released class only exposes
  `prepare_inputs()` / `next_token_logits()` / `generate()`. Nothing in ComfyUI
  can call it directly.

That last point is the reason a plain "load the HF folder" integration cannot
work and why the work below re-implements the sweep on top of ComfyUI's own
Qwen3-VL classes instead.

---

## 3. What the released checkpoint actually contains

```
config.json                  model_type "cvrr_merged", arch "CVRRMergedModel"
cvrr_release_config.json     ell_star 22, recurrent_layer 23, upper_decoder_start 24,
                             inference_T 4, inference_beta 0.33, lora_rank 32, ...
merged_transition.safetensors  772 MB fp32 dense transition (7 projections)
modeling_cvrr_merged.py      inference-only loader (custom from_pretrained)
native_backbone/             plain Qwen3-VL-8B: 4 shards, 17.5 GB, HF keys
  config.json                text 36L/4096/32h/8kv/128hd, mrope [24,20,20],
                             vision depth 27 / 1152 / patch 16 / deepstack [8,16,24]
```

Two facts make the integration tractable:

1. **The backbone is stock.** ComfyUI already ships the exact config
   (`QWEN3VL_VISION["qwen3vl_8b"]`, `Qwen3VL_8BConfig`) and a full
   Qwen3-VL-8B text encoder (`comfy/text_encoders/qwen3vl.py`), including
   the image → `(merged, deepstack)` preprocessing and the interleaved M-RoPE
   positions. Nothing in the vision tower or the embedding path has to change.
2. **The novelty is one layer plus a sweep.** `install_merged()` in the
   release replaces `text_model.layers[23]`'s projections with
   native-weight ⊕ merged-weight wrappers. In ComfyUI terms this is the same
   surgery on `clip.cond_stage_model.text_model.model.layers[23]`.

The merged transition is FP32 on purpose (the release re-forces fp32 in
`_apply` so a dtype cast cannot truncate it). ComfyUI already runs text encoders
with fp32 compute (`CLIP.__init__` → `patcher.set_model_compute_dtype(torch.float32)`),
so the two agree by construction.

---

## 4. Option A — through ComfyUI's regular loader (upstream)

What an upstream PR would look like (no node pack, stock nodes):

1. `comfy/text_encoders/qwen3vl.py`: add a `CVRR`-aware model class (or a
   `cvrr` flag on `Qwen3VLClipModel`) that runs the recurrent sweep instead of
   one `Llama2_.forward`. Most of the machinery is already used by other
   variants in the file (`intermediate_output`, `visual_pos_masks`,
   `deepstack_embeds`).
2. `comfy/sd.py`: a new `CLIPType` (e.g. `CVRR_QWEN3VL_8B = 36`) and a branch in
   `load_text_encoder_state_dicts` that forwards to it, exactly like the
   existing `ideogram4` / `joyimage` / `krea2` variants.
3. `nodes.py`: add the type string to `CLIPLoader`'s `type` list.
4. Detect the encoder: the release's own `cvrr_release_config.json` sitting next
   to the file is a reliable marker (it is what the converter copies).

Blockers/caveats for option A:

* The dual-encoding requirement means the encoder needs the *text-only* token
  sequence as well as the multimodal one. ComfyUI's `encode_token_weights`
  gets one token list, so the implementation has to derive the text-only pass
  internally (tokenize-free: strip the image placeholder rows from the same
  embeds). Our implementation does exactly this, which is why the node pack
  works without touching ComfyUI.
* Prompt weighting cannot be supported (the emitted token count differs from the
  tokenized prompt). Our encoder raises a clear error instead of mis-encoding.
* The stock `ClipTarget`/`CLIP` path already accepts everything else, so a
  prototype is genuinely small — this repository is that prototype.

## 5. Option B — the node pack in this repository (implemented)

`comfyui_cvrr/` is a self-contained ComfyUI custom node pack:

| File | Role |
|---|---|
| `cvrr_core.py` | algorithm only (no ComfyUI import): `CVRRSpec`, FP32 `MergedProjection` / `install_merged_transition`, the branch sweep (`CVRRTextEncoderCore.encode`) with `aligned` / `strict` / `vl` layouts, tap stacking, masks. |
| `cvrr_te.py` | the ComfyUI host: `CVRRTextModel` (subclass of `Qwen3VL`), `CVRRQwen3VLClipModel` (subclass of `SDClipModel`), `CVRRTE` (subclass of `Flux2TEModel`, i.e. Klein's `[B, L, 3×4096]` stacking), `build_clip()` (builds a real `comfy.sd.CLIP`), `apply_transition()`. |
| `nodes.py` | `CVRRTextEncoderLoader`, `CVRRApplyTransition`, `CVRRTextEncode`, `CVRREditTextEncode`, `CVRRSetReferenceLatent`. |
| `convert_cvrr_to_comfy.py` | HF release → ComfyUI text-encoder file(s) + `merged_transition.safetensors`, with `--verify` against ComfyUI's real parameter names. |

The design rule was: **never patch ComfyUI**. All ComfyUI classes are
subclassed, and stock behaviour is the default (`CVRRTextModel.forward`
delegates to `Qwen3VL.forward` unless CVRR is configured *and* an image is
present). If the pack is removed, everything reverts to stock.

## 5a. Option C — LoRA-style attach to any Qwen3-VL-8B encoder (implemented)

The conversion in §5 turns out to be optional, because the decomposition of
the release pushes that far: the only CVRR-specific *weights* are the seven
FP32 projection tensors of `merged_transition.safetensors`. So instead of a
converted checkpoint per encoder, the **CVRR Apply Transition** node can take
any already-loaded Qwen3-VL CLIP — stock `CLIPLoader` output, or a community
Qwen3-VL-8B finetune — and upgrade it in place:

```
CLIPLoader (type = flux2, any Qwen3-VL-8B file)
        └─► CVRR Apply Transition (merged_transition.safetensors) → CLIP ─► CVRR encode nodes
```

How it works without ComfyUI's blessing:

1. `clip.clone()` gives a fresh CLIP handle (ComfyUI's own API for
   "return a modified CLIP"; the patcher is isolated, the encoder module is
   shared).  The transition file is selected from `models/text_encoders` with
   the exact same combo as `CLIPLoader` — recursively, name-agnostically — and
   is validated by *contents* (the seven projection tensors), never by its
   filename or location.
2. `ensure_cvrr_module()` validates the stack — must be a `Flux2TEModel` +
   Qwen3-VL decoder *with the vision tower* (`qwen3vl.Qwen3VL.visual`), with
   enough layers for the spec — then re-types the modules to the CVRR
   subclasses (`__class__` swap; the subclasses add only plain attributes).
3. `install_merged_transition()` wraps the recurrent layer's seven projections
   in `MergedProjection`s holding the FP32 merged weight as a **non-persistent
   buffer** — so `state_dict()` is untouched and ComfyUI's weight machinery
   (ModelPatcher, offloading, extra reloads) does not see it.

Why **not** ComfyUI's actual LoRA/weight-patch machinery? A patcher diff would
apply the merge *permanently while loaded* — but CVRR's adapter-off pass
(`text_anchor`) must run on the *native* weights while the recurrent passes
run on the merged ones. That per-pass gating is exactly what
`MergedProjection.enabled` does, so the weight patch and the gating must live
at module level. Load-time patching stays the fallback story; runtime gating
is the algorithm.

Caveats (documented in the README):

- The encoder object is shared between CLIP handles, so attaching a
  *different* transition file replaces the previous one for all handles of
  that loader output; attaching the same file twice is a no-op refresh.
  Non-CVRR encodes are unaffected either way (gating, not weights, decides).
- Text-only Qwen3 encoders — Klein's stock `qwen_3_8b` TE — cannot host CVRR
  at all (no vision tower, and the encoder config itself is Qwen3 not
  Qwen3-VL); the node raises with guidance instead of failing later.  The same
  applies to *Qwen3-VL* files exported **without the vision tower**
  (text-only / `--drop-vision` conversions): never-loaded visual modules stay
  on meta tensors under dynamic VRAM and previously crashed the first image
  encode — now refused at encode time with an actionable error, and flagged
  with a warning + `cvrr_no_vision` marker at build in the dedicated loader.
- Quantized bases (bf16 casts, scaled fp8, int8 tensor-including-convrot, and
  other `QuantizedTensor` layouts) are handled by construction: the wrap
  contract needs only a logical-shape `.weight` and ComfyUI's own dequantizing
  forward; the merged weight itself always stays fp32 dense. Tests cover the
  bf16 / fp8-e4m3 / int8-convrot attach+encode roundtrip.
- Same-shape finetunes work mechanically; the release transition was trained
  against the instruct backbone's activations, so quality is a finetune-
  distance-dependent unknown (same open question as §6, amplified).
- The 772 MB stays one file for *every* finetune — no re-merging, no extra
  converted checkpoints.

### Reference-encode nodes (mapping to the examples that were named)

| Named example | What we built | Difference |
|---|---|---|
| `CLIP Text Encode (Prompt)` + `Set Reference Latent` | `CVRREditTextEncode` — one node that tokenizes the prompt *with* the image, runs CVRR, and appends the VAE-encoded reference latent via `conditioning_set_values(..., {"reference_latents": [...]}, append=True)` | Same conditioning semantics, one node instead of two; plus `mode`, VAE reference resolution, `reference_latents_method`, and a second reference image. |
| `EditUtils: Flux2Klein Edit Text Encode` (lrzjason) | `CVRRTextEncode` — prompt + optional image → CVRR conditioning + info string; `CVRRSetReferenceLatent` — chainable `ReferenceLatent` equivalent that also returns the latent it set | The CVRR encoder is the point: the conditioning is produced by the recurrent visual path, not by a static forward pass. |
| `Reference Latent+` | `CVRRSetReferenceLatent` (chainable, `index` / `offset` / `uxo` / `index_timestep_zero`) | Functionally identical; it exists so the "prompt" and "reference" steps can be split, as in the stock ComfyUI graph. |

All reference-latent plumbing is stock ComfyUI behaviour: FLUX.2's
`Flux._forward` concatenates `ref_latents` per `ref_latents_method`, so nothing
downstream needs to know CVRR exists.

### Modes

* `aligned` (default) — Klein's token layout, CVRR recurrence on the full
  multimodal scaffold, visual rows present.
* `strict` — the released inference layout (visual rows dropped from the
  emitted tokens; only the post-recurrence tap is image-aware).
* `vl` — emit all rows (diagnostics).

---

## 6. The actual goal: a vision-aware encoder for FLUX.2 Klein 9B

The intent was better image understanding / coherence when editing with
edit-capable models that sit in the Qwen3 8B family — FLUX.2 Klein 9B in
particular, whose text tower is **Qwen3-8B** (an LLM, no vision tower), while
CVRR's backbone is **Qwen3-VL-8B**.

**Why it is mechanically plausible:** both towers are 4096-dimensional Qwen3-8B
lineages; Klein's TE slot consumes a `[B, L, 3 × 4096]` conditioning tensor and
does not care how the numbers were produced. CVRR's whole point is to write
visual information *into the question token states*; that is precisely the
information a text-only encoder cannot deliver. The two combined mean "give the
diffusion model conditioning that has looked at the reference image".

**Why it is not proven:** Klein 9B's TE was trained on plain Qwen3-8B
activations from a single forward pass. CVRR's states are (a) from a *VL*
backbone, (b) recurrence-blended at layer 23, and (c) tapped at layers 9/18/27
of a sweep that ran four passes. Nothing guarantees the diffusion model's
expectation matches. Mechanically the conditioning is valid; whether it helps is
an empirical question that needs a GPU and the real weights.

**Practical consequences to keep in mind:**

* The taps are configurable (`tap_gains`, and `CVRRSpec.taps`). Tap 27 is the
  only tap downstream of the recurrence (layer 23), so it is the one that
  carries the recurrently-refined state; 9/18 are pre-recurrence context.
* Two images are supported (`image` + `image2`); more references can be added by
  chaining `CVRRSetReferenceLatent` on the conditioning from `CVRRTextEncode`.
* A cheaper, fully-supported fallback exists in the same pack: use CVRR as a
  *describer* (it is a VL model with a vision tower) and feed the resulting
  text to Klein with stock nodes. That path needs no new semantics in the
  diffusion model at all.

---

## 7. What has actually been verified

`48 tests, all green on CPU` (`pytest`, with a ComfyUI checkout at
`COMFYUI_PATH`) — see `tests/`:

* `test_cvrr_core.py` — algorithm: mode-vs-reference equality, FP32 transition
  and idempotency, token layouts, image-awareness invariants
  (`block_visual_access`), batch equivalence, release-config parsing, tap
  stacking.
* `test_comfy_integration.py` — against **real** ComfyUI classes: an
  unconfigured `CVRRTextModel` is numerically identical to `Qwen3VL`; the full
  encode runs end-to-end through `SDClipModel`/`Flux2TEModel`; image presence
  changes the output (causality); Klein's token layout and the emitted attention
  mask are correct; all three modes run.
* `test_loader.py` — a converted file loads through the real `ClipTarget` +
  `comfy.sd.CLIP`, the loaded weights match the source file, and
  `apply_transition` flips the encoder into the recurrent path.
* `test_converter.py` — a synthetic HF-layout release converts, verifies against
  ComfyUI's parameter names, and re-loads; `--transition-only` produces the
  7-tensor FP32 file.
* `test_nodes.py` — node schemas, mappings, the example workflow's widgets,
  and the §5a attach path: a *stock* `CLIPLoader`-style `Flux2TEModel` CLIP
  gets retrofitted, keeps producing bit-identical stock encodes for
  non-CVRR use, and encodes CVRR conditioning (deterministically, with the
  stock state restored afterwards); vision-tower-less encoders,
  shape-mismatched transitions and not-yet-loaded quantized weights are
  rejected with clear errors; the attach+encode roundtrip is additionally
  exercised on bf16, scaled fp8 (`float8_e4m3fn`, mixed-precision ops) and
  `int8_tensorwise` **with convrot** base encoders, proving the wrap is
  storage-format agnostic and keeps the merged weights in fp32.

Not verified (no GPU, no weights in this sandbox): real 8B inference, output
quality, speed, VRAM. The 17.5 GB backbone and the 772 MB transition were never
downloaded.

## 8. Remaining work

1. **GPU validation** with the real release: convert (`--dtype bf16` keeps the
   backbone in bf16; the merged transition stays FP32), load, run the example
   workflow, compare against `modeling_cvrr_merged.py` logits for the same
   prompt/image (the node pack and the release should agree to ~fp32 noise).
2. **A/B quality test** vs plain Klein 9B and vs Klein + stock Qwen3-8B TE,
   at fixed seeds, to answer §6.
3. **Upstream decision** — if the A/B test is positive, option A (§4) is the
   natural follow-up: ~1 new `CLIPType`, one class, one loader-list entry.
4. **Multi-reference** — currently 2 images; extend to N with per-image
   reference latents (Klein reads them sequentially).
5. **Prompt weighting** — either stay unsupported (current: explicit error) or
   map weights onto the visual-token-free layout.

## 9. Layout of this repository

```
comfyui_cvrr/          the custom node pack (see §5)
tests/                 48 CPU tests (tiny stand-in config, real ComfyUI modules)
examples/              klein9b_cvrr_edit.json — ready-to-load ComfyUI workflow
docs/FEASIBILITY.md    this document
```
