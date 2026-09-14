# Qwen3-VL-8B-CVRR × ComfyUI

ComfyUI integration for [`dmis-lab/Qwen3-VL-8B-CVRR`](https://huggingface.co/dmis-lab/Qwen3-VL-8B-CVRR)
(*CVRR* — a recurrent visual-reasoning Qwen3-VL-8B), built so that its
conditioning can drive **FLUX.2 Klein 9B** editing. The released checkpoint is a
text encoder that has *looked at the reference image*: it repeatedly refines a
question state through a shared recurrent decoder layer while the visual tokens
stay in context.

* **Feasibility write-up:** [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md)
* **Example graph:** [`examples/klein9b_cvrr_edit.json`](examples/klein9b_cvrr_edit.json)
* **Tests:** 42 CPU tests (`pytest` in this directory), run against the real
  ComfyUI modules — no ComfyUI patching, nothing is monkey-patched at import.

## Why this exists

Klein 9B's text tower is a plain Qwen3-8B (no vision). CVRR's tower is
Qwen3-VL-8B with a recurrent visual path, and it emits a 4096-d conditioning
tensor — the same slot Klein expects. This pack supplies the missing half:
encode with vision, and hand the result to Klein together with the reference
latents it already understands.

```
CVRRTextEncoderLoader ── clip ──► CVRREditTextEncode ── positive/negative ──► KSampler
        (converted encoder + merged_transition)   │
                                                  └─ adds reference latents (VAE-encoded image)
```

## Install

```bash
git clone <this repo> ~/ComfyUI/custom_nodes/ComfyUI-CVRR
# or: ln -s "$PWD" ~/ComfyUI/custom_nodes/ComfyUI-CVRR
```

ComfyUI will pick up the node pack on restart (nodes appear under
`model/conditioning/cvrr`). No ComfyUI source changes are needed; all ComfyUI
classes are *subclassed*, and the stock path stays the default.

## Convert the released checkpoint

Download `dmis-lab/Qwen3-VL-8B-CVRR` (~17.5 GB) from Hugging Face, then:

```bash
python -m comfyui_cvrr.convert_cvrr_to_comfy \
    --release-dir ~/models/Qwen3-VL-8B-CVRR \
    --output-dir  ~/ComfyUI/models/text_encoders/cvrr \
    --dtype bf16 --verify
```

This writes

```
models/text_encoders/cvrr/cvrr_qwen3vl_8b-00001.safetensors (+ index)   the encoder
models/text_encoders/cvrr/cvrr_merged_transition.safetensors           the recurrent layer (fp32)
models/text_encoders/cvrr/cvrr_release_config.json                     ell_star / T / beta
```

Only the 7 projections of decoder layer 23 differ from a stock Qwen3-VL-8B
checkpoint; everything else is a key-prefix rewrite ComfyUI already understands.
`--verify` checks every converted tensor against ComfyUI's real parameter names.
Add `--dry-run` to inspect the mapping without writing anything.

> Keep the merged transition in FP32 (`--dtype keep` for the transition) — the
> release stores it that way on purpose and ComfyUI runs text encoders in fp32
> compute.

## Nodes

| Node | Purpose |
|---|---|
| **CVRR Qwen3-VL Text Encoder Loader** | Loads the converted encoder + `cvrr_merged_transition.safetensors`; `mode` = `aligned` (default) / `strict` / `vl`; optional `blocksize`, `inference_steps`, `beta`, `tap_gains` overrides; reports the resolved CVRR geometry. |
| **CVRR Apply Merged Transition** | **LoRA-style attach:** patches `merged_transition.safetensors` onto *any* already-loaded Qwen3-VL CLIP (stock `CLIPLoader` output with type `flux2`, or a Qwen3-VL-8B finetune — no converted file per finetune needed). Returns a new CLIP handle; the merged weights sit as gated fp32 buffers on the recurrent layer, so non-CVRR encodes of the same encoder are untouched. |
| **CVRR Text Encode (Qwen3-VL / Klein)** | `CLIP Text Encode (Prompt)` for CVRR: prompt + optional image → `CONDITIONING` (+ info). |
| **CVRR Edit Text Encode (Klein ref-latent)** | The Klein edit node: prompt + image + VAE → positive/negative `CONDITIONING`, **with the reference latent already appended** (equivalent to `CLIP Text Encode` → `VAEEncode` → `Set Reference Latent`). Optional second image, `reference_latents_method`, per-tap gains. |
| **CVRR Set Reference Latent+** | Chainable `Reference Latent+`: appends one more reference latent to a conditioning and returns it. |

`CVRREditTextEncode` and `CVRRSetReferenceLatent` both write
`conditioning_set_values(..., {"reference_latents": [...]}, append=True)` — the
exact metadata FLUX.2's sampler reads — so no downstream node needs to know
CVRR exists.

## Use (FLUX.2 Klein 9B edit)

1. **CVRR Qwen3-VL Text Encoder Loader** → `cvrr/cvrr_qwen3vl_8b-00001.safetensors`,
   `mode = aligned`, `cvrr/cvrr_merged_transition.safetensors`.
2. Load `flux-2-klein-9b-fp8.safetensors` + `flux2-vae.safetensors` as usual.
3. **CVRR Edit Text Encode (Klein ref-latent)**: prompt, your image, the VAE, `vl_megapixels ≈ 0.25`
   (what CVRR sees), `reference_megapixels ≈ 1.0` (what the VAE encodes).
4. `Empty Flux 2 Latent` → **KSampler** (Klein 9B distilled: 4 steps, cfg 1.0)
   → `VAE Decode`.

More reference images: feed the conditioning through extra
**CVRR Set Reference Latent+** nodes, or use the node's second image input.

### Alternative: LoRA-style attach to any Qwen3-VL-8B encoder

The converted full checkpoint from step 1 is optional. The only CVRR-specific
weight file is `merged_transition.safetensors` (772 MB), so you can attach it
to any Qwen3-VL-8B text encoder you already have — including finetunes that
kept the architecture:

1. **CLIPLoader** (`type = flux2`) → your `qwen3vl-8b*-finetuned.safetensors`.
2. **CVRR Apply Merged Transition** → `cvrr/cvrr_merged_transition.safetensors`
   (drop it into `models/text_encoders/`). This returns a new CLIP handle.
3. Continue at step 3 of the recipe above with that handle.

The node validates before attaching (must be a Qwen3-**VL** stack with a
vision tower — Klein's stock `qwen_3_8b` text-only TE is rejected with a clear
error — and the seven projection shapes must match). Attaching is inert for
regular encodes: the transition weights are runtime-gated buffers, so a CLIP
handle that never goes through the CVRR encode nodes behaves exactly like the
stock encoder. Two notes:

- The underlying encoder object is shared between CLIP handles cloned from
  the same loader output, so attaching a *different* transition file replaces
  the previous one for all of them (same caveat as any LoRA-relevant mutation
  of a cached model; attaching the same file twice is just a refresh).
- **Quantized base encoders are supported.** The merged weights stay fp32
  regardless of base storage, and the base projections are used through their
  own forward (so their dequantization keeps working). Verified in tests:
  bf16 (plain cast), fp8 `float8_e4m3fn` (scaled, mixed-precision ops),
  `int8_tensorwise` including **convrot** (per-channel, Hadamard-rotated).
  Other `QuantizedTensor` layouts (e5m2, NVFP4, W4A8/AWQ…) work by the same
  contract: the projection must expose its logical-shape `.weight` and a
  dequantizing forward; anything past-load ComfyUI produces satisfies that.
- The released transition was trained against the instruct Qwen3-VL-8B
  backbone; on a diverged finetune the mechanics hold, but the visual
  recurrence targets different activations — expect the quality question mark
  to grow with the distance of the finetune.

## Tests

```bash
COMFYUI_PATH=~/ComfyUI pytest     # 42 tests, CPU only, ~2 GB RAM
```

`COMFYUI_PATH` defaults to `../ComfyUI_src`; without a checkout the
ComfyUI-backed tests are skipped instead of failing. The suite runs the
*algorithm* and the *integration* against real ComfyUI classes at tiny
dimensions (a stand-in Qwen3-VL config registered under a private `model_type`),
so it needs a ComfyUI checkout but no GPU and no weights. It covers
algorithm↔reference equivalence, the merged-transition machinery, Klein's token
layout and attention mask, loader round-trips, and the converter's key mapping.

## Status & limitations

* **Not yet validated on a GPU with the real 17.5 GB checkpoint** — the CPU
  tests prove the plumbing, not the output quality. See §7–8 of the
  feasibility document for the validation plan.
* CVRR is **inference-only**: greedy, no `device_map="auto"`, no
  `save_pretrained()`. The converter works around this by rewriting keys, not by
  re-serialising the model.
* **Prompt weighting is rejected** on CVRR conditioning (the emitted token count
  differs from the tokenized prompt) — the encoder raises instead of silently
  mis-encoding.
* Flowing CVRR states into Klein's TE slot is a **distribution shift** for the
  diffusion model (Klein was trained on plain Qwen3-8B states). It is valid
  conditioning; whether it improves edits is the experiment this repository
  exists to run.
