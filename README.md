# Qwen3-VL-8B-CVRR × ComfyUI

ComfyUI integration for [`dmis-lab/Qwen3-VL-8B-CVRR`](https://huggingface.co/dmis-lab/Qwen3-VL-8B-CVRR)
(*CVRR* — a recurrent visual-reasoning Qwen3-VL-8B), built so that its
conditioning can drive **FLUX.2 Klein 9B** editing. The released checkpoint is a
text encoder that has *looked at the reference image*: it repeatedly refines a
question state through a shared recurrent decoder layer while the visual tokens
stay in context.

* **Feasibility write-up:** [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md)
* **Example graph:** [`examples/klein9b_cvrr_edit.json`](examples/klein9b_cvrr_edit.json)
* **Tests:** 35 CPU tests (`pytest` in this directory), run against the real
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
| **CVRR Apply Merged Transition** | Adds/refreshes the recurrent transition on any CLIP produced by another loader. |
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

## Tests

```bash
pytest                       # 35 tests, CPU only, ~2 GB RAM
```

The suite runs the algorithm and the integration against real ComfyUI classes at
tiny dimensions (a stand-in Qwen3-VL config), so it needs a ComfyUI checkout on
`PYTHONPATH` (or installed) but no GPU and no weights. It covers
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
