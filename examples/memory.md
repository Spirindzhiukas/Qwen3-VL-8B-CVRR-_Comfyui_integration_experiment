# memory.md — examples

## Workflows

- `klein9b_cvrr_edit.json` — the **dedicated-loader** path:
  `CVRRTextEncoderLoader` → `CVRREditTextEncode` → `EmptyFlux2LatentImage` →
  `KSampler` (Klein 9B distilled: 4 steps, cfg 1.0, euler/simple, denoise 1.0)
  → `VAEDecode` → `SaveImage`; plus `UNETLoader` (flux-2-klein-9b-fp8),
  `VAELoader` (flux2-vae), `LoadImage`.
- `CVRR_demo_workflow.json` — the **recommended LoRA-style** path: core
  `CLIPLoader` (type `flux2`) feeding the CVRR backbone/finetune →
  `CVRRApplyTransition` → same edit chain. Transition widget intentionally uses
  the aliased name `cvrr/cvrr_qwen3vl8b_adapter_fp32.safetensors` to
  demonstrate name-agnostic selection from `models/text_encoders`.

## Editing/validation discipline

- Widgets are **positional**: ComfyUI serializes required-then-optional widget
  values in declaration order; a schema change = silent mis-wiring until
  `tests/test_nodes.py::test_example_workflows_match_node_schemas` catches it
  (it iterates every `examples/*.json`; widget membership for combos is checked
  against the fixture's stub file list — when you rename a demo file, extend
  the stub list in `nodes_module`).
- `inputs` entries list `link` ids; `links` rows are
  `[id, src_node, src_slot, dst_node, dst_slot, type]`; keep `last_link_id` /
  `last_link_id` and `order` consistent (topological).
- Widget orders for our nodes (required…, then optional…):
  - `CVRRTextEncoderLoader`: clip_name, mode, merged_transition ｜ device,
    blocksize, inference_steps, beta, tap_gains
  - `CVRRApplyTransition`: merged_transition ｜ blocksize, inference_steps, beta
  - `CVRREditTextEncode`: prompt, mode, vl_megapixels, reference_megapixels ｜
    image2(link), negative_prompt, reference_latents_method, tap_gains
  - `CVRRTextEncode`: prompt, mode, vl_megapixels, tap_gains,
    block_visual_access ｜ image(link), image2(link)
  - `CVRRSetReferenceLatent`: megapixels ｜ image2(link),
    reference_latents_method
- Note: link-type inputs (`clip`, `vae`, `image`, `image2`, `conditioning`)
  are NOT in `widgets_values` — only scalar/combo/string inputs are.
