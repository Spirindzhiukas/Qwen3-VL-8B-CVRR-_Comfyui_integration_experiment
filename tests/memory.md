# memory.md — tests

## Layout & what each file proves

- `conftest.py` — everything shared: `COMFY_PATH` discovery (`COMFYUI_PATH`
  env, default `../ComfyUI_src`), `pytestmark` skip hook (no checkout → skip,
  never error), fixtures `comfy` (session: `comfy.options.args_parsing=False`,
  `args.cpu=True`, registers the tiny config), `spec` (tiny CVRRSpec), `tokenizer`;
  helpers `stock_te_class()`, `random_transition(model, spec, seed)`; constants
  `TINY_*`.
- `test_cvrr_core.py` (16) — pure algorithm: mode equalities, fp32 + idempotent
  install, layouts, masks, batch equivalence, config parsing, tap stacking.
  No ComfyUI import — runs even without a checkout.
- `test_comfy_integration.py` (6) — real ComfyUI classes at tiny dims: numerical
  parity of the unconfigured path with stock `Qwen3VL` (`_same(equal_nan=True)`),
  stacked Klein layout + emitted mask, image-causality, modes+ablation,
  text-only fallback. `_same` treats NaN==NaN; each forward gets its own
  `embeds.clone()`.
- `test_converter.py` (4) — synthetic release → convert → verify → reload.
- `test_loader.py` (4) — converted file loads via real `ClipTarget` +
  `comfy.sd.CLIP`; `apply_transition` flips to recurrent path; vision-less
  (text-only export) files are flagged at build (`cvrr_no_vision` + warning).
- `test_nodes.py` (14) — needs stub `folder_paths`/`node_helpers` (fixture
  `nodes_module`). Covers schemas/mappings, example-workflow widget validation
  (iterates **all** `examples/*.json`), the retrofit attach on a stock CLIP
  (encode parity, determinism, state restore, weighted-prompt escape hatch,
  refresh idempotence), rejection paths (no vision tower, shape mismatch,
  unloaded quantized weight), quantized bases (bf16, fp8-e4m3, int8 convrot),
  and the vision-less-encoder encode guard (meta-device visual params →
  actionable RuntimeError; text-only encodes unaffected).

## The tiny model laws (violating any → confusing crashes hours later)

1. tiny config = `@dataclass` subclass of `Qwen3VL_8BConfig` with
   `hidden_size=32, intermediate_size=64, num_hidden_layers=6,
   num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=151936`.
   The 8B parent is a dataclass: **without `@dataclass` you inherit its
   `__init__` and rebuild the full 8B → OOM (exit 137)**.
2. `rope_dims = [2, 1, 1]` assigned **after** the class body (real: `[24,20,20]`).
3. Vision: `QWEN3VL_VISION[TINY] = dict(hidden_size=64, num_heads=4,
   intermediate_size=128, depth=2, deepstack_visual_indexes=[0])`.
4. Tiny spec: `TINY_ELL_STAR=2` (recurrent layer 3, upper 4), `TINY_TAPS=(1,3,5)`,
   `T` 4, β 0.33.

## Fixture/monkeypatch conventions

- `nodes_module` fixture stubs `folder_paths` (with plausible `get_filename_list`
  contents — combo validation needs membership) and `node_helpers
  .conditioning_set_values`, then re-imports `comfyui_cvrr.nodes`.
- `tiny_spec_in_nodes` monkeypatches `nodes_module.RELEASE_SPEC = spec` (taps!)
  and `_write_transition` also writes `cvrr_release_config.json` next to the
  fake file with the tiny geometry. Patching `cvrr_te.RELEASE_SPEC` is NOT what
  `nodes.py` reads — patch the nodes namespace, and put the geometry json on
  disk for `load_release_spec`.
- `TRANSITION_ENTRY = "custom/adapter_v1.safetensors"` — deliberately alien
  filename proving name-agnostic loading. `_serve_file` monkeypatches
  `get_full_path` to resolve it to the tmp file.
- `_stock_tiny_clip` = CLIP built via `ClipTarget` + real `comfy.sd.CLIP`;
  `_mixed_precision_tiny_clip` = TE module built FIRST (CLIP.__init__ probes
  dtypes), weights initialised by geometry, then wrapped by hand
  (`CoreModelPatcher`, `EnumHookMode.MinVram`) — mirrors the real loader's
  construct-with-state_dict ordering. `_quantize_clip_weights` replaces weights
  with Parameter-wrapped `QuantizedTensor`s (fp8 layout or int8 convrot
  layout) + `quant_format` + `_full_precision_mm=True`.

## Known traps (hit at least once each)

- Calculating with stdout redirected got SIGKILLed (exit 137) with empty
  output in an earlier phase; run pytest plainly and only `tail` the result.
- `TransformerBlock` mutates the caller's `embeds` in place → integration
  tests clone per forward; the suite's production code clones too.
- Weighted-prompt token entries with images: image placeholders are
  `(dict, 1.0)` tuples and the vision expansion changes row shapes — do the
  weighted-prompt regression **text-only**.
- deepcopy of tokens rows is safe for text-only, not for image placeholders.
- `torch.equal` for run-to-run determinism; `torch.allclose(..., equal_nan=True)`
  only when NaN-free can't be guaranteed (weightless models emit NaNs!).
- `INPUT_TYPES` combos may be 1-element tuples `([...],)`; accept `(list, dict)`
  too.
- Full suite ≈ 44 tests in ~10–25 s, ~2 GB RAM peak.
