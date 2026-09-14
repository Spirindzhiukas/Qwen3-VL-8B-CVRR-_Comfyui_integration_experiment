# memory.md — comfyui_cvrr (the node pack)

## File roles

- `cvrr_core.py` — algorithm only, **no ComfyUI imports**: `CVRRSpec` (release
  defaults ell*=22, T=4, β=0.33, taps (9,18,27)), `CVRRTextEncoderCore.encode`
  (modes `aligned` default / `strict` / `vl`), `MergedProjection` /
  `install_merged_transition` / `merged_transition_keys`, `klein_stack_taps`,
  masks. Unit tests import this with zero ComfyUI on the path — keep it clean.
- `cvrr_te.py` — ComfyUI host classes: `CVRRTextModel(qwen3vl.Qwen3VL)`,
  `CVRRQwen3VLClipModel(sd1_clip.SDClipModel)`, `CVRRTE(flux.Flux2TEModel)`,
  `build_clip()` (real `ClipTarget` + `comfy.sd.CLIP`), `apply_transition()`,
  `ensure_cvrr_module()` (retires stock stacks), `is_cvrr_clip`,
  `load_release_spec`, `CVRROptions`/`RELEASE_SPEC`, `_make_te_factory`.
- `nodes.py` — five nodes (display names in `NODE_DISPLAY_NAME_MAPPINGS` are
  the source of truth; README table mirrors them).
- `convert_cvrr_to_comfy.py` — HF release → ComfyUI files with `--verify`.
- `__init__.py` — mappings exposed only when `folder_paths` importable (so
  pytest works without ComfyUI). **Never import `nodes` unconditionally here.**

## Hard invariants (breaking one silently corrupts behavior)

1. **Runtime gating, never weight patching.** The adapter-off pass MUST see
   native weights. Merged weights live as fp32 **non-persistent buffers** on
   `MergedProjection` and are used only when `.enabled` — which the core
   toggles per pass. Do not "simplify" into a persistent merge or a ModelPatcher
   `add_patches` diff (those apply statically).
2. **fp32 merged weights.** `_apply` override keeps them fp32 across device
   casts — the recurrence quality depends on it (mirrors the release's
   `NativeMergedLinear`).
3. **Clone-before-encode of embeds.** ComfyUI's `TransformerBlock` writes its
   output into the caller's tensor *in place*; CVRR runs several passes over
   the buffer, so `cvrr_te` clones per forward (`mm_embeds=embeds...clone()`).
   This was the nastiest bug (1.6e-3 "numerical mismatch" that was actually
   cross-pass contamination).
4. **Configure-then-restore in nodes.** `_encode_with_options` saves the
   previous `cvrr_options`, configures for the call, restores in `finally`.
   This is what makes the shared-module design safe.
5. **Attach is atomic.** `install_merged_transition` validates all 7
   projections before rewiring any module (refresh path excepted).
6. **Name-agnostic adapter files.** The transition file is selected by combo
   from `text_encoders` exactly like CLIPLoader's list; validity is decided by
   *contents* (`merged_transition_keys` requires all 7 `<path>.weight`
   entries). Never require the filename `merged_transition.safetensors`.
7. **Quant/dtype contract.** Base projections may be `nn.Linear` (any cast),
   `fp8_ops`/`manual_cast` Linears, or `MixedPrecisionOp` modules with
   Parameter-wrapped `QuantizedTensor` weights (fp8/int8/convrot/NVFP4/W4A8).
   Contract = logical-shape `.weight` + dequantizing forward. `MergedProjection`
   shape check reads `.shape` (logical); a `quant_format`-tagged module with
   `.weight is None` = "not finished loading" → clear error.

## Retrofit path (option C)

`ensure_cvrr_module(clip)`: validates Flux2TEModel+Qwen3VL stack
(`.visual` present — Klein's stock `qwen_3_8b` TE is text-only and REFUSED),
`spec.validate(num_layers)`, then `__class__` swap to the CVRR subclasses +
attr defaults (`cvrr_options/core/transition/last_cvrr_result`), `model.layer`
re-set to spec taps. The apply node does `clip.clone()` first — the encoder
module is SHARED across handles, so attaching another transition file replaces
it for all of them (documented caveat; attaching is inert for stock encodes
because gating decides).

## Lessons from testing (things that failed before being understood)

- Declaring the tiny config subclass **without** `@dataclass` inherited the
  parent's generated `__init__` → rebuilt at 4096 width → exit 137 OOM.
- `rope_dims` must be assigned **after** the class body; declared inside a
  dataclass body it becomes a `None`-valued field and silently disables
  interleaved M-RoPE (`[12,-1,8]` reshape crash).
- `model_type` is read as a **class** attribute in `Qwen3VL.__init__` — bind on
  the class (`type(...)`), never the instance.
- `vocab_size=151936` is required for the real tokenizer despite the tiny dims.
- `comfy.utils.load_torch_file(path, safe_load=True)` returns the state dict
  only; `(sd, meta)` needs `return_metadata=True`.
- `comfy.sd.CLIP.__init__` probes `module.dtype` immediately → a
  mixed-precision stack *cannot* be constructed and then filled (weightless
  quant Linears have no dtype) — real loader passes state_dict at construction
  time; tests build the TE first and wrap manually (see tests/memory.md).
- `TensorWiseINT8Layout.quantize` with `convrot=True` requires
  `per_channel=True` and a power-of-4 Hadamard groupsize ≤ input width.
- Weighted-prompt rejection in `CVRRTE` must apply **only while a CVRR encode
  is configured** — retrofitted stacks are shared with stock consumers.
- torchvision must be 0.24.1 (0.29 wants torch 2.10 and breaks import).
- Importing `comfy.model_management` without `args.cpu=True` crashes on
  `torch.cuda.current_device()`; and set `comfy.options.args_parsing = False`
  before importing `comfy.cli_args` or it eats pytest's argv.
