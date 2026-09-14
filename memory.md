# memory.md — session level (read first)

## What this project is

Feasibility study + working implementation of `dmis-lab/Qwen3-VL-8B-CVRR` as a
ComfyUI text encoder for **FLUX.2 Klein 9B** image editing. Deliverables:
`docs/FEASIBILITY.md` (assessment), `comfyui_cvrr/` (node pack), `tests/` (42
CPU tests, all green), `examples/` (two workflows). Pushed to branch
`arena/01a0a0bc-qwen3-vl-8b-cvrr-comfyui-integ`; PR #1 on
`Spirindzhiukas/Qwen3-VL-8B-CVRR-_Comfyui_integration_experiment`.

Commits so far: `5b6b3e4` initial → `a1eec9a` main pack → `9153bea` COMFYUI_PATH
docs → `d0b1175` LoRA-style attach → `cc8a38d` quantized-base support →
(this) memory files + name-agnostic adapter loading + demo workflow.

## Environment (sandbox)

- Repo: `/home/user/Qwen3-VL-8B-CVRR-_Comfyui_integration_experiment`
- **venv**: `/home/user/.venv` (torch 2.9.1, torchvision 0.24.1, torchaudio
  2.9.1, transformers 4.57.6, comfy-kitchen 0.2.33, comfy-aimdo 0.5.3, pytest,
  safetensors, pillow, numpy, einops, psutil, scipy). System pip is
  PEP-668-blocked — always `/home/user/.venv/bin/pip`.
- **ComfyUI reference checkout**: `/home/user/ComfyUI_src`, **pinned at commit
  `b00584967778871f329c2d31190a5e9b767b8366`** (re-pin after any fresh clone:
  `git fetch --depth 1 origin b0058496... && git checkout FETCH_HEAD` — GitHub
  allows fetch-by-sha). Main moved to `db70adb+`; APIs my code subclasses
  (qwen3vl.py, flux.py, ops.py quant stack) could drift.
- No GPU, 2 CPUs, ~3 GB RAM, ~20 GB disk: **no real 8B inference possible**;
  all validation is CPU tiny-config against real ComfyUI modules.
- Network: `huggingface.co` unreachable via curl/git (works via the
  fetch-page tool on raw/tree URLs). PyPI + GitHub fine.

## Running the tests (the rules that matter)

```bash
cd /home/user/Qwen3-VL-8B-CVRR-_Comfyui_integration_experiment
COMFYUI_PATH=$HOME/ComfyUI_src /home/user/.venv/bin/python -m pytest -q
```

- **NEVER redirect pytest stdout/stderr to a file** (`> /tmp/x.txt 2>&1`
  is fine; what killed runs before was piping inside the harness differently —
  see tests/memory.md). If a run dies with exit 137 and empty output, suspect
  OOM, not the test.
- Expected: `42 passed`. Anything less, read `tests/memory.md` first.

## Sandbox/git hazards actually hit

- Mid-session reset wiped `/home/user/.venv` and `/home/user/ComfyUI_src` but
  kept the repo; local git was reset to `5b6b3e4` while the remote branch had
  committed work. Recovery used: `git fetch origin <branch>` (the
  `origin/<branch>` ref did not materialize — use `FETCH_HEAD`), then
  `git reset --soft FETCH_HEAD; git commit -C <local-head>; git push`.
  After any reset: re-create venv, re-clone+re-pin ComfyUI, re-run tests
  BEFORE believing any diff.
- Persistent storage = the git repo only. Keep nothing important elsewhere.

## Session roadmap status

- [x] feasibility (options A/B/C in `docs/FEASIBILITY.md`)
- [x] node pack + converter + tests (38)
- [x] LoRA-style attach to any Qwen3-VL ClIP (`d0b1175`)
- [x] bf16/fp8/int8(+convrot) quantized bases (`cc8a38d`, 42 tests)
- [x] name-agnostic transition loading via text_encoders combo
- [x] demo workflow `examples/CVRR_demo_workflow.json`
- [ ] GPU validation with real weights (logit parity vs
      `modeling_cvrr_merged.py`, then quality A/B) — blocked on hardware
