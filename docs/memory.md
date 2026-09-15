# memory.md — docs

## Files & promises

- `FEASIBILITY.md` — the assessment. Sections: 1 verdict, 2 what CVRR is, 3
  release contents, 4 option A (upstream loader patch), 5 option B (node pack),
  **5a option C (LoRA-style attach, implemented later)**, 6 the Klein-9B
  quality question (the honest open part), 7 what's verified, 8 remaining
  work, 9 repo layout.
- root `README.md` — install/convert/node table/usage/tests/limitations.

## Honesty contract (do not weaken)

- Verified = CPU tiny-config only. Never write "works" where we mean
  "plumbing verified". GPU + real weights: unverified; §7/§8 state this.
- Klein quality improvement is *mechanically plausible, empirically unproven*
  (distribution shift; Klein trained on plain Qwen3-8B states) — §6 keeps that
  framing on purpose.
- The 772 MB adapter was trained on the instruct backbone; finetune distance =
  quality unknown (amplified §6 caveat).

## Numbers that must stay in sync when tests change

- Test count appears in: `README.md` header bullet *and* the pytest block
  comment, `docs/FEASIBILITY.md` §7 heading + §9 layout line. Current: **48**
  (core 16, integration 6, converter 4, loader 4, nodes 18). Use
  `pytest --collect-only -q | tail -6` to recount.

## Option summary (for quick recall)

- **A**: upstream — new `CLIPType` (~36) + class + loader-list entry; still
  needs the recurrence somewhere. Blocked on nothing but willingness/PR.
- **B**: this repo's node pack with a converted checkpoint (dedicated loader).
- **C** (recommended day-to-day): keep **any** Qwen3-VL-8B encoder file
  (stock/finetune, bf16/fp8/int8-convrot), attach the adapter via the
  `CVRR Apply Transition` node; adapter picked from `text_encoders` by combo,
  name-agnostic, validated by contents.
