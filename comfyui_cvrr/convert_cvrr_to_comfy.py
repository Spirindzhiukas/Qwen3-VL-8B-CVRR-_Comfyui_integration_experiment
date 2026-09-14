#!/usr/bin/env python3
"""Convert a released CVRR checkpoint into files ComfyUI can load.

The released layout (``dmis-lab/Qwen3-VL-8B-CVRR``) is::

    config.json                      # release metadata (ell_star, T, beta, ...)
    cvrr_release_config.json         # same block, standalone
    merged_transition.safetensors    # 7 fp32 projections of the recurrent layer
    native_backbone/                 # untouched Qwen3-VL-8B-Instruct weights
        config.json
        model.safetensors.index.json
        native-0000{1..4}.safetensors
        tokenizer*.json, merges.txt, vocab.json, preprocessor_config.json
    modeling_cvrr_merged.py, configuration_cvrr_merged.py, ...

ComfyUI's Qwen3-VL text encoder expects *its own* key layout::

    model.layers.N.self_attn.q_proj.weight      (HF: model.language_model.layers.N...)
    model.norm.weight                           (HF: model.language_model.norm.weight)
    model.embed_tokens.weight                   (HF: model.language_model.embed_tokens.weight)
    model.lm_head.weight                        (HF: lm_head.weight)
    visual.<...>                                (HF: model.visual.<...>)

which is exactly the prefix rewrite ComfyUI itself applies to Qwen3-VL
checkpoints, so the conversion is lossless and hardware-free.

Usage
-----

    # text encoder only (about 16 GB bf16, or ~8.7 GB fp8)
    python -m comfyui_cvrr.convert_cvrr_to_comfy \
        --release-dir  ~/models/Qwen3-VL-8B-CVRR \
        --output-dir   ~/ComfyUI/models/text_encoders/cvrr \
        --dtype bf16

    # the 772 MB merged transition, re-written as fp16 (~386 MB) if you like
    python -m comfyui_cvrr.convert_cvrr_to_comfy \
        --release-dir  ~/models/Qwen3-VL-8B-CVRR \
        --output-dir   ~/ComfyUI/models/text_encoders/cvrr \
        --transition-only --dtype keep

Drop ``--dtype keep`` to keep the original bfloat16 weights untouched (fastest
and most faithful; ComfyUI runs text encoders in float32 compute anyway).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# --------------------------------------------------------------------------
# key mapping (no torch needed for the mapping itself)
# --------------------------------------------------------------------------

#: HF prefix -> ComfyUI prefix, identical to ``comfy.utils.state_dict_prefix_replace``
#: usage inside ``comfy/sd.py`` for the Qwen3-VL branches.
PREFIX_MAP: Tuple[Tuple[str, str], ...] = (
    ("model.language_model.", "model."),
    ("model.visual.", "visual."),
    ("lm_head.", "model.lm_head."),
)


def map_key(key: str, prefix: str = "", keep_vision: bool = True) -> Optional[str]:
    """Map a released-checkpoint key to ComfyUI's Qwen3-VL text-encoder layout.

    Returns ``None`` for keys that ComfyUI does not need (for example the
    top-level ``model.visual`` rotary buffers, which are recomputed).
    """
    mapped = key
    for source, target in PREFIX_MAP:
        if mapped.startswith(source):
            mapped = target + mapped[len(source):]
            break
    else:
        # already in ComfyUI layout, or an unrelated key
        if not mapped.startswith(("model.", "visual.")):
            return None
    if not keep_vision and mapped.startswith("visual."):
        return None
    if mapped.endswith(".inv_freq"):
        return None
    return prefix + mapped


def iter_shards(directory: str) -> Iterator[str]:
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        for name in sorted(set(weight_map.values())):
            yield os.path.join(directory, name)
        return
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors") and name != "merged_transition.safetensors":
            yield os.path.join(directory, name)


def convert(
    release_dir: str,
    output_dir: str,
    prefix: str = "",
    dtype: str = "keep",
    keep_vision: bool = True,
    max_shard_gb: float = 4.0,
    transition_only: bool = False,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Convert a release directory; returns a mapping of produced files."""
    import torch
    from safetensors.torch import load_file, save_file

    native_dir = os.path.join(release_dir, "native_backbone")
    if not os.path.isdir(native_dir):
        raise FileNotFoundError(f"no native_backbone/ inside {release_dir}")
    os.makedirs(output_dir, exist_ok=True)

    produced: Dict[str, str] = {}
    dtype_map = {
        "keep": None,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(dtype_map)}")
    target_dtype = dtype_map[dtype]

    if transition_only:
        return _convert_transition(release_dir, output_dir, target_dtype, dry_run, produced)

    # --- text encoder shards ------------------------------------------
    shard: Dict[str, torch.Tensor] = {}
    shard_bytes = 0
    limit = int(max_shard_gb * (1024 ** 3))
    shard_index = 0
    total_keys = 0
    skipped: List[str] = []
    index_map: Dict[str, str] = {}

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_index
        if not shard:
            return
        name = f"cvrr_qwen3vl_8b-{shard_index + 1:05d}.safetensors"
        path = os.path.join(output_dir, name)
        if not dry_run:
            save_file(shard, path, metadata={"format": "pt", "source": "cvrr-native-backbone"})
        for key in shard:
            index_map[key] = name
        produced[name] = path
        shard = {}
        shard_bytes = 0
        shard_index += 1

    for path in iter_shards(native_dir):
        for key, tensor in load_file(path).items():
            mapped = map_key(key, prefix=prefix, keep_vision=keep_vision)
            if mapped is None:
                skipped.append(key)
                continue
            if target_dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(target_dtype)
            shard[mapped] = tensor
            shard_bytes += tensor.numel() * tensor.element_size()
            total_keys += 1
            if shard_bytes >= limit:
                flush()
        del path
    flush()

    if not dry_run and index_map:
        index_path = os.path.join(output_dir, "cvrr_qwen3vl_8b.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"metadata": {"total_size": sum(
                    os.path.getsize(p) for p in produced.values())},
                 "weight_map": index_map},
                handle,
                indent=2,
            )
        produced["cvrr_qwen3vl_8b.safetensors.index.json"] = index_path

    _copy_release_metadata(release_dir, output_dir, dry_run, produced)
    report = {
        "checkpoint_keys": total_keys,
        "skipped_keys": len(skipped),
        "shards": shard_index,
        "dtype": dtype,
        "prefix": prefix,
        "dry_run": dry_run,
    }
    print(json.dumps(report, indent=2))

    # --- merged transition --------------------------------------------
    _convert_transition(release_dir, output_dir, target_dtype, dry_run, produced)
    return produced


def _convert_transition(release_dir, output_dir, target_dtype, dry_run, produced):
    import torch
    from safetensors.torch import load_file, save_file

    source = os.path.join(release_dir, "merged_transition.safetensors")
    if not os.path.isfile(source):
        raise FileNotFoundError(f"no merged_transition.safetensors in {release_dir}")
    weights = load_file(source)
    if target_dtype is not None:
        weights = {k: v.to(target_dtype) for k, v in weights.items()}
    path = os.path.join(output_dir, "cvrr_merged_transition.safetensors")
    if not dry_run:
        save_file(weights, path, metadata={"format": "pt", "dtype": str(next(iter(weights.values())).dtype)})
    produced["cvrr_merged_transition.safetensors"] = path
    print(f"merged transition: {len(weights)} tensors -> {path}")
    return produced


def _copy_release_metadata(release_dir, output_dir, dry_run, produced):
    for name in ("cvrr_release_config.json", "config.json"):
        source = os.path.join(release_dir, name)
        if os.path.isfile(source):
            destination = os.path.join(output_dir, "cvrr_release_config.json")
            if not dry_run:
                shutil.copy2(source, destination)
            produced["cvrr_release_config.json"] = destination
            break
    else:
        if not dry_run:
            metadata = {
                "name": "CVRR-Qwen3-VL-8B",
                "base_model": "Qwen/Qwen3-VL-8B-Instruct",
                "ell_star": 22,
                "recurrent_layer": 23,
                "upper_decoder_start": 24,
                "inference_T": 4,
                "inference_beta": 0.33,
                "note": "defaults written by convert_cvrr_to_comfy.py",
            }
            destination = os.path.join(output_dir, "cvrr_release_config.json")
            with open(destination, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            produced["cvrr_release_config.json"] = destination


def verify(output_dir: str, prefix: str = "", strict: bool = True,
           model_type: str = "qwen3vl_8b", with_vision: bool = True) -> int:
    """Check a converted file against ComfyUI's expected parameter names.

    Instantiates the matching ``Qwen3VL`` class on the ``meta`` device and compares
    its ``state_dict()`` keys with the converted ones.  ``model_type`` may be any
    key registered in ``comfy.text_encoders.qwen3vl.QWEN3VL_CONFIGS``.
    """
    from safetensors.torch import load_file

    files = sorted(f for f in os.listdir(output_dir) if f.endswith(".safetensors")
                   and "transition" not in f)
    if not files:
        print("no text encoder shards found")
        return 1

    try:
        import torch

        import comfy.text_encoders.qwen3vl  # noqa: F401  (registers the vision configs)
        from comfy.text_encoders.qwen3vl import QWEN3VL_CONFIGS, QWEN3VL_VISION, Qwen3VL
    except Exception as error:  # pragma: no cover - only for hand runs
        print(f"cannot import ComfyUI ({error}); skipping structural check")
        return 0

    model_class = type("VerifyQwen3VL", (Qwen3VL,), {"model_type": model_type})
    if model_type not in QWEN3VL_VISION:  # pragma: no cover - custom registrations
        print(f"unknown model_type {model_type!r}; skipping structural check")
        return 0
    with torch.device("meta"):
        model = model_class({}, torch.float32, torch.device("meta"), torch.nn)
    expected = set(model.state_dict().keys())
    if not with_vision:
        expected = {key for key in expected if not key.startswith("visual.")}
    del model

    seen = set()
    for name in files:
        seen.update(load_file(os.path.join(output_dir, name)).keys())
    stripped = {key[len(prefix):] if prefix and key.startswith(prefix) else key for key in seen}

    missing = sorted(expected - stripped)
    unexpected = sorted(stripped - expected)
    print(f"converted keys: {len(stripped)}   comfy params: {len(expected)}")
    if missing:
        print(f"MISSING ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"UNEXPECTED ({len(unexpected)}): {unexpected[:10]}")
    if not missing and not unexpected:
        print("OK: every converted tensor maps onto a ComfyUI Qwen3-VL parameter")
        return 0
    return 1 if strict else 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", required=True,
                        help="downloaded dmis-lab/Qwen3-VL-8B-CVRR directory")
    parser.add_argument("--output-dir", default=None,
                        help="target directory (usually ComfyUI/models/text_encoders/<name>)")
    parser.add_argument("--prefix", default="",
                        help="optional key prefix, e.g. 'text_encoders.' for single-file packs")
    parser.add_argument("--dtype", default="keep", choices=["keep", "fp32", "bf16", "fp16"])
    parser.add_argument("--drop-vision", action="store_true",
                        help="omit the vision tower (smaller file; CVRR then cannot run)")
    parser.add_argument("--transition-only", action="store_true")
    parser.add_argument("--max-shard-gb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="check the output against ComfyUI's parameter names")
    parser.add_argument("--model-type", default="qwen3vl_8b",
                        help="ComfyUI model_type used by --verify (default: qwen3vl_8b)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = args.output_dir or os.path.join(args.release_dir, "comfy_text_encoder")
    convert(
        args.release_dir,
        output_dir,
        prefix=args.prefix,
        dtype=args.dtype,
        keep_vision=not args.drop_vision,
        max_shard_gb=args.max_shard_gb,
        transition_only=args.transition_only,
        dry_run=args.dry_run,
    )
    if args.verify:
        return verify(output_dir, prefix=args.prefix, model_type=args.model_type)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
