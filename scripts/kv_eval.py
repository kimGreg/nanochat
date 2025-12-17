"""
Evaluate key-value retrieval accuracy on a parquet dataset produced by kv_data.

Example:
python -m scripts.kv_eval --source base --data-dir kv_runs/exp1/eval --model-tag d8
"""

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.engine import Engine
from nanochat.report import get_report


def _iter_rows(data_dir: Path, max_samples: Optional[int]) -> Iterable[Dict]:
    count = 0
    for path in sorted(data_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        cols = ["text", "q_value", "num_pairs", "q_pair_idx"]
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=cols)
            rows = zip(
                rg.column(0).to_pylist(),
                rg.column(1).to_pylist(),
                rg.column(2).to_pylist(),
                rg.column(3).to_pylist(),
            )
            for text, q_value, num_pairs, q_idx in rows:
                yield {
                    "text": text,
                    "q_value": q_value,
                    "num_pairs": num_pairs,
                    "q_idx": q_idx,
                }
                count += 1
                if max_samples is not None and count >= max_samples:
                    return


def _bucket_length(num_pairs: int) -> str:
    if num_pairs <= 16:
        return "len<=16"
    if num_pairs <= 64:
        return "len<=64"
    if num_pairs <= 128:
        return "len<=128"
    return "len>128"


def _bucket_position(q_idx: int, num_pairs: int) -> str:
    if num_pairs <= 1:
        return "front"
    frac = q_idx / max(1, num_pairs - 1)
    if frac < 0.25:
        return "front"
    if frac > 0.75:
        return "back"
    return "middle"


def _extract_prompt(text: str):
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("Sample text is too short to contain question/answer.")
    q_line = lines[-2]
    prompt_lines = lines[:-1]
    prompt_lines[-1] = "A "
    prompt = "\n".join(prompt_lines)
    return prompt, q_line


def _compute_heatmap(records, bins: int, max_pairs_hint: Optional[int] = None):
    if not records:
        return {"matrix": [], "x_edges": [], "y_edges": []}

    pair_vals = np.array([r["num_pairs"] for r in records], dtype=np.float64)
    q_vals = np.array([r["q_idx"] for r in records], dtype=np.float64)
    correct = np.array([1.0 if r["correct"] else 0.0 for r in records], dtype=np.float64)

    max_pairs_seen = int(pair_vals.max())
    max_q_seen = int(q_vals.max())
    max_pairs = max(max_pairs_seen, max_pairs_hint or 0)
    max_q = max(max_q_seen, max_pairs - 1)

    x_edges = np.linspace(0.5, max_pairs + 0.5, bins + 1)
    y_edges = np.linspace(-0.5, max_q + 0.5, bins + 1)

    total_hist, _, _ = np.histogram2d(pair_vals, q_vals, bins=[x_edges, y_edges])
    correct_hist, _, _ = np.histogram2d(pair_vals, q_vals, bins=[x_edges, y_edges], weights=correct)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = correct_hist / total_hist

    return {
        "matrix": acc.tolist(),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
    }


def _print_heatmap(matrix, x_edges, y_edges):
    if not matrix:
        print0("[heatmap] no data")
        return
    mat = np.array(matrix)
    rows, cols = mat.shape
    print0(f"[heatmap] accuracy (num_pairs x q_idx), bins={cols}")
    # Show top-left corner (highest q_idx at top)
    for y in range(rows - 1, -1, -1):
        row_vals = []
        for x in range(cols):
            v = mat[x, y]
            if np.isnan(v):
                row_vals.append(" -- ")
            else:
                row_vals.append(f"{100*v:4.0f}")
        print0(" ".join(row_vals))
    print0(f"[heatmap] x_edges (num_pairs): {x_edges}")
    print0(f"[heatmap] y_edges (q_idx): {y_edges}")


def evaluate_kv(
    data_dir: Path,
    model,
    tokenizer,
    engine: Engine,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_samples: Optional[int],
    heatmap_bins: int,
    max_pairs_hint: Optional[int],
):
    stats = {
        "total": 0,
        "correct": 0,
        "by_length": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_position": defaultdict(lambda: {"correct": 0, "total": 0}),
    }
    bos = tokenizer.get_bos_token_id()
    records = []

    for row in _iter_rows(data_dir, max_samples):
        prompt_text, _ = _extract_prompt(row["text"])
        gold = row["q_value"].strip()

        prompt_tokens = tokenizer.encode(prompt_text, prepend=bos)

        with torch.no_grad():
            results, _ = engine.generate_batch(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        completion_tokens = results[0][len(prompt_tokens) :]
        completion_text = tokenizer.decode(completion_tokens)
        first_line = completion_text.splitlines()[0] if completion_text else ""
        predicted = first_line.strip().split(" ")[0] if first_line else ""

        is_correct = predicted.lower() == gold.lower()

        stats["total"] += 1
        stats["correct"] += int(is_correct)
        records.append(
            {
                "num_pairs": row["num_pairs"],
                "q_idx": row["q_idx"],
                "correct": is_correct,
            }
        )

        length_bucket = _bucket_length(row["num_pairs"])
        pos_bucket = _bucket_position(row["q_idx"], row["num_pairs"])
        for bucket_dict, bucket in (
            (stats["by_length"], length_bucket),
            (stats["by_position"], pos_bucket),
        ):
            bucket_dict[bucket]["total"] += 1
            bucket_dict[bucket]["correct"] += int(is_correct)

        if stats["total"] % 100 == 0:
            acc = 100 * stats["correct"] / max(1, stats["total"])
            print0(f"[progress] {stats['total']} samples | acc={acc:.2f}%")

    def _acc(partial):
        return {
            k: v["correct"] / v["total"] for k, v in partial.items() if v["total"] > 0
        }

    overall = stats["correct"] / max(1, stats["total"])
    heatmap = _compute_heatmap(records, bins=heatmap_bins, max_pairs_hint=max_pairs_hint)
    return {
        "accuracy": overall,
        "by_length": _acc(stats["by_length"]),
        "by_position": _acc(stats["by_position"]),
        "total_samples": stats["total"],
        "heatmap": heatmap,
    }


def _resolve_data_dir_and_hint(data_dir: Optional[str], manifest: Optional[Path]) -> Tuple[Path, Optional[int]]:
    if data_dir:
        return Path(data_dir), None
    if manifest:
        with open(manifest, "r") as f:
            man = json.load(f)
        if "eval" not in man or "data_dir" not in man["eval"]:
            raise ValueError("Manifest does not contain eval.data_dir")
        max_pairs_hint = None
        cfg = man.get("eval", {}).get("config", {})
        if isinstance(cfg, dict):
            max_pairs_hint = cfg.get("max_pairs") or cfg.get("heatmap_max_pairs")
        return Path(man["eval"]["data_dir"]), max_pairs_hint
    raise ValueError("Provide either --data-dir or --manifest")


def main():
    parser = argparse.ArgumentParser(description="Key-value retrieval evaluator.")
    parser.add_argument("--data-dir", type=str, default=None, help="Parquet directory to evaluate on.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest produced by kv_data.experiment.")
    parser.add_argument("--source", type=str, required=True, help="Model source: base|mid|sft|rl")
    parser.add_argument("--model-tag", type=str, default=None, help="Checkpoint tag (e.g., d8).")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--heatmap-bins", type=int, default=20, help="Number of bins per axis for accuracy heatmap.")
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--label", type=str, default="", help="Optional name for report logging.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save raw eval results as JSON.")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    )

    data_dir, max_pairs_hint = _resolve_data_dir_and_hint(args.data_dir, args.manifest)
    print0(f"[*] Evaluating KV retrieval on {data_dir}")

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    with autocast_ctx:
        results = evaluate_kv(
            data_dir=data_dir,
            model=model,
            tokenizer=tokenizer,
            engine=engine,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_samples=args.max_samples,
            heatmap_bins=args.heatmap_bins,
            max_pairs_hint=max_pairs_hint,
        )

    print0(f"[*] KV accuracy: {100 * results['accuracy']:.2f}% over {results['total_samples']} samples")
    print0(f"    by_length: {results['by_length']}")
    print0(f"    by_position: {results['by_position']}")
    _print_heatmap(results["heatmap"]["matrix"], results["heatmap"]["x_edges"], results["heatmap"]["y_edges"])

    if args.save_json is not None:
        payload = {
            "args": {
                "data_dir": str(data_dir),
                "source": args.source,
                "model_tag": args.model_tag,
                "step": args.step,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_samples": args.max_samples,
                "heatmap_bins": args.heatmap_bins,
                "label": args.label,
            },
            "results": results,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print0(f"[*] Saved eval JSON to {args.save_json}")

    get_report().log(
        section="KV Retrieval",
        data=[
            {
                "label": args.label or data_dir.name,
                "data_dir": str(data_dir),
                "source": args.source,
                "model_tag": args.model_tag,
                "step": args.step,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "max_samples": args.max_samples,
                "heatmap_bins": args.heatmap_bins,
            },
            results,
        ],
    )

    compute_cleanup()


if __name__ == "__main__":
    main()
