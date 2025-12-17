"""
Prepare train/eval splits for key-value retrieval experiments with controllable
distributions. This script builds multiple training sets (one per scenario)
and a single evaluation set that mixes the full distribution.

Example:
python -m kv_data.experiment --output-root kv_runs/exp1 --train-samples 50000 --eval-samples 5000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from kv_data.generate import generate_dataset


DEFAULT_SCENARIOS: Dict[str, Dict] = {
    "baseline_mixed": {
        "min_pairs": 4,
        "max_pairs": 200,
        "pair_count_mode": "linear",
        "q_position_mode": "uniform",
        "description": "Full distribution (baseline).",
    },
    "short_only": {
        "min_pairs": 4,
        "max_pairs": 32,
        "pair_count_mode": "linear",
        "q_position_mode": "uniform",
        "description": "Only short contexts; stress generalization to longer eval docs.",
    },
    "front_key_bias": {
        "min_pairs": 4,
        "max_pairs": 200,
        "pair_count_mode": "linear",
        "q_position_mode": "front",
        "description": "Query key tends to appear near the beginning.",
    },
    "end_key_bias": {
        "min_pairs": 4,
        "max_pairs": 200,
        "pair_count_mode": "linear",
        "q_position_mode": "back",
        "description": "Query key tends to appear near the end (closest to the question).",
    },
}


def _has_parquet(data_dir: Path) -> bool:
    return any(p.suffix == ".parquet" for p in data_dir.glob("*.parquet"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_experiments(
    output_root: Path,
    scenario_names: Iterable[str],
    train_samples: int,
    eval_samples: int,
    samples_per_shard: int,
    seed: int,
    max_dict_words: int,
    force: bool,
    eval_min_pairs: int,
    eval_max_pairs: int,
) -> Path:
    output_root = _ensure_dir(output_root)
    manifest = {"output_root": str(output_root), "train_configs": []}

    eval_dir = output_root / "eval"
    eval_cfg = {
        "total_samples": eval_samples,
        "samples_per_shard": samples_per_shard,
        "min_pairs": eval_min_pairs,
        "max_pairs": eval_max_pairs,
        "pair_count_mode": "linear",
        "q_position_mode": "uniform",
        "max_dict_words": max_dict_words,
        "seed": seed,
        "shard_prefix": "eval",
        "heatmap_max_pairs": eval_max_pairs,
    }
    if force or not _has_parquet(eval_dir):
        generate_dataset(out_root=str(eval_dir), **eval_cfg)
    else:
        print(f"[*] Reusing existing eval set at {eval_dir}")
    manifest["eval"] = {"data_dir": str(eval_dir), "config": eval_cfg}

    for idx, name in enumerate(scenario_names):
        if name not in DEFAULT_SCENARIOS:
            raise ValueError(f"Unknown scenario '{name}'. Options: {list(DEFAULT_SCENARIOS)}")
        cfg = DEFAULT_SCENARIOS[name].copy()
        cfg.update(
            {
                "total_samples": train_samples,
                "samples_per_shard": samples_per_shard,
                "max_dict_words": max_dict_words,
                "seed": seed + idx + 1,
                "shard_prefix": "train",
                "heatmap_max_pairs": eval_max_pairs,
            }
        )

        train_dir = output_root / f"{name}_train"
        if force or not _has_parquet(train_dir):
            generate_dataset(out_root=str(train_dir), **cfg)
        else:
            print(f"[*] Reusing existing train set for {name} at {train_dir}")

        manifest["train_configs"].append(
            {
                "name": name,
                "data_dir": str(train_dir),
                "config": cfg,
                "description": DEFAULT_SCENARIOS[name]["description"],
            }
        )

    manifest_path = output_root / "kv_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[*] Manifest written to {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Prepare KV retrieval experiment datasets.")
    parser.add_argument("--output-root", type=Path, default=Path("kv_runs"))
    parser.add_argument("--scenarios", type=str, default="baseline_mixed,short_only,front_key_bias,end_key_bias")
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--eval-samples", type=int, default=10_000)
    parser.add_argument("--samples-per-shard", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dict-words", type=int, default=170_000)
    parser.add_argument("--eval-min-pairs", type=int, default=4)
    parser.add_argument("--eval-max-pairs", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="Regenerate even if shards already exist.")
    args = parser.parse_args()

    scenario_names: List[str] = [s for s in args.scenarios.split(",") if s]
    build_experiments(
        output_root=args.output_root,
        scenario_names=scenario_names,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        samples_per_shard=args.samples_per_shard,
        seed=args.seed,
        max_dict_words=args.max_dict_words,
        force=args.force,
        eval_min_pairs=args.eval_min_pairs,
        eval_max_pairs=args.eval_max_pairs,
    )


if __name__ == "__main__":
    main()
