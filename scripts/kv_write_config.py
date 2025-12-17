"""
Persist pipeline configuration to JSON for reproducibility.

This is a small helper so shell scripts don't need inline Python blocks.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--train-samples", type=int, required=True)
    p.add_argument("--eval-samples", type=int, required=True)
    p.add_argument("--samples-per-shard", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--eval-min-pairs", type=int, required=True)
    p.add_argument("--eval-max-pairs", type=int, required=True)
    p.add_argument("--max-dict-words", type=int, required=True)
    p.add_argument("--scenarios", required=True, help="Comma-separated scenario names.")
    p.add_argument("--train-iters", type=int, required=True)
    p.add_argument("--train-args", required=True)
    p.add_argument("--eval-args", required=True)
    p.add_argument("--tokenizer-corpus", required=True)
    p.add_argument("--tokenizer-max-chars", type=int, required=True)
    p.add_argument("--tokenizer-doc-cap", type=int, required=True)
    p.add_argument("--model-source", required=True)
    p.add_argument("--model-tag-prefix", required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--base-dir", required=True)
    args = p.parse_args()

    data = {
        "out_root": args.out_root,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "samples_per_shard": args.samples_per_shard,
        "seed": args.seed,
        "eval_min_pairs": args.eval_min_pairs,
        "eval_max_pairs": args.eval_max_pairs,
        "max_dict_words": args.max_dict_words,
        "scenarios": args.scenarios.split(",") if args.scenarios else [],
        "train_iters": args.train_iters,
        "train_args": args.train_args,
        "eval_args": args.eval_args,
        "tokenizer_corpus": args.tokenizer_corpus,
        "tokenizer_max_chars": args.tokenizer_max_chars,
        "tokenizer_doc_cap": args.tokenizer_doc_cap,
        "model_source": args.model_source,
        "model_tag_prefix": args.model_tag_prefix,
        "run_prefix": args.run_prefix,
        "base_dir": args.base_dir,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[*] config written to {args.out}")


if __name__ == "__main__":
    main()
