"""
Aggregate KV eval JSON files into a single summary.

Example:
python -m scripts.kv_collect --results-dir kv_runs/exp_shell1/results
"""

import argparse
import glob
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="Defaults to results-dir/eval_summary.json")
    args = p.parse_args()

    out_path = args.out or args.results_dir / "eval_summary.json"
    summary = {}
    for path in glob.glob(str(args.results_dir / "*_eval.json")):
        name = Path(path).name.replace("_eval.json", "")
        with open(path) as f:
            summary[name] = json.load(f)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] summary written to {out_path}")


if __name__ == "__main__":
    main()
