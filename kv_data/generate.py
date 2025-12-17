import os
import random
import json
from typing import List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import argparse

from nanochat.common import get_base_dir


base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

WORDS = None  # set in worker initializer


def load_english_words(
    max_words: Optional[int] = None,
    filter_by_length: bool = True,
) -> List[str]:
    ds = load_dataset(
        "Maximax67/English-Valid-Words",
        "sorted_by_frequency",
        split="train",
    )
    words = [row["Word"] for row in ds if row["Word"]]
    if filter_by_length:
        words = [w for w in words if 2 <= len(w) <= 15]
    if max_words is not None:
        words = words[:max_words]
    if not words:
        raise ValueError("No words loaded")
    return words


def make_sample(
    num_pairs: int,
    words: List[str],
    rng: random.Random,
    q_position_mode: str = "uniform",
) -> Tuple[str, int, int, str, str]:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")

    needed = num_pairs * 2
    if needed > len(words):
        raise ValueError(
            f"Not enough words: need {needed}, have {len(words)} "
            "(increase max_dict_words or reduce max_pairs)"
        )

    sampled = rng.sample(words, needed)
    keys = sampled[:num_pairs]
    values = sampled[num_pairs:]

    items = list(zip(keys, values))
    rng.shuffle(items)

    q_pair_idx = choose_q_index(num_pairs, rng, q_position_mode)
    q_key, q_value = items[q_pair_idx]

    kv_lines = [f"{k} {v}" for k, v in items]
    text = "\n".join(kv_lines) + f"\nQ {q_key}\nA {q_value}"

    return text, num_pairs, q_pair_idx, q_key, q_value


def write_shard(
    rows,
    out_dir: str,
    shard_idx: int,
    prefix: str = "shard",
):
    if not rows:
        return

    os.makedirs(out_dir, exist_ok=True)

    table = pa.Table.from_pydict(
        {
            "text": [r["text"] for r in rows],
            "num_pairs": [r["num_pairs"] for r in rows],
            "q_pair_idx": [r["q_pair_idx"] for r in rows],
            "q_key": [r["q_key"] for r in rows],
            "q_value": [r["q_value"] for r in rows],
        }
    )

    fname = os.path.join(out_dir, f"{prefix}_{shard_idx:05d}.parquet")
    pq.write_table(table, fname)
    print(f"[OK] {out_dir}: {os.path.basename(fname)} written ({len(rows)} samples)")


def _make_pair_counts(
    total_samples: int,
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
    mode: str = "linear",
):
    if min_pairs <= 0 or max_pairs < min_pairs:
        raise ValueError("Invalid min_pairs / max_pairs")

    if mode == "uniform":
        return [rng.randint(min_pairs, max_pairs) for _ in range(total_samples)]

    if mode == "linear":
        population = list(range(min_pairs, max_pairs + 1))
        weights = [n for n in population]
        return rng.choices(population, weights=weights, k=total_samples)

    raise ValueError(f"Unknown mode: {mode}")


def choose_q_index(num_pairs: int, rng: random.Random, mode: str = "uniform") -> int:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if mode == "uniform":
        return rng.randrange(num_pairs)
    if mode in ("front", "beginning"):
        window = max(1, num_pairs // 4)
        return rng.randrange(window)
    if mode in ("back", "end", "near_query"):
        window = max(1, num_pairs // 4)
        return num_pairs - window + rng.randrange(window)
    if mode == "middle":
        center = (num_pairs - 1) / 2
        std = max(1.0, num_pairs * 0.1)
        idx = int(round(rng.gauss(center, std)))
        return max(0, min(num_pairs - 1, idx))
    raise ValueError(f"Unknown q_position_mode: {mode}")


def _save_heatmap(heat_pairs, heat_qidx, out_root: str, max_pairs_hint: int = None, max_qidx_hint: int = None):
    if not heat_pairs:
        return

    max_pairs_seen = max(heat_pairs)
    max_qidx_seen = max(heat_qidx)

    max_pairs_plot = max(max_pairs_seen, max_pairs_hint) if max_pairs_hint else max_pairs_seen
    max_qidx_plot = max(max_qidx_seen, max_qidx_hint) if max_qidx_hint else max_qidx_seen

    x_bins = max_pairs_plot
    y_bins = max_qidx_plot + 1

    H, xedges, yedges = np.histogram2d(
        heat_pairs,
        heat_qidx,
        bins=[x_bins, y_bins],
        range=[[0.5, max_pairs_plot + 0.5], [-0.5, max_qidx_plot + 0.5]],
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.colorbar(label="count")
    plt.xlabel("num_pairs")
    plt.ylabel("q_idx")
    plt.title("num_pairs vs q_idx")
    plt.tight_layout()

    os.makedirs(out_root, exist_ok=True)
    heatmap_path = os.path.join(out_root, "dataset_distribution_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    print(f"[*] heatmap saved to {heatmap_path}")


def _init_worker(words):
    global WORDS
    WORDS = words


def _build_sample(args):
    idx, num_pairs, base_seed, q_mode = args
    if WORDS is None:
        raise RuntimeError("WORDS is not initialized in worker")
    rng = random.Random(base_seed + idx)
    text, n_pairs, q_idx, q_key, q_value = make_sample(
        num_pairs, WORDS, rng, q_position_mode=q_mode
    )
    row = {
        "text": text,
        "num_pairs": n_pairs,
        "q_pair_idx": q_idx,
        "q_key": q_key,
        "q_value": q_value,
    }
    return idx, row


def generate_dataset(
    total_samples: int = 800_000,
    samples_per_shard: int = 10_000,
    min_pairs: int = 4,
    max_pairs: int = 2_000,
    out_root: str = "kv_base_data_english_simple",
    max_dict_words: int = 170_000,
    seed: int = 42,
    pair_count_mode: str = "linear",
    q_position_mode: str = "uniform",
    shard_prefix: str = "shard",
    skip_if_exists: bool = False,
    write_manifest: bool = True,
    heatmap_max_pairs: Optional[int] = None,
    **kwargs,
):
    os.makedirs(out_root, exist_ok=True)
    existing = [
        f for f in os.listdir(out_root)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    ]
    if skip_if_exists and existing:
        print(f"[*] Skipping generation for {out_root} (found {len(existing)} parquet files)")
        return

    rng = random.Random(seed)

    print("[*] load english words..")
    words = load_english_words(max_words=max_dict_words, filter_by_length=True)
    print(f"[*] {len(words)} words loaded")

    if max_pairs * 2 > len(words):
        raise ValueError(
            f"max_pairs * 2 ({max_pairs * 2}) > available words ({len(words)}); "
            "increase max_dict_words or reduce max_pairs"
        )

    print("[*] Synthetic KV Retrieval Dataset Generation Started...")

    pair_counts = _make_pair_counts(
        total_samples, min_pairs, max_pairs, rng=rng, mode=pair_count_mode
    )

    buffer = []
    shard_idx = 0

    heat_pairs = []
    heat_qidx = []

    print("[*] generating samples in parallel and writing shards on the fly...")
    max_workers = os.cpu_count() or 4

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(words,),
    ) as executor:
        args_iter = ((i, pc, seed, q_position_mode) for i, pc in enumerate(pair_counts))
        for idx, row in tqdm(
            executor.map(_build_sample, args_iter, chunksize=128),
            total=total_samples,
            desc="Generating samples (proc)",
        ):
            buffer.append(row)

            heat_pairs.append(row["num_pairs"])
            heat_qidx.append(row["q_pair_idx"])

            if len(buffer) >= samples_per_shard:
                write_shard(buffer, out_root, shard_idx=shard_idx, prefix=shard_prefix)
                shard_idx += 1
                buffer.clear()

    if buffer:
        write_shard(buffer, out_root, shard_idx=shard_idx, prefix=shard_prefix)

    max_pairs_hint = heatmap_max_pairs if heatmap_max_pairs is not None else max_pairs
    max_qidx_hint = max_pairs_hint - 1 if max_pairs_hint else None
    _save_heatmap(heat_pairs, heat_qidx, out_root, max_pairs_hint=max_pairs_hint, max_qidx_hint=max_qidx_hint)

    print("[DONE] dataset generated")
    print(f" - total samples: {total_samples}")
    print(f" - shards: {shard_idx + (1 if buffer else 0)}")

    if write_manifest:
        manifest = {
            "total_samples": total_samples,
            "samples_per_shard": samples_per_shard,
            "min_pairs": min_pairs,
            "max_pairs": max_pairs,
            "pair_count_mode": pair_count_mode,
            "q_position_mode": q_position_mode,
            "max_dict_words": max_dict_words,
            "seed": seed,
            "shard_prefix": shard_prefix,
            "data_dir": os.path.abspath(out_root),
            "heatmap_max_pairs": heatmap_max_pairs,
        }
        with open(os.path.join(out_root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[*] manifest written to {os.path.join(out_root, 'manifest.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KV dataset")
    parser.add_argument("-n", "--num-sample", type=int, default=100_000)
    parser.add_argument("--min-pairs", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-mode", type=str, default="linear", choices=["linear", "uniform"])
    parser.add_argument(
        "--q-position-mode",
        type=str,
        default="uniform",
        choices=["uniform", "front", "beginning", "middle", "back", "end", "near_query"],
    )
    parser.add_argument("--out-root", type=str, default=DATA_DIR)
    parser.add_argument("--samples-per-shard", type=int, default=10_000)
    parser.add_argument("--heatmap-max-pairs", type=int, default=None)

    args = parser.parse_args()

    generate_dataset(
        total_samples=args.num_sample,
        samples_per_shard=args.samples_per_shard,
        min_pairs=args.min_pairs,
        max_pairs=args.max_pairs,
        out_root=args.out_root,
        max_dict_words=170_000,
        seed=args.seed,
        pair_count_mode=args.pair_mode,
        q_position_mode=args.q_position_mode,
        heatmap_max_pairs=args.heatmap_max_pairs,
    )
