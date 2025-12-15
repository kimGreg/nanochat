import os
import random
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

    q_pair_idx = rng.randrange(num_pairs)
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


def _save_heatmap(heat_pairs, heat_qidx, out_root: str):
    if not heat_pairs:
        return

    max_pairs_seen = max(heat_pairs)
    max_qidx_seen = max(heat_qidx)

    # x_bins = min(max_pairs_seen, 100)
    # y_bins = min(max_qidx_seen + 1, 100) if max_qidx_seen > 0 else 1

    # H, xedges, yedges = np.histogram2d(
    #     heat_pairs,
    #     heat_qidx,
    #     bins=[x_bins, y_bins],
    # )
    
    x_bins = max_pairs_seen      # 정수 개수만큼
    y_bins = max_qidx_seen + 1                   # 0 ~ max_qidx

    H, xedges, yedges = np.histogram2d(
        heat_pairs,
        heat_qidx,
        bins=[x_bins, y_bins],
        range=[[0.5, max_pairs_seen + 0.5],
               [-0.5, max_qidx_seen + 0.5]],
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
    idx, num_pairs, base_seed = args
    if WORDS is None:
        raise RuntimeError("WORDS is not initialized in worker")
    rng = random.Random(base_seed + idx)
    text, n_pairs, q_idx, q_key, q_value = make_sample(num_pairs, WORDS, rng)
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
):
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
        total_samples, min_pairs, max_pairs, rng=rng, mode="linear"
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
        args_iter = ((i, pc, seed) for i, pc in enumerate(pair_counts))
        for idx, row in tqdm(
            executor.map(_build_sample, args_iter, chunksize=128),
            total=total_samples,
            desc="Generating samples (proc)",
        ):
            buffer.append(row)

            heat_pairs.append(row["num_pairs"])
            heat_qidx.append(row["q_pair_idx"])

            if len(buffer) >= samples_per_shard:
                write_shard(buffer, out_root, shard_idx=shard_idx)
                shard_idx += 1
                buffer.clear()

    if buffer:
        write_shard(buffer, out_root, shard_idx=shard_idx)

    _save_heatmap(heat_pairs, heat_qidx, out_root)

    print("[DONE] dataset generated")
    print(f" - total samples: {total_samples}")
    print(f" - shards: {shard_idx + (1 if buffer else 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KV dataset")
    parser.add_argument("-n", "--num-sample", type=int, default=100_000)
    parser.add_argument("--min-pairs", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_dataset(
        total_samples=args.num_sample,
        samples_per_shard=10_000,
        min_pairs=args.min_pairs,
        max_pairs=args.max_pairs,
        out_root=DATA_DIR,
        max_dict_words=170_000,
        seed=args.seed,
    )
