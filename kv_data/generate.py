import os
import math
import random
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

def load_english_words(
    max_words: int = None,
    filter_by_length: bool = False,
) -> List[str]:
    ds = load_dataset("Maximax67/English-Valid-Words", "sorted_by_frequency", split="train")
    words = [row["Word"] for row in ds if row["Word"]]
    
    if filter_by_length:
        words = [w for w in words if 2 <= len(w) <= 15]

    if max_words is not None:
        words = words[:max_words]

    return words


def make_sample(
    num_pairs: int,
    words: List[str],
) -> Tuple[str, int, int, str, str]:
    assert num_pairs > 0
    needed = num_pairs * 2
    assert needed <= len(words), "Not enough words"

    sampled = random.sample(words, needed)
    keys = sampled[:num_pairs]
    values = sampled[num_pairs:]

    items = list(zip(keys, values))
    random.shuffle(items)

    q_pair_idx = random.randrange(num_pairs)
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
    """rows 리스트(파이썬 dict 리스트)를 하나의 parquet shard 로 저장"""
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

    print(f"[OK] {out_dir}: shard_{shard_idx:05d} written ({len(rows)} samples)")


def _make_pair_counts(
    total_samples: int,
    min_pairs: int,
    max_pairs: int,
    mode: str = "linear",
):
    if mode == "uniform":
        return [random.randint(min_pairs, max_pairs) for _ in range(total_samples)]

    elif mode == "linear":
        population = list(range(min_pairs, max_pairs + 1))
        weights = [n for n in population]

        pair_counts = random.choices(
            population,
            weights=weights,
            k=total_samples,
        )
        return pair_counts

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _save_heatmap(heat_pairs, heat_qidx, out_root: str):
    if not heat_pairs:
        return

    max_pairs_seen = max(heat_pairs)
    max_qidx_seen = max(heat_qidx)

    x_bins = min(max_pairs_seen, 50)
    y_bins = min(max_qidx_seen + 1, 50) if max_qidx_seen > 0 else 1

    H, xedges, yedges = np.histogram2d(
        heat_pairs,
        heat_qidx,
        bins=[x_bins, y_bins],
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


WORDS = None

def _init_worker(words):
    """ProcessPoolExecutor initializer: 전역 WORDS 설정"""
    global WORDS
    WORDS = words


def _build_sample(args):
    """(idx, num_pairs)를 받아서 (idx, row) 반환"""
    idx, num_pairs = args
    text, n_pairs, q_idx, q_key, q_value = make_sample(num_pairs, WORDS)
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
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    samples_per_shard: int = 10_000,
    min_pairs: int = 4,
    max_pairs: int = 20_000,
    out_root: str = "kv_base_data_english_simple",
    max_dict_words: int = 170_000,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    print("[*] load english words..")
    words = load_english_words(max_words=max_dict_words)
    print(f"[*] {len(words)} words loaded")
    print("[*] Synthetic KV Retrieval Dataset Generation Started...")

    pair_counts = _make_pair_counts(total_samples, min_pairs, max_pairs)

    indices = list(range(total_samples))
    random.shuffle(indices)

    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val

    splits = [None] * total_samples  # "train", "val", "test"
    for rank, idx in enumerate(indices):
        if rank < n_train:
            splits[idx] = "train"
        elif rank < n_train + n_val:
            splits[idx] = "val"
        else:
            splits[idx] = "test"

    buffers = {
        "train": [],
        "val": [],
        "test": [],
    }
    shard_counters = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    def flush(split_name: str):
        rows = buffers[split_name]
        if not rows:
            return
        out_dir = os.path.join(out_root, split_name)
        shard_idx = shard_counters[split_name]
        write_shard(rows, out_dir, shard_idx=shard_idx)
        shard_counters[split_name] += 1
        buffers[split_name].clear()

    heat_pairs = []
    heat_qidx = []

    print("[*] generating samples in parallel and writing shards on the fly...")
    max_workers = os.cpu_count() or 4

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(words,),
    ) as executor:
        futures = [
            executor.submit(_build_sample, (i, pc))
            for i, pc in enumerate(pair_counts)
        ]

        for f in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_samples,
            desc="Generating samples (proc)",
        ):
            idx, row = f.result()
            split = splits[idx]

            buffers[split].append(row)

            heat_pairs.append(row["num_pairs"])
            heat_qidx.append(row["q_pair_idx"])

            if len(buffers[split]) >= samples_per_shard:
                flush(split)

    for split_name in ("train", "val", "test"):
        flush(split_name)

    _save_heatmap(heat_pairs, heat_qidx, out_root)

    print("[DONE] dataset generated")
    print(f" - train: ~{n_train} samples")
    print(f" - val:   ~{n_val} samples")
    print(f" - test:  ~{n_test} samples")


if __name__ == "__main__":
    random.seed(42)
    generate_dataset(
        total_samples=100_000,
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        samples_per_shard=10_000,
        min_pairs=4,
        max_pairs=10_000, 
        out_root="kv_base_data_english_simple",
        max_dict_words=170_000,
    )
