# Key-value retrieval pipeline

This folder contains tooling to build controlled key-value retrieval datasets and evaluate checkpoints.

## Generate datasets

Use `kv_data.experiment` to create multiple training splits (one per scenario) and a single mixed evaluation split:

```bash
python -m kv_data.experiment \
  --output-root kv_runs/exp1 \
  --train-samples 50000 \
  --eval-samples 10000
```

This writes:

- `kv_runs/exp1/eval`: mixed distribution eval set (heatmap + manifest).
- `kv_runs/exp1/*_train`: one directory per scenario (baseline_mixed, short_only, front_key_bias, end_key_bias). Each directory's last shard is treated as val by `scripts.base_train`.

If you rerun with `--force` the shards are regenerated; otherwise existing shards are reused.

## Train on a scenario

Point training to a scenario directory by exporting `NANOCHAT_DATA_DIR` so the dataloader reads those shards:

```bash
export NANOCHAT_DATA_DIR=$PWD/kv_runs/exp1/front_key_bias_train
python -m scripts.base_train --depth=8 --max_seq_len=1024 --device_batch_size=1 --eval_tokens=1024 --core_metric_every=-1 --total_batch_size=1024 --num_iterations=-1 --run=kv-front
```

## Evaluate retrieval

Run the dedicated evaluator on the mixed eval split:

```bash
python -m scripts.kv_eval --source base --data-dir kv_runs/exp1/eval --model-tag d8 --max-samples 2000
```

You can also pass `--manifest kv_runs/exp1/kv_manifest.json` instead of `--data-dir`. Results (overall, by length bucket, by key position bucket) are printed and logged to `nanochat.report`.

## One-shot pipeline

To generate all splits, train once per scenario, and evaluate on the mixed set:

```bash
python -m scripts.kv_pipeline \
  --output-root kv_runs/exp1 \
  --train-samples 50000 \
  --eval-samples 10000 \
  --train-iters 1000 \
  --train-args "--depth=8 --max_seq_len=1024 --device_batch_size=1 --eval_tokens=1024 --core_metric_every=-1 --total_batch_size=1024"
```

Flags:
- `--skip-train` / `--skip-eval` to run only part of the pipeline.
- `--train-args` / `--eval-args` to pass through additional args to `scripts.base_train` and `scripts.kv_eval`.
- `--force-data` to regenerate data even if shards already exist.

If you want a zero-argument shell runner that also calls `tokenize.sh` first, use `bash kv_pipeline.sh` and edit the defaults inside that file.
The shell runner now:
- generates the scenario datasets,
- trains the tokenizer on `baseline_mixed_train` (by default; change `TOKENIZER_CORPUS` inside the script),
- trains each scenario,
- evaluates on the mixed eval split.

Heatmaps: generation uses a fixed axis range based on the eval max-pairs (so train-set heatmaps are comparable to eval distribution coverage).
