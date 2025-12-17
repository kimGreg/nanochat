#!/bin/bash
set -euo pipefail

# End-to-end KV retrieval experiment runner with baked-in defaults.
# Order: dataset generation -> tokenizer (trained on baseline_mixed_train) -> per-scenario training -> eval.
# Edit the variables below to change dataset sizes, scenarios, tokenizer corpus, or training args.

## Paths and sizes
OUT_ROOT="${OUT_ROOT:-kv_runs/exp_shell1}"
TRAIN_SAMPLES=${TRAIN_SAMPLES:-600000}
EVAL_SAMPLES=${EVAL_SAMPLES:-200000}
SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-10000}
SEED=${SEED:-42}
EVAL_MIN_PAIRS=${EVAL_MIN_PAIRS:-4}
EVAL_MAX_PAIRS=${EVAL_MAX_PAIRS:-200}
MAX_DICT_WORDS=${MAX_DICT_WORDS:-170000}

## Scenarios to run (comma-separated, must match kv_data.experiment defaults)
SCENARIOS="${SCENARIOS:-baseline_mixed,short_only,front_key_bias,end_key_bias}"


## Training config (adjust for your hardware)
TRAIN_ITERS=${TRAIN_ITERS:-100000} # -1 -> use base_train default; set e.g. 1000 for a quick test
TRAIN_ARGS="${TRAIN_ARGS:---depth=8 --max_seq_len=1024 --device_batch_size=1 --eval_tokens=1024 --core_metric_every=-1 --total_batch_size=1024}"
RUN_PREFIX="${RUN_PREFIX:-kv_1_}"
MODEL_TAG_PREFIX="${MODEL_TAG_PREFIX:-kv_}"

## Eval config
MODEL_SOURCE="${MODEL_SOURCE:-base}"
EVAL_ARGS="${EVAL_ARGS:---max-new-tokens 8 --temperature 0.0 --top-k 0 --max-samples 2000 --heatmap-bins 21}"

## Control stages / tokenizer corpus
SKIP_TOKENIZE="${SKIP_TOKENIZE:-0}" # set to 1 to skip tokenizer stage
TOKENIZER_CORPUS="${TOKENIZER_CORPUS:-baseline_mixed_train}" # which generated split to train tokenizer on
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-10000}"

# Activate virtualenv if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Cache / output dirs
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
RESULTS_DIR="$OUT_ROOT/results"
MODELS_DIR="$OUT_ROOT/models"
OUT_TOKENIZER_DIR="$OUT_ROOT/tokenizer"
mkdir -p "$RESULTS_DIR" "$MODELS_DIR"

echo "[*] Generating datasets into $OUT_ROOT"
python -m kv_data.experiment \
  --output-root "$OUT_ROOT" \
  --scenarios "$SCENARIOS" \
  --train-samples "$TRAIN_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --samples-per-shard "$SAMPLES_PER_SHARD" \
  --seed "$SEED" \
  --eval-min-pairs "$EVAL_MIN_PAIRS" \
  --eval-max-pairs "$EVAL_MAX_PAIRS" \
  --max-dict-words "$MAX_DICT_WORDS"

MANIFEST="$OUT_ROOT/kv_manifest.json"

# Tokenizer trained on selected corpus (defaults to baseline_mixed_train)
if [ "$SKIP_TOKENIZE" != "1" ]; then
  TOKENIZER_DATA_DIR="$OUT_ROOT/$TOKENIZER_CORPUS"
  if [ ! -d "$TOKENIZER_DATA_DIR" ]; then
    echo "[!] Tokenizer corpus dir not found: $TOKENIZER_DATA_DIR"
    exit 1
  fi
  echo "[*] Building RustBPE extension (idempotent)"
  uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
  echo "[*] Training tokenizer on $TOKENIZER_DATA_DIR"
  env NANOCHAT_DATA_DIR="$TOKENIZER_DATA_DIR" \
    python -m scripts.tok_train \
      --max_chars "$TOKENIZER_MAX_CHARS" \
      --doc_cap "$TOKENIZER_DOC_CAP"
  echo "[*] Copying tokenizer to $OUT_TOKENIZER_DIR"
  mkdir -p "$OUT_TOKENIZER_DIR"
  rsync -a "$BASE_DIR/tokenizer/" "$OUT_TOKENIZER_DIR/"
else
  echo "[*] Skipping tokenizer stage (SKIP_TOKENIZE=1)"
fi

IFS=',' read -ra scenario_list <<< "$SCENARIOS"
for scenario in "${scenario_list[@]}"; do
  data_dir="$OUT_ROOT/${scenario}_train"
  model_tag="${MODEL_TAG_PREFIX}${scenario}"
  run_name="${RUN_PREFIX}${scenario}"

  echo "[*] Training scenario '$scenario' (data: $data_dir) -> model_tag=$model_tag"
  NUM_ITERS_ARG=""
  if [ "$TRAIN_ITERS" != "-1" ]; then
    NUM_ITERS_ARG="--num_iterations=$TRAIN_ITERS"
  fi

  env NANOCHAT_DATA_DIR="$data_dir" \
    python -m scripts.base_train \
      --model_tag="$model_tag" \
      --run="$run_name" \
      $NUM_ITERS_ARG \
      $TRAIN_ARGS
  echo "[*] Copying checkpoint to $MODELS_DIR/$model_tag"
  rsync -a "$BASE_DIR/base_checkpoints/$model_tag/" "$MODELS_DIR/$model_tag/"

  echo "[*] Evaluating scenario '$scenario' on mixed eval set"
  eval_json="$RESULTS_DIR/${scenario}_eval.json"
  python -m scripts.kv_eval \
    --manifest "$MANIFEST" \
    --source "$MODEL_SOURCE" \
    --model-tag "$model_tag" \
    --label "$scenario" \
    --save-json "$eval_json" \
    $EVAL_ARGS
done

echo "[*] Writing pipeline config to $RESULTS_DIR/pipeline_config.json"
python -m scripts.kv_write_config \
  --out "$RESULTS_DIR/pipeline_config.json" \
  --out-root "$OUT_ROOT" \
  --train-samples "$TRAIN_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --samples-per-shard "$SAMPLES_PER_SHARD" \
  --seed "$SEED" \
  --eval-min-pairs "$EVAL_MIN_PAIRS" \
  --eval-max-pairs "$EVAL_MAX_PAIRS" \
  --max-dict-words "$MAX_DICT_WORDS" \
  --scenarios "$SCENARIOS" \
  --train-iters "$TRAIN_ITERS" \
  --train-args "$TRAIN_ARGS" \
  --eval-args "$EVAL_ARGS" \
  --tokenizer-corpus "$TOKENIZER_CORPUS" \
  --tokenizer-max-chars "$TOKENIZER_MAX_CHARS" \
  --tokenizer-doc-cap "$TOKENIZER_DOC_CAP" \
  --model-source "$MODEL_SOURCE" \
  --model-tag-prefix "$MODEL_TAG_PREFIX" \
  --run-prefix "$RUN_PREFIX" \
  --base-dir "$BASE_DIR"

echo "[*] Writing eval summary to $RESULTS_DIR/eval_summary.json"
python -m scripts.kv_collect --results-dir "$RESULTS_DIR"

echo "[DONE] KV pipeline completed for scenarios: $SCENARIOS"
