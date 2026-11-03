#!/usr/bin/env bash
set -euo pipefail

model_path=$1
embedding_path=$2
workers=${3:-1}
output=${4:-rerun}
cd "$(dirname "$0")"
mkdir -p "$output"
python fetch_images.py
pids=()
for ((worker=0; worker<workers; worker++)); do
    CUDA_VISIBLE_DEVICES=$worker python run_inference.py \
        --model-path "$model_path" --shard "$worker" --shards "$workers" \
        --output "$output/part-$worker.jsonl" > "$output/part-$worker.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
python analyze.py --embedding-path "$embedding_path" \
    --responses "$output"/part-*.jsonl --label internvl --output "$output/analysis"
