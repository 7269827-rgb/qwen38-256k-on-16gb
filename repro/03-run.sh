#!/usr/bin/env bash
# 03-run.sh - run the 200k K30 configuration
# (Windows: adapt paths; this is the canonical launch)
set -e
MODEL=./configs/pareto-bf16.gguf
export GGML_VK_FA_SELECT_KEEP=30
export GGML_VK_FA_SELECT_MINKV=32768
llama-server \
  --model "" \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.60 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 --port 8090 --ctx-size 262144 \
  -fa on -ctkd q4_0 -ctvd q4_0
