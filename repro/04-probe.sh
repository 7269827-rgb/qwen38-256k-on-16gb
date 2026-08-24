#!/usr/bin/env bash
# 04-probe.sh - reproduce the gate cells + the 200k K30 ladder
# Requires a running server from 03-run.sh (or your own launch) on :8090.
set -e
BASE=http://127.0.0.1:8090/v1

echo "== Gate screen (working depth): 40k resident, 256k alloc =="
# Fill the cache to 40k resident, then measure generation on short prompts.
# Reference result: 4th-best-of-5 = 56.6 t/s (results/pareto-screen-c1..c5.json).

echo "== 200k K30 ladder: fill to ~200k resident, then probe =="
# Stock anchor ~33 t/s; K30 (GGML_VK_FA_SELECT_KEEP=30) ~46 t/s.
# Quality: depth-GSM8K 29/30 at 199.7k (results/bench-depth-gsm8k.json);
# multi-hop 3/3 (results/bench-r6l2-mh-K30.json).
echo "See results/README.md for the exact per-file protocol."

echo "== llama-bench raw curve (MTP off, community protocol) =="
for d in 4096 16384 32768 65536 131072 200000; do
  llama-bench -m pareto-bf16.gguf -p 512 -n 128 -d $d -b 512 -ub 256 \
    -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -r 3 -o json
done
