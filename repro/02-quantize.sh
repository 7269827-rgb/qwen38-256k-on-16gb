#!/usr/bin/env bash
# 02-quantize.sh - build the pareto blob from the BF16 base
# Produces pareto-bf16.gguf (SHA256 57DDA505ECD2B0731341C002FF03EDE1526740DC899E6F840AE1DFB7F1B3FA81)
set -e
BASE=unsloth-bf16/Qwen3.8-27B-BF16.gguf   # adjust to actual unsloth filename
IMATRIX=unsloth-bf16/imatrix_unsloth.gguf

llama-quantize \
  "$BASE" \
  pareto-bf16.gguf \
  --allow-requantize \
  --imatrix "$IMATRIX" \
  --tensor-type-file configs/pareto-map.txt

# Verify
sha256sum pareto-bf16.gguf
echo "expected: 57DDA505ECD2B0731341C002FF03EDE1526740DC899E6F840AE1DFB7F1B3FA81"
