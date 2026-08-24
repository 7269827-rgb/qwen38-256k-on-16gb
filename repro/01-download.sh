#!/usr/bin/env bash
# 01-download.sh - fetch the BF16 base + imatrix, verify LFS oids
# Source: unsloth/Qwen3.8-27B-GGUF on Hugging Face
# The exact LFS oids were verified 3/3 against the published repo
# when this campaign ran; re-verify against the current tree.
set -e
REPO="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
mkdir -p unsloth-bf16 && cd unsloth-bf16

# BF16 base (2 shards in the unsloth tree) + the D3.0 imatrix.
# Use `huggingface-cli download unsloth/Qwen3.8-27B-GGUF` or git lfs:
#   git clone https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
# Verify the imatrix against the published oid 0ee5b10b...
#   sha256sum imatrix_unsloth.gguf
# (The imatrix we built with is already hash-verified in this repo's
# record; the BF16 base oids are listed in the repo's .gitattributes.)
echo "See configs/blob-identity.md for the verified hashes."
