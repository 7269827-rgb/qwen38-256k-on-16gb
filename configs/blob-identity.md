# The model blob - identity, hash, how to obtain

## pareto-bf16.gguf

- Size: 9,298,488,288 bytes (8,867.73 MiB)
- SHA256: `57DDA505ECD2B0731341C002FF03EDE1526740DC899E6F840AE1DFB7F1B3FA81`
- Build: llama.cpp b10537 (`build_commit` bf0040e15), Vulkan backend,
  custom r6 selector build (`llama-server-v02-select-r6.exe`), shader
  toolchain glslc 2026.4-dev.
- Quantization: 2.72 BPW mixed blob, single-quantized from the unsloth
  BF16 GGUF base (`unsloth/Qwen3.8-27B-GGUF`, LFS oids verified 3/3).
  The tensor-type map is `configs/pareto-map.txt`; the imatrix is
  unsloth's `imatrix_unsloth.gguf` (SHA256 of the published LFS oid
  `0ee5b10b...` verified against our copy).
- Quantizer: `llama-quantize --allow-requantize --imatrix
  imatrix-unsloth-dl.gguf --tensor-type-file pareto-map.txt`

## The selector mechanism: env var, not a compile flag

`GGML_VK_FA_SELECT_KEEP` and `GGML_VK_FA_SELECT_MINKV` are environment
variables read at runtime by the r6 build. No recompile is needed to
switch between dense and selective operation:

- Unset (or KEEP=100): dense attention, byte-identical dispatch to stock.
- `GGML_VK_FA_SELECT_KEEP=30 GGML_VK_FA_SELECT_MINKV=32768`: the K30
  operating point (skip K/V tiles outside the keep-set; keep-set =
  sink blocks + recent window + query-relevant blocks; MINKV gates
  shallow contexts where skipping is never beneficial).

The keep-all control (KEEP=100 through the same selector machinery)
measures zero fixed tax: 33.00 vs 32.87 t/s stock at 199.7k resident.

## How to reproduce the blob

1. Download the unsloth BF16 base + `imatrix_unsloth.gguf` from
   `unsloth/Qwen3.8-27B-GGUF` on Hugging Face.
2. Verify the LFS oids (see `repro/01-download.sh`).
3. Run llama-quantize with `--tensor-type-file configs/pareto-map.txt`.
4. Check the result hashes to `57DDA505...` (above).

This is a research artifact, not an official unsloth file. The same
numbers are reproducible from the map + imatrix; we publish the map
exactly as used.
