# Artifact identity — legacy mixed-quantization GGUF

**Status update: 2026-10-22.** Retained for historical reference; not recommended
for new deployments. See [the repository overview](../README.md). This document identifies
the artifact and recorded provenance; it does not guarantee a byte-identical
rebuild, current compatibility, or reproduction of the speed measurements.

## Published file

- Historical filename: `pareto-bf16.gguf`.
- Size: **9,298,488,288 bytes**, approximately **8.66 GiB**.
- Published SHA256:
  `57dda505ecd2b0731341c002ff03ede1526740dc899e6f840ae1dfb7f1b3fa81`.
- Hub revision inspected: `c9ce8172c4a9e0416abcc623e2f5061e3571fdb9`.
- Recorded source distribution: `unsloth/Qwen3.8-27B-GGUF`, BF16 source weights.
- Overall artifact size corresponds to approximately **2.72 effective bits per
  parameter**; that average includes file overhead and is not a uniform tensor
  precision.

The `bf16` filename suffix is misleading as a description of the output tensor
types. A bounded GGUF header/tensor-directory inspection found 27,320,697,856
tensor elements and this exact inventory, matching `pareto-map.txt`:

| Tensor type | Tensor count |
|---|---:|
| F32 | 408 |
| Q2_K | 100 |
| Q3_K | 2 |
| Q4_K | 31 |
| Q8_0 | 48 |
| IQ2_XXS | 176 |
| IQ3_XXS | 80 |
| IQ4_XS | 21 |
| **Total** | **866** |

There are **no BF16 tensors** in that inventory. The Q2_K entries consist of
99 FFN gate/up tensors plus `token_embd.weight`; the remaining tensors use
several formats, not exclusively IQ2_XXS. Header inspection is not a full-model
checksum verification or an inference test.

The original file also retains a build-machine path in its quantization metadata.
The bytes and published checksum are preserved; this documentation update does
not sanitize embedded metadata or remove previous repository revisions.

## Recorded quantization provenance and its limits

The original notes describe quantization from the Unsloth BF16 source using
an importance matrix and [the published tensor map](pareto-map.txt). The recorded
quantizer options include `--allow-requantize`, `--imatrix`, and
`--tensor-type-file`. The public notes provide only an abbreviated importance-
matrix hash, so they do not establish a complete immutable build-input manifest.

The output checksum identifies the published file. It does not show that the
currently available source weights, importance matrix, tool versions, and map
will regenerate identical bytes. An independent rebuild would need to record
all full input hashes, the quantizer revision, exact command, and output checksum.

## Experimental selector and inference-build provenance

The later deep-context experiment used a custom llama.cpp Vulkan binary described
as b10537-era, based on commit `bf0040e15`. The earlier working-depth gate numbers
came from a different b10502-lineage build; see [launch notes](launch-flags.md).
The GGUF weights and the inference executable are separate artifacts.

`GGML_VK_FA_SELECT_KEEP` and `GGML_VK_FA_SELECT_MINKV` were settings for the custom
selector binary. The original measured source diff was not preserved. The
[published reconstruction](https://github.com/7269827-rgb/qwen38-256k-on-16gb/blob/c7abad32e4f2465617da890429831ddf534795db/patches/selective-skip.diff)
does not implement its core selection and bitmap-writing functions, so it must
not be described as verified behavior-equivalent source. Environment settings
alone do not enable this feature in upstream llama.cpp.

Recreating a weight file would not recreate that missing implementation or
resolve the unreproduced fast execution conditions. Historical throughput and
quality observations remain author-reported, with the qualifications in the
repository overview. This is an independently published research artifact, not an official
Unsloth release.
