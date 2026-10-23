# Qwen3.8-27B on 16 GB — legacy quantization and inference experiment

> **Status update — 2026-10-22: legacy; not recommended for new deployments.**
> This release is retained as a record of a single-GPU quantization and inference
> experiment. It is no longer presented as a current model or performance
> recommendation. The figures below are historical, author-reported measurements,
> not current rankings or performance guarantees. No new inference benchmark was
> run for this documentation update. There is no verified like-for-like comparison
> here establishing that a particular newer model outperforms this artifact.

## What this repository contains

This repository contains the available benchmark records, tensor map, historical
launch notes, incomplete reproduction scripts, and a technical writeup for a
**community quantization of Qwen3.8-27B**. The downloadable weight artifact is
published separately on
[Hugging Face](https://huggingface.co/Kindadodgy/qwen38-256k-on-16gb).
The original build notes identify the source as the
[Unsloth GGUF distribution](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
It is not a newly trained foundation model, a fine-tune, an official Qwen release,
or an official Unsloth quantization. The documented experiment used an AMD
RX 9070 XT with 16 GB VRAM, Windows, and llama.cpp's Vulkan backend.

- **Task:** text generation. The archived experiment does not establish working image or
  video inference; the upstream model's broader capabilities are not validation
  of this GGUF configuration.
- **Format:** GGUF, approximately **2.72 effective bits per parameter** overall.
- **Quantization:** a mixed tensor map using Q2_K, IQ2_XXS, IQ3_XXS, IQ4_XS,
  Q3_K, Q4_K, Q8_0, and F32; see [the published tensor map](configs/pareto-map.txt).
  It is not uniformly 2-bit, and the earlier description "IQ2_XXS elsewhere"
  omitted these exceptions.
- **Context:** the recorded configuration allocated **262,144 tokens (256 Ki)**.
  Allocated capacity is different from occupied KV-cache depth and does not
  establish reliable performance across the entire context window.

**Filename warning:** `pareto-bf16.gguf` is the historical filename of the
quantized artifact. The original build notes describe BF16 source weights; the
published output is mixed-quantized, not a full-BF16 release. A bounded inspection
of the published GGUF header found **866 tensors and no BF16 tensors**, matching
the mixed-type map above. The Hub currently
displays a BF16/16-bit label; that label is misleading. The original filename,
download URL, and checksum are retained for existing references.

## Intended use and limitations

Appropriate uses are studying the archived measurements, inspecting the tensor
allocation recipe, and undertaking independent replication. This release is
**not recommended as a default for new deployments or as a production-ready
performance configuration**. Compatibility, output quality, memory use, and
throughput need fresh evaluation on the intended hardware and software.

For new evaluations, consult the maintained
[Qwen organization page](https://huggingface.co/Qwen),
[Unsloth distributions](https://huggingface.co/unsloth), and
[upstream llama.cpp](https://github.com/ggml-org/llama.cpp).
These are starting points for evaluation, not benchmarked replacements endorsed
by this repository. No direct successor to this artifact is designated here.

## Historical results — read with their conditions

The original records are retained in [results/](results/README.md), with the
pre-correction state available at
[revision c7abad32](https://github.com/7269827-rgb/qwen38-256k-on-16gb/tree/c7abad32e4f2465617da890429831ddf534795db).
The table preserves author-reported measurements. Checking the saved arithmetic
is not independent replication of the experiment or authentication of its logs.
These results are not directly comparable with current model leaderboards.

| Measurement | Historically reported result | Conditions and evidence |
|---|---|---|
| Gate speed | 56.6 tokens/s, fourth-highest of five per-launch cache-hit means (two cached probes per launch) | 256 Ki allocated, approximately 40k resident; gate-era b10502-lineage build. `results/pareto-screen-c1..c5.json`. Not a measurement of the later selector binary. |
| Working-depth 20-minute rolling-median floor | 58.5 tokens/s | Reported `rollingMedianFloor` field; this is not the minimum individual sample, which was 56.12 tokens/s. [Original record](https://github.com/7269827-rgb/qwen38-256k-on-16gb/blob/c7abad32e4f2465617da890429831ddf534795db/results/sustained-pareto256.json). Not a 200k-resident floor. |
| Selector, server path with MTP enabled | 46.16 tokens/s three-run mean; separate fresh-boot three-run mean approximately 44.6 | Reported approximately 200k resident; `results/bench-r6l2-mh-K30.json` and `results/bench-r6op-mh.json`. State-dependent campaign results, not individual-run maxima. |
| Per-problem speed proxy | 42.21 tokens/s mean across 30 problems | `results/bench-depth-gsm8k.json`. A workload mean, not a measured 20-minute rolling floor. |
| Stock/no-selector server control | Approximately 33 tokens/s | Reported approximately 200k resident; `results/bench-r6l2-mh-STOCK.json`. |
| Depth-GSM8K check | 29/30 (96.7%) | Small author-run selection-active test; `results/bench-depth-gsm8k.json`. Not a full official GSM8K score or broad quality certification. |
| Retrieval smoke check | Three successful repetitions | `results/bench-r6l2-mh-K30.json`. Repetitions of a fixed question, not three independent retrieval tasks. |
| LongBench v2 subset | 14/25 (56%) | 22 short and three medium examples; ten easy and 15 hard. [Saved record](results/bench-longbench-v2.json). Not the full benchmark; no examples labeled long. |
| Raw decode, MTP disabled | 45.3 tokens/s at 4k to 28.3 at 200k | `results/bench-lb-depth-*.json`. Different measurement regime from the server/MTP rows. |

**A later check did not reproduce the headline speeds.** The saved records
report **23.0–29.3 tokens/s** across six empty/loaded-cache shallow
probes and a separate clean-cell mean of **27.2 tokens/s** (six probes spanning
25.2–30.5). These post-publication checks were unresolved/UNCLASSIFIED. The record leaves
open whether this reflects a slower hardware/driver state or conditions that
could not be reproduced. See
[the caveats](notes/caveats.md),
`results/protocol-vs-state-2026-08-24.json`, and
`results/working-depth-clean-cell-2026-08-24.json`.

## Evaluation and reproduction boundaries

- Server measurements with multi-token prediction/speculative decoding enabled
  and `llama-bench` raw decoding with MTP disabled are different regimes. Their
  tokens/s figures should not be combined into a single speed claim.
- The published records do not support a blanket statement that every comparison
  was interleaved. The paper discusses sequential arms, possible order effects,
  driver-state variation, and sustained drift.
- Resident depth in the server checks is author-recorded; the saved result JSONs
  do not provide a complete independent audit of the cache-fill procedure.
- A small depth-GSM8K sample and a repeated retrieval prompt do not establish
  general long-context reasoning or task quality. The published record does not
  contain a completed RULER evaluation. No new evaluation is promised here.
- The exact diff for the binary used in the selective-attention measurements was
  not preserved. The supplied
  [selector patch](https://github.com/7269827-rgb/qwen38-256k-on-16gb/blob/c7abad32e4f2465617da890429831ddf534795db/patches/selective-skip.diff)
  is an incomplete reconstruction, with unfinished selector/mask logic. It is
  not a verified, behavior-equivalent implementation of the measured binary.
- The reproduction scripts contain manual or unimplemented steps. A tensor map,
  historical command line, and published checksum do not by themselves guarantee
  a byte-identical rebuild or reproduction of the reported speed.
- Other GPU architectures, operating systems, current llama.cpp revisions, and
  general production workloads were not validated by this documentation audit.

## Artifact identity and historical launch notes

The associated Hugging Face repository records one model file:

- File: [`pareto-bf16.gguf`](https://huggingface.co/Kindadodgy/qwen38-256k-on-16gb/resolve/main/pareto-bf16.gguf)
- Size: **9,298,488,288 bytes**, approximately **8.66 GiB**.
- Published SHA256:
  `57dda505ecd2b0731341c002ff03ede1526740dc899e6f840ae1dfb7f1b3fa81`.
- [Tensor map](configs/pareto-map.txt), [artifact notes](configs/blob-identity.md),
  and [historical launch flags](configs/launch-flags.md).

The SHA256 is the published artifact identifier, not a claim that this
documentation audit downloaded and rehashed the entire model. Verify a downloaded
copy before using it. Historical launch examples are retained for research and
must not be read as "works everywhere" instructions or present-day compatibility
guarantees. Upstream llama.cpp does not gain the custom selector merely by setting
the documented environment variables.

## Navigating the archive

| Path | Purpose |
|---|---|
| [results/](results/README.md) | Original JSON measurements and corrected interpretation guide. |
| [configs/](configs/blob-identity.md) | Weight identity, mixed tensor map, and historical launch settings. |
| [repro/](repro/README.md) | Historical shell fragments; not a complete automated reproduction. |
| [patches/](patches/README.md) | Incomplete selector sketch; the measured source diff is missing. |
| [paper/](paper/README.md) | Legacy technical note, original figures, and figure CSVs. |
| [notes/caveats.md](notes/caveats.md) | Evaluation and provenance limitations. |
| [AI-README.md](AI-README.md) | Compact evidence summary for readers and summarization tools. |

See [contribution guidance](.github/CONTRIBUTING.md),
[code of conduct](.github/CODE_OF_CONDUCT.md), and
[security guidance](.github/SECURITY.md). The documentation cleanup does not
promise new experiments, active deployment support, or a peer-reviewed paper.

## Project note

This repository preserves the available experimental results and configuration
notes for research reference. The limitations above describe gaps in the archived
record. Corrections and independent replication reports can be shared through
[repository issues](https://github.com/7269827-rgb/qwen38-256k-on-16gb/issues).

## Credits and licenses

- Base model: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
  [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE).
- Conversion/quantization lineage: Unsloth's GGUF distribution and llama.cpp.
  This artifact is independently published by Kindadodgy.
- The experimental selective-attention work was described as drawing on
  Quest/NSA-style research; this attribution is not an implementation-validation
  claim.
- The original publication disclosed human oversight and AI-assisted
  implementation and analysis.
- Model weights retain their declared **Apache-2.0** license. Recipe/configuration
  code retains the repository's [MIT license](LICENSE);
  paper text and figures retain their declared [CC-BY-4.0 license](paper/LICENSE). This status update does not change
  those licenses or remove the artifact.

## Citing the work

Cite this as a community repository or technical note, not a peer-reviewed paper
or an independently verified performance result. Include the revision you used
and the measurement conditions when quoting a number. For example:

> Kindadodgy. (2026). Qwen3.8-27B on 16 GB: legacy quantization and inference
> experiment. GitHub repository. https://github.com/7269827-rgb/qwen38-256k-on-16gb

The original records and previous prose remain available in Git history.
