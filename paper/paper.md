# 256k Context on a 16 GB GPU: A Legacy Quantization and Inference Experiment

> **Documentation revision — 2026-10-22.** This is a revised technical note
> about historical author-reported measurements, not a current deployment
> recommendation or an independently replicated result. No arXiv publication or
> peer review is established by this repository. This revision corrects
> unsupported descriptions without changing raw result files, figures, or CSVs.
> No new inference benchmark was run. Earlier wording remains in Git history.

## Abstract

The original experiment explored a community mixed-quantization GGUF of
Qwen3.8-27B on one AMD RX 9070 XT with 16 GB VRAM, Windows, and custom
llama.cpp Vulkan builds. The recorded configuration allocated 262,144 context
tokens. Gate-era measurements reported 56.6 tokens/s as the fourth-highest of
five per-launch cache-hit means at approximately 40k resident, and a separate
20-minute record reported a 58.5 tokens/s rolling-median floor. A later custom
selector configuration reported a three-run server/MTP mean of 46.16 tokens/s
at approximately 200k resident, against 32.87 for stock.

These numbers describe different measurements and builds. A separate 42.21
tokens/s average across 30 problems is a workload proxy, not a sustained-floor
test. Later shallow checks did not reproduce the headline working-depth
rates, and their cause remains unresolved. The exact measured selector diff
and a complete benchmark harness were not preserved. The available files
support inspection of the recorded arithmetic and tensor recipe; they do not
establish general quality, a complete replication, or present-day superiority.

## 1. Scope and intended use

This project is a quantization and inference experiment, not the training or
fine-tuning of a new model. The artifact was independently published from the
Unsloth GGUF distribution of Qwen3.8-27B. The documented task is text generation;
the record does not validate image or video inference for this artifact.

The useful questions are how tensor precision was allocated, how reported
throughput changes between benchmark regimes, and which reproduction evidence
is missing. The repository is retained for research reference and independent
replication. It is not recommended as a default configuration for new
deployments.

Allocated context and occupied cache depth are different quantities.
The 262,144-token allocation does not establish reliable inference throughout
that capacity. Server depth labels below follow the author's notes: the saved
JSONs do not independently reconstruct all cache-fill steps.

## 2. Artifact and historical methods

### 2.1 Hardware and build identity

The campaign notes identify one AMD RX 9070 XT with 16 GB VRAM, Windows,
and the Vulkan backend of llama.cpp. The working-depth gate used a
b10502-lineage build with q8_0 draft KV. The later selector records identify a
b10537-era/bf0040e15 custom build with q4_0 draft KV. The working-depth gate
must not be presented as a measurement of that later selector binary.

The source and launch descriptions are incomplete provenance records, not
immutable manifests for every measured executable. Current llama.cpp
compatibility and performance on other systems were not tested by this audit.

### 2.2 Mixed quantization

The artifact is published as `pareto-bf16.gguf`, with a recorded size of
9,298,488,288 bytes and published SHA256:

```text
57dda505ecd2b0731341c002ff03ede1526740dc899e6f840ae1dfb7f1b3fa81
```

The filename is historical. The output is mixed-quantized, not BF16. A bounded
inspection of its header and tensor directory found 866 tensors, no BF16
tensors, and the following types, consistent with
[the tensor map](../configs/pareto-map.txt):

| Type | Tensor count |
|---|---:|
| F32 | 408 |
| IQ2_XXS | 176 |
| Q2_K | 100 |
| IQ3_XXS | 80 |
| Q8_0 | 48 |
| Q4_K | 31 |
| IQ4_XS | 21 |
| Q3_K | 2 |

Q2_K covers 99 FFN gate/up tensors and `token_embd.weight`. Earlier prose
saying it occurred only on FFN tensors, with IQ2_XXS everywhere else, was
incomplete. The file is approximately 2.72 effective bits per tensor element
overall, not uniformly 2-bit. The original notes describe a BF16 source, but
this documentation audit did not independently reproduce the full input-to-output
build lineage or download and hash the entire model.

The original rationale was to place aggressive quantization on tensors thought
to dominate decode cost while retaining other formats elsewhere. The present
record does not independently validate every kernel-path explanation or prove
that this mix preserves quality across general workloads.

### 2.3 Proposed selective-attention mechanism

The original design notes describe selecting 128-token KV blocks for each
layer/head, protecting sink/recent blocks, and choosing additional blocks by
query relevance before skipping others in the attention tile loop. They refer
to `GGML_VK_FA_SELECT_KEEP` and `GGML_VK_FA_SELECT_MINKV` as runtime
controls in a custom executable.

This is a description of design intent. The exact measured source diff was
not archived, and [the reconstructed patch](../patches/selective-skip.diff)
does not implement the full scoring/keep-set or mask-writing logic. It is not
a verified implementation of that description or a behavior-equivalent
replacement for the measured executable. Setting those environment variables
in stock llama.cpp does not supply the custom selector.

The notes cite Quest/NSA-style inspiration. That attribution does not establish
algorithmic equivalence, research novelty, or priority over other implementations.

### 2.4 Measurement protocol and limits

The campaign narrative describes filling a cache before short follow-up
requests, measuring server generation with speculative multi-token prediction
(MTP), and separately using `llama-bench` with MTP disabled. Those paths
must be reported separately.

The cache-fill loader and complete server harness are not present as a runnable
end-to-end workflow. Some JSON fields, including requested `docTokens` and
per-request `promptN`, cannot by themselves establish actual resident depth.
The paper's original ladder description used sequential arms, not a fully
interleaved design. Possible order effects, launch variability, and differing
builds limit causal comparisons.

## 3. Historical recorded results

### 3.1 Working-depth gate

| Statistic | Reported result | Evidence |
|---|---|---|
| Fourth-highest per-launch cache-hit mean out of five | 56.6 tokens/s | `pareto-screen-c1.json` through `pareto-screen-c5.json`; two cached probes per launch |
| Twenty-minute rolling-median floor | 58.5 tokens/s | `sustained-pareto256.json`, field `rollingMedianFloor` |
| Minimum individual sample in that record | 56.12 tokens/s | Minimum of the 20 saved sample rates |
| Gate-era custom GSM8K check | 142/150 | `gsm8k-pareto64.json` |

These are working-depth records (approximately 40k resident, 256 Ki allocated),
not 200k-resident results. The saved rolling-median field does not mean every
sample exceeded 58.5 tokens/s; a complete definition of the original smoothing
procedure is not archived. The gate's aggregation must not be described as the
fourth-highest individual probe.

### 3.2 Selector and stock server/MTP records

| Record | Mean generation rate | Evidence |
|---|---:|---|
| Stock/no selector | 32.87 tokens/s | `bench-r6l2-mh-STOCK.json` |
| Selector KEEP=30 | 46.16 tokens/s | `bench-r6l2-mh-K30.json`, three runs |
| Separate fresh-boot KEEP=30 | 44.6 tokens/s | `bench-r6op-mh.json`, three runs |
| Keep-all control | 33.00 tokens/s | `bench-r6-keepall.json`, five runs |
| Depth-GSM8K speed proxy | 42.21 tokens/s | `bench-depth-gsm8k.json`, mean of 30 per-problem rates |

The author described these as approximately 200k-resident experiments. The
46.16 value is a three-run mean, not the maximum observed run; the underlying
rates are approximately 45.36, 44.91, and 48.20. Its ratio to the recorded stock
mean is approximately 1.404, an observed difference of about 40.4% in those
records. This does not by itself prove causal attribution or reproducibility.

The keep-all mean differs from stock by about 0.13 tokens/s. A small observed
difference is not proof of exactly zero fixed overhead. Similarly, the
42.21 tokens/s arithmetic mean is not aggregate token-weighted throughput and
is not a formal 20-minute floor. Fresh-boot results describe that recorded
launch, not a guarantee for another one.

### 3.3 Raw decoding with MTP disabled

The stock `llama-bench` curve records approximately 45.3 tokens/s at 4,096
tokens of depth and 28.3 at 200,000. The corresponding selector curve is
nearly unchanged at the recorded depths (about 0.12 tokens/s maximum difference).
This dataset does not show a meaningful raw-decode selector gain.

Evidence: `bench-lb-depth-*.json` and
`bench-lb-k30-depth-*.json`. Their metrics must not be mixed with server/MTP
rates or assumed directly comparable to other published runs with different
models, builds, or settings.

### 3.4 Quality samples

| Check | Recorded result | Boundary |
|---|---|---|
| Depth-GSM8K | 29/30 (96.7%) | Small custom sample described as selection-active |
| Later base GSM8K | 144/150 (96.0%) | Bare-question check, selection inactive at shallow occupancy |
| Recorded reference GSM8K | 145/150 | Custom reference, not an official leaderboard baseline |
| Retrieval smoke check | Three successful repetitions | One fixed question with a substring checker |
| LongBench v2 subset | 14/25 (56%) | 22 short and three medium examples; thinking disabled |

Sources are `bench-depth-gsm8k.json`, `bench-r6op-gsm8k.json`,
`gsm8k-dense-baseline.json`, `bench-r6l2-mh-K30.json`, and
`bench-longbench-v2.json`.

The base and depth checks test different conditions. Three repetitions of one
retrieval prompt do not measure performance over three independent tasks.
Custom prompts, answer extraction, sample selection, and stochasticity limit
generalization. The files do not establish broad quality preservation, full
official GSM8K/LongBench performance, or robust long-context reasoning. No
completed RULER evaluation is present.

## 4. Later checks that did not reproduce the headline

[protocol-vs-state-2026-08-24.json](../results/protocol-vs-state-2026-08-24.json)
records six shallow empty/loaded-cache probes from 23.0 to 29.3 tokens/s.
[working-depth-clean-cell-2026-08-24.json](../results/working-depth-clean-cell-2026-08-24.json)
records six rates from 25.2 to 30.5, with a mean of approximately 27.2.

These checks did not reproduce the headline working-depth rates and were
labeled UNCLASSIFIED. The earlier narrative proposed hardware/driver state
variation as an explanation. The preserved record does not establish that
cause or exclude differences in build, protocol, runtime state, or missing
configuration. The slower observations remain part of the evidence.

Original notes also describe unsuccessful kernel/dispatch experiments and
bandwidth/pass-budget hypotheses. Complete executable artifacts and supporting
logs for every such claim are not included. They are historical hypotheses and
reported engineering observations, not independently established diagnoses or
a complete falsification record.

## 5. Figures and interpretation

The PNGs and CSVs in [figures/](figures/) are unchanged historical artifacts:

- `fig-depth-curve`: selected server/MTP rates against author-reported depth.
- `fig-k30-vs-stock-mtp`: historical selector/stock server-path comparison.
- `fig-llamabench`: MTP-disabled raw-decode curves.

A figure is a presentation of selected records, not additional replication.
Labels inherited from the original campaign should be interpreted with the
corrected statistic definitions above. No figure establishes current rankings,
a 200k-resident sustained floor, or universal hardware compatibility.

## 6. Reproduction gaps and possible future work

An independent replication would need an immutable manifest of model inputs,
quantizer/runtime versions, the complete selector implementation, the cache-fill
and scoring harnesses, and full run logs. Measurements should separate build
changes from selector changes, record actual occupied cache depth, use repeated
and interleaved controls, and report both per-request rates and appropriately
weighted totals.

Quality evaluation would need independent retrieval tasks and larger, documented
samples, including failures. Exact source and runtime provenance would be needed
before attributing speed changes to a specific attention mechanism. Those are
requirements for future work, not claims of completed or scheduled experiments.

The current `repro/` scripts contain manual and unimplemented steps. The map
and published checksum alone do not guarantee a byte-identical rebuild or
reproduction of measured rates. The artifact's suitability for a new system
requires fresh testing. No present-day replacement or newer-model superiority
is established by this record.

## 7. Project note, attribution, and licenses

This repository preserves available experimental results and configuration
notes for research reference. Feedback and independent replication findings
can be shared through repository issues.

Credit belongs to Qwen for the base model, Unsloth for the referenced GGUF
distribution, and llama.cpp for the inference/quantization tooling. The original
publication disclosed human oversight and AI-assisted implementation and
analysis. Its Quest/NSA-style attribution describes inspiration, not a verified
implementation or novelty claim.

Model weights retain their declared Apache-2.0 license. The root
[MIT license](../LICENSE) covers repository code/configuration, and this note and
its figures retain [CC-BY-4.0](LICENSE). See
[the artifact notes](../configs/blob-identity.md) for the published model
identity and [the root README](../README.md#credits-and-licenses) for license sources.

The [results index](../results/README.md) explains the saved evidence. Raw records
take precedence over inconsistent prose for what the files report; their
presence does not independently authenticate the original experiment.
