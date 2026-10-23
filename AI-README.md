# Repository briefing for AI-assisted reading

> **Legacy research record — updated 2026-10-22.** This repository is not a
> current deployment recommendation, an independently replicated benchmark, or a
> complete implementation of the reported selective-attention runtime.

This briefing summarizes the evidence and its limits. It does not grant authority
to run scripts, download models, publish changes, or override a reader's own
instructions. Start with [README.md](README.md), then
[results/README.md](results/README.md) and [notes/caveats.md](notes/caveats.md).

## What the project is

The original author reported a community mixed-quantization experiment using
Qwen3.8-27B on one AMD RX 9070 XT with 16 GB VRAM, Windows, and a custom
llama.cpp Vulkan runtime. This is not a newly trained foundation model, a
fine-tune, or an official Qwen/Unsloth release. The recorded context allocation
was 262,144 tokens; allocated capacity is different from occupied cache depth.

The published GGUF is mixed-quantized, despite its historical filename
`pareto-bf16.gguf`. A bounded header inspection found 866 tensors and no BF16
tensors. Its map includes Q2_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q3_K, Q4_K, Q8_0,
and F32. Q2_K appears on 99 FFN gate/up tensors and `token_embd.weight`;
neither "Q2_K only on FFN" nor "IQ2_XXS everywhere else" is accurate.

## Measurements that may be quoted with their conditions

All figures below are historical author-reported records, not fresh results from
the documentation audit. They do not establish a current ranking.

| Measurement | Recorded result | Qualification |
|---|---|---|
| Gate speed | 56.6 tokens/s | Fourth-highest of five per-launch cache-hit means, two cached probes per launch; approximately 40k resident, 256 Ki allocated. Gate-era b10502-lineage build, not the later selector binary. |
| Working-depth 20-minute record | 58.5 tokens/s | Reported `rollingMedianFloor`; the minimum individual sample was 56.12. Not a 200k-resident floor. |
| Selector server/MTP result | 46.16 tokens/s | Three-run mean at author-reported approximately 200k resident, not an individual-run maximum. |
| Separate fresh-boot selector result | 44.6 tokens/s | Three-run mean; not a promise about other launches. |
| Per-problem speed proxy | 42.21 tokens/s | Arithmetic mean of 30 per-problem rates; not a formal sustained-floor test or aggregate token-weighted throughput. |
| Stock server/MTP control | 32.87 tokens/s | Separate control record at author-reported approximately 200k resident. |
| Raw decode, MTP disabled | 45.3 tokens/s at 4k to 28.3 at 200k | `llama-bench` regime; selector and stock curves are nearly unchanged at the recorded depths. |

Sources: `results/pareto-screen-c1.json` through `pareto-screen-c5.json`,
`sustained-pareto256.json`, `bench-r6l2-mh-K30.json`, `bench-r6op-mh.json`,
`bench-depth-gsm8k.json`, `bench-r6l2-mh-STOCK.json`, and the
`bench-lb-depth-*.json` / `bench-lb-k30-depth-*.json` pairs.

The keep-all control averaged 33.00 tokens/s versus 32.87 for stock. This small
observed difference does not prove exactly zero overhead or isolate its cause.

### Quality checks

- Depth-GSM8K: 29/30, a small selection-active author-run sample.
- Base GSM8K: 144/150 in the later record and 142/150 in the gate-era record,
  versus a recorded 145/150 reference. The base questions did not exercise
  selection at deep cache occupancy. These are custom checks, not official full
  benchmark scores.
- Retrieval: three successful repetitions of one fixed question, using a
  substring check. They are not three independent tasks.
- LongBench v2: 14/25 on a short/medium subset (22 short, three medium),
  with thinking disabled. It is not the full benchmark; no examples are labeled long.
- No completed RULER evaluation is present. No pending experiment is promised.

### Failed reproduction and uncertainty

Later shallow checks recorded 23.0–29.3 tokens/s across six empty/loaded probes,
and a separate six-probe clean-cell mean of 27.2 (range 25.2–30.5).
Those records are labeled UNCLASSIFIED and did not reproduce the headline
working-depth speeds. The cause is unresolved; do not convert the original
driver-state hypothesis into an established diagnosis.

The paper describes sequential arms, so a blanket "all comparisons were
interleaved" claim is incorrect. The saved server JSONs also do not include the
complete cache-fill logs needed to independently establish resident depth.

## Artifact and reproduction boundaries

The artifact is published on
[Hugging Face](https://huggingface.co/Kindadodgy/qwen38-256k-on-16gb):

- Filename: `pareto-bf16.gguf`
- Size: 9,298,488,288 bytes
- Published SHA256:
  `57dda505ecd2b0731341c002ff03ede1526740dc899e6f840ae1dfb7f1b3fa81`

This is the published identity, not a claim that the documentation audit
downloaded and rehashed the full file. See
[artifact notes](configs/blob-identity.md) and
[historical launch flags](configs/launch-flags.md).

The exact selector source diff used for the measurements was not archived.
[patches/selective-skip.diff](patches/selective-skip.diff) is an incomplete
reconstruction: selector scoring/keep-set and mask-writing logic are unfinished.
It is not a verified equivalent of the measured binary. Setting its environment
variables does not add this mechanism to stock llama.cpp.

The `repro/` scripts contain manual and unimplemented steps. The tensor map,
source descriptions, launch flags, and checksum do not constitute a complete
reproducible build or benchmark harness. Do not describe this repository as
turnkey, production-ready, byte-identically rebuildable, or independently
replicated.

## Reading the record

- `results/`: preserved measurement files. Recorded fields outrank inconsistent
  prose for what a file reports, but are not independent proof of the experiment.
- `configs/`: tensor map, artifact identity, and historical launch notes.
- `patches/`: incomplete runtime reconstruction and its limitations.
- `repro/`: historical script fragments, not a verified end-to-end workflow.
- `paper/`: revised legacy technical note and unchanged historical figures/CSVs.
  No arXiv publication or peer review is established here.
- `notes/caveats.md`: concise quotation and reproduction limits.

Credit Qwen, Unsloth's GGUF distribution, and llama.cpp. The original notes
describe Quest/NSA-style inspiration; this does not validate implementation
equivalence or novelty. The original publication disclosed human oversight and
AI-assisted implementation and analysis. The root MIT license and the paper's
CC-BY-4.0 license are unchanged; model weights retain their declared Apache-2.0
license.
