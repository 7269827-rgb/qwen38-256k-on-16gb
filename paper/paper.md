# 256k Context on a 16 GB GPU: Quantization Placement and Selective Tile-Skip for Hybrid-Attention Models on Vulkan

## Abstract

Running 27B-class language models at very long context has been framed as a
multi-GPU problem. I show that a single 16 GB consumer GPU (AMD RX 9070 XT,
Windows, Vulkan via llama.cpp) can run a 27B hybrid-attention model
(Qwen3.8-27B: 48 gated-delta-net layers plus 16 full-attention layers) with
a 256k allocation, generating at 56-58 tokens/s at working depth (<=64k
resident) and approximately 42-46 tokens/s at 200k resident context, under
documented conditions. Two mechanisms carry the result. First, a
placement-aware quantization recipe ("pareto") applies 2-bit Q2_K only to
the feed-forward gate/up tensors, where decode time concentrates, reaching
the integer-dot MMVQ fast path while preserving quality (GSM8K 144/150 on a
self-consistent protocol at the K30 operating point vs a 145/150
dense-reference baseline, receipt `results/gsm8k-dense-baseline.json`). Second, a selective-attention tile-skip
mechanism skips K/V tiles outside an attention-relevance keep-set during
generation; at KEEP=30% it delivers a 40% decode speedup over the same
build without selection (46.16 vs 32.87 tokens/s cache-hit at 199.7k
resident) with retrieval and acceptance unchanged. A keep-all control
measures zero fixed overhead, and a simple bandwidth model predicts the
measured ladder within 1.5 tokens/s. I publish the complete measurement
record, including negative results (an attention-window patch that measured
neutral, a fused-kernel design that regressed), the bistable driver-state
characterization, and all receipts. All numbers are single-rig measurements
on one card.

---

## 1. Introduction

Long-context inference for 27B-class models is usually presented as a
datacenter capability; budget-hardware users are told to truncate context
or rent compute. The people this helps most are often not equipped with
100+ GB rigs.

Why it is hard on this hardware class:

- 16 GB VRAM must hold the weights and a 200k-token KV cache at the same
  time.
- On Windows/AMD the practical backends differ from CUDA: HIP holds
  ~455-464 GB/s on attention shapes where Windows Vulkan was measured
  collapsed to ~98 GB/s for hidden>=4096 (llama.cpp issue #26663); the
  depth wall is therefore kernel-side, not silicon-side.
- Speculative decoding (MTP) changes the cost anatomy: every decode step
  scans the full KV once per verify batch, making attention cost
  batch-linear in context length.

The claim, in one sentence: on one 16 GB RDNA4 card under Windows/Vulkan,
a placement-aware 2-bit quantization recipe plus a selective-attention
tile-skip mechanism deliver 256k-allocation generation at 56-58 t/s at
working depth, and ~42-46 t/s at 200k resident, with quality gates intact.

Contributions:

- C1. The pareto quantization recipe (q2_K concentrated on the FFN
  gate/up tensors) and its measured frontier (speed x quality x VRAM fit).
- C2. A selective-attention tile-skip mechanism for Vulkan flash
  attention, with a pre-registered KEEP ladder, a keep-all control, and
  quality gating; this extends the prior selective-attention research
  lineage (Quest/NSA style, largely from the Chinese open-source
  community) to an end-to-end consumer-Vulkan validation at 200k. To my
  knowledge, no prior end-to-end consumer-Vulkan validation at this
  depth with quality gating has been published.
- C3. A measured depth anatomy and pass-budget model for MTP decoding on
  this class (attention ~55-72% of the pass, batch-linear), predictive to
  ~1.5 tokens/s.
- C4. An honest engineering record: the closed config space, the bistable
  driver states, and four falsified mechanisms with receipts.

Non-claims (explicit): I do not claim novelty for the draft-attention-
window patch (already merged upstream in vllm-ascend #10023, speculators
#523, sparkinfer #751, ik_llama.cpp #2021); it measured neutral on this
model anyway, because Qwen3.8's attention is global and windowing is not
honored. The GSM8K protocol used here is self-consistent, not
leaderboard-comparable. All numbers are from one rig.

## 2. Methods

### 2.1 Hardware and software stack

- GPU: AMD RX 9070 XT (RDNA4, 16 GB, 64 CUs). CPU: 16-thread desktop. OS:
  Windows. Stack: llama.cpp Vulkan backend, custom build (commit
  bf0040e15, b10537-era) with the selector build for the tile-skip;
  shader toolchain glslc 2026.4-dev.
- Model: Qwen3.8-27B (hybrid: 48 Gated-DeltaNet layers plus 16
  full-attention layers, GQA 24:4, head_dim 256, vocab 248,320), MTP
  draft head (--spec-draft-n-max 2, p_min 0.60), draft KV q4_0.
- Production blob: pareto-bf16.gguf, 8,867.73 MiB on disk (2.72 BPW):
  Q2_K on the 99 FFN gate/up tensors only, IQ2_XXS elsewhere, 31 Q4_K
  FFN tensors, Q8_0 ssm guards, single-quantized from the verified Unsloth
  BF16 base. SHA256:
  57DDA505ECD2B0731341C002FF03EDE1526740DC899E6F840AE1DFB7F1B3FA81.

### 2.2 The pareto quantization recipe

On this stack, decode-time matmul throughput is dominated by whether the
weights reach the integer-dot MMVQ kernels (K-quants do; i-quants do
not). Placing the aggressive 2-bit type exactly on the tensors that own
most decode time (FFN gate/up) buys the fast path where it pays, while
keeping IQ2_XXS (better per-bit quality, float-dequant path) on the
remaining tensors. The measured frontier: an all-Q2_K blob is fastest
(60.3 t/s at 64k) but GSM8K drops to 101/150 (rejected); gate-types-only
preserves quality but is slow; the pareto mix gets both (56.6 t/s
4th-best-of-5, 58.5 t/s 20-minute sustained floor at 40k resident with
256k allocated; GSM8K 142/150 at the gate, 144/150 at the K30 operating
point).

### 2.3 Selective-attention tile-skip

During generation, a selector computes a keep-set over 128-token KV
blocks per (layer, kv-head); blocks outside the keep-set are skipped in
the flash-attention tile loop (their K/V loads are never issued), via the
shader's existing mask path. Keep-set composition: structural protection
(sink blocks plus a recent window) union query-relevant blocks by
first-order bounds (Quest-style min/max metadata). The environment
variable GGML_VK_FA_SELECT_KEEP sets the keep fraction (rho); MINKV gates
shallow contexts so skipping never engages below 32k of KV.

What it does NOT do: it does not recompute or approximate attention
scores (no proxy scoring inside the kernel); it does not evict or destroy
the cache (dense operation is restored by unsetting the environment
variable); it does not use trained sparsity. Enabling it required relaxing
a host-side decode gate so the existing skip path engages at decode
shapes, and writing the selective bitmap into the mask path of the
shader.

Intellectual credit: the selective-attention lineage (Quest, NSA,
SnapKV/H2O protection results, COBS second-order analysis, and CUDA
block-skip work in the llama.cpp family) is the ancestor; the
Chinese-community selective-attention literature was the sourced map. The
contribution here is the end-to-end consumer-Vulkan validation at 200k
with pre-registered quality gating.

Control: a keep-all run (KEEP=100 through the same selector machinery)
measures zero fixed tax (33.00 t/s vs 32.87 t/s stock), so the measured
speedups are attributable to skipped tiles, and the residual budget miss
at K30 (~1.5 t/s) is tile-skip inefficiency on frontier partial tiles,
not dispatch overhead.

### 2.4 Benchmark protocol

- Cache-hit protocol: fill to depth via a chunked thermal loader (with
  temperature abort gates), then measure generation on short follow-ups
  so every number is a true cache-hit read at the stated depth.
- Pre-registration: verdict bands were declared before runs; kill rules
  were fired or explicitly not fired and logged.
- Quality gates: multi-hop retrieval at true depth (two facts at 12%/88%
  positions, combined answer; used as a smoke test, see Limitations),
  GSM8K on a fixed 150-problem sample (self-consistent protocol, not
  leaderboard-comparable), and draft acceptance tracking per arm.
- Honesty rules: same-session anchors bracket every comparison; cross-
  boot spread is treated as a noise band; the fast/slow bistable driver
  state is classified per session.

## 3. Results

### 3.1 The selective-skip ladder at 199.7k resident

Same boot, same fill, sequential arms, with a dense (stock) anchor:

| Rung | KEEP | gen mean t/s | vs stock | multi-hop | acceptance p |
|---|---|---|---|---|---|
| STOCK | dense | 32.87 | - | 3/3 | 0.9394 |
| K75 | 75% | 33.80 | +2.8% | 3/3 | 0.9481 |
| K50 | 50% | 38.53 | +17.2% | 3/3 | 0.9427 |
| K40 | 40% | 41.88 | +27.4% | 3/3 | 0.9075 |
| **K30** | **30%** | **46.16** | **+40.4%** | **3/3** | **0.9374** |
| K25 | 25% | 46.90 | +42.7% | **2/3 (miss)** | 0.9143 |

Operating point = K30 (K25 adds only 0.7 t/s at the first retrieval
failure). Keep-all control: 33.00 t/s = +0.13 vs stock, i.e. zero fixed
tax. Budget model fit: eta=0.80; the model predicted 47.6 t/s at K30,
measured 46.16 (within 1.5).

### 3.2 Operating-point replication (fresh boot)

Multi-hop 3/3 (mean 44.6 t/s, cross-boot drift inside the noise band),
draft acceptance 0.9167, GSM8K 144/150 (96.0%) vs the 145/150 dense
baseline (`results/gsm8k-dense-baseline.json`): two arithmetic slips,
attributed to decode stochasticity (the runner sends no temperature or
seed).

### 3.3 The depth curve and pass anatomy

Decode t/s vs resident depth is plotted in the figures (fig-depth-curve).
Pass anatomy at 200k: a 75.3 ms pass = weights 20.7 ms + draft ~13.0 ms
(flat, depth-independent) + attention 41.6 ms (batch-linear in KV).
Attention is ~55% of the pass at 200k and ~72% of the logged-op dump at
250k; the selective-skip result follows directly from attacking the
batch-linear term.

### 3.4 Working-depth headline

At 40k resident / 256k alloc, the pareto recipe alone delivers 56.6 t/s
4th-best-of-5 and a 58.5 t/s 20-minute sustained floor with GSM8K
142/150: the recipe is the production configuration today.

### 3.5 The failure record

| mechanism | outcome |
|---|---|
| draft-attention-window patch | NEUTRAL at 200k (model attention is global; windows are not honored); also not novel (merged upstream) |
| row-fused flash-attention kernel | REGRESSED -33% vs stock, same build |
| token-packed flash attention | ENGAGED and NEGATIVE (-33%) |
| scalar routing at batch 3 | NULL (no amortization) |
| MAX_NODES dispatch theory | bistable-state mirage (+82% withdrawn after a controlled rerun) |

Each failure redirected the work toward the dead-time anatomy and then
the selective-skip; the decision-log method is itself part of the
contribution.

## 4. Figures

- F1 fig-depth-curve: decode t/s vs resident context (server/MTP path,
  stock and K30).
- F2 fig-k30-vs-stock-mtp: the K30 vs stock comparison on the MTP-on
  path.
- F3 fig-llamabench: the llama-bench MTP-off raw curve (stock and K30
  nearly overlap: the skip is neutral on the raw path).
- The raw CSVs behind all three figures are included in the repo.

## 4.1 Peer context (verified at source, 2026-08-24)

Checked the public record before publishing: community benchmark tables
(the sudoingX/qwen38-mtp table and its radeon sweep notes), Hugging Face
model cards, r/LocalLLaMA, and Chinese community boards.

- At true 200k resident on a single 16 GB AMD card with quality gates:
  to my knowledge, no published result matches 46.16 t/s plus multi-hop
  3/3 at depth plus depth-GSM8K 29/30 with selection active (GSM8K
  144/150 on bare questions is a build-regression check, not a
  selectivity receipt).
- The closest single-16GB peers all measure at much shallower depth:
  an RX 7900 GRE 16 GB row reports 47.8 t/s average but at 29-37k
  context only (5-6x shallower), and a reported RX 9070 XT at 46 t/s
  was measured at 32k context.
- Higher-VRAM cards (RTX 5090 32 GB, RTX 4090 24 GB, RX 7900 XTX
  24 GB) beat this work on raw short-context speed. That is expected and
  not a claim to the contrary: the differentiator here is depth (200k
  resident) without collapsing speed, via the selective-attention
  tile-skip, with quality gates intact, on the smallest VRAM tier.

Sources: github.com/sudoingX/qwen38-mtp (community table and
sweeps/radeon.md), the kbin post by cicadagen (32k context stated),
and the sudoingX table's 4090/5090 rows.

## 5. Discussion

For local AI, the combination (placement-aware quantization plus
selective skip) moves a 27B long-context workload from "needs a
datacenter rental" to "runs on the card under a gaming monitor's power
budget". The selective-attention path is training-free, env-gated,
dense-restorable, and costs nothing when fully kept: properties that make
it deployable rather than merely publishable. The budget model (eta ~0.8,
zero fixed tax) says further gains come from better gather efficiency at
low rho (frontier partial tiles), and the quality cliff at KEEP<0.30 says
the selector's keep-set composition (structural protection plus
first-order bounds) is currently the binding constraint; second-order
scores are the identified upgrade if harder retrieval demands it.

Scaling note: adding GPUs via layer-split scales fill speed (1350-1826
tok/s) but not decode per GPU. Per GPU, this single-card result exceeds a
known dual-RDNA4 rig at comparable depth (~21 t/s/GPU at 98k).

## 6. Limitations

1. All headline numbers are single-session, single-rig, n=1
   measurements. The full certification set (two sessions, ABAB
   interleave, sustained floor) is planned but has not run.
2. Cross-boot spread of +-2.25 t/s and launch-order drift of +-5 t/s
   exist; sequential arms are bracketed but not interleaved.
3. Bistable driver states (fast/slow, ~1.36-1.84x swings by path) exist
   on this rig; sessions were fast-consistent, but state classification
   must accompany every future number.
4. The multi-hop test is a smoke test: fixed question, lexically
   overlapping facts, substring checker, N=3 (Clopper-Pearson upper bound
   on failure ~63%). Chained N>=50 with paraphrase-gap needles is
   mandatory certification and may fail where the smoke passes.
5. The GSM8K protocol is self-consistent (own prompt, own extraction,
   fixed 150 sample); it is never leaderboard-comparable.
6. The custom quantization is not leaderboard-comparable either (a
   different blob than public releases).
7. Single rig, single GPU vendor, single OS; RDNA4-specific paths
   (coopmat1 flash attention, mask_opt availability) may differ
   elsewhere.
8. The selector keep-set is first-order (Quest-class bounds); hard
   multi-hop under-selection is the known failure mode.
9. The selective-skip source change is published as a reconstruction in
   `patches/` (written from the build notes; the exact build-machine
   diff was not archived, so the shipped binary is the source of record
   for the measurements). Anyone rebuilding from the reconstruction
   must re-run the quality gates before trusting it.
10. MTP acceptance depends on the draft seeing consistent context; the
    selector masking policy for the draft path is documented, but the
    interaction is only bounded by acceptance receipts, not exhaustively
    mapped.
11. LongBench v2 was run on a 25-question short/medium subset at the
    K30 operating point (14/25 = 56%; contexts up to ~46k words, thinking
    off, temperature 0). RULER is the next benchmark step; the quality
    story rests on depth-GSM8K at 200k, base GSM8K, the multi-hop smoke
    test, and the LongBench subset.

## 7. Related work

Selective/sparse attention lineage (credited ancestors): Quest block
min/max bounds; SnapKV/H2O/StreamingLLM protection results (scoring-only
collapses at high eviction; arXiv 2605.18053); COBS second-order block
covariance (arXiv 2607.09052); opencoti-llamafile #551 (CUDA-only
llama.cpp-family block-skip; coverage thresholds at 24-48k);
FlashPrefill thresholding at 256k (arXiv 2603.06199). Upstream llama.cpp
has no training-free selective attention: DSA #23346 needs a trained
indexer; MiniMax-M3 MSA #24908 is trained-semantics; SparQ streams all
KV. Draft-window precedents (not claimed here): vllm-ascend #10023,
speculators #523, sparkinfer #751, ik_llama.cpp #2021. Kernel-side
context: llama.cpp #26663 (Windows Vulkan bandwidth collapse at
hidden>=4096), vLLM #47763 (verify-as-decode re-stream), and MagicDec
(arXiv 2408.11049) for selective drafting at depth.

---

## A note from the author

This is my only GPU, and I am still paying it off. It is also my school
and everything rig, so I could not go as deep into this as I wanted.
I left all the data and receipts here so people can improve on this and
do not waste time on ground I already covered. I hope this helps people
who cannot afford a big rig, and I welcome feedback and replication. If
something here is wrong, show me the receipt and I will fix it.

---

## Credits

Model: Qwen3.8-27B (Apache-2.0) and the Unsloth GGUF pipeline (BF16
base, imatrix, MTP heads). Tooling: llama.cpp. The selective-attention
tile-skip builds on the Chinese open-source selective-attention research
lineage (Quest/NSA style). This work was produced with human oversight
plus AI assistants by role: DeepSeek-flash (execution), Claude and Opus
(analysis), and an anonymous advisor (pre-registered decision logic).

## Receipts

Every number above traces to a data file in this repository's
`results/` directory (see `results/README.md` for the field explainer).
The result files are the source of record; if prose and a result file
disagree, the result file is right. The complete decision log and
negative-result records are archived alongside this submission.
