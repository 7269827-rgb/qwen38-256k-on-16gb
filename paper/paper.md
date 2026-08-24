# PAPER DRAFT v2 - arXiv track (2026-08-23 evening)
# STATUS: SHIPPING (2026-08-23 user decision; ship bar set at ~42 sustained @200k). arXiv submission pending final peer checks.
# Rules applied: every number traces to a receipt file; "candidate" status explicit;
# no novelty claimed for draft-window (merged elsewhere); selective tile-skip is our
# contribution with the CN lineage credited.

---

## TITLE (draft options)

1. "45 tokens/s at 200k Context on a Single 16 GB Consumer GPU: Quantization
   Placement and Selective Tile-Skip for Hybrid-Attention Models on Vulkan"
2. "Depth Without a Datacenter: Selective KV Tile-Skip and Placement-Aware
   Quantization for 200k-Context Inference on One RDNA4 Card"
(recommend 1; number 2 as blog title)

## ABSTRACT (draft, ~210 words)

Running 27B-class language models at very long context has been framed as a
multi-GPU problem. We show that a single 16 GB consumer GPU (AMD RX 9070 XT,
Windows, Vulkan via llama.cpp) can run a 27B hybrid-attention model
(Qwen3.8-27B: 48 gated-delta-net layers plus 16 full-attention layers) with a
256k allocation and generate at approximately 45 tokens/s at 200k-token
resident context, under documented conditions and pending final certification.
Two mechanisms carry the result. First, a placement-aware quantization recipe
("pareto") applies 2-bit Q2_K only to the feed-forward gate/up tensors, where
decode time concentrates, reaching the integer-dot MMVQ fast path while
preserving quality (GSM8K 144/150 on our self-consistent protocol vs 145/150
dense-reference baseline). Second, a selective-attention tile-skip mechanism
skips K/V tiles outside an attention-relevance keep-set during generation; at
KEEP=30% it delivers a 40% decode speedup over the same build without
selection (46.16 vs 32.87 tokens/s cache-hit at 199.7k resident) with retrieval
and acceptance unchanged. A keep-all control measures zero fixed overhead, and
a simple bandwidth model predicts the measured ladder within 1.5 tokens/s. We
publish the complete decision log including negative results (an attention-
window patch that measured neutral, a fused-kernel design that regressed 33%),
the bistable driver-state characterization, and all receipts. All numbers are
single-rig measurements on one card; certification runs are listed as future
work.

---

## 1. INTRODUCTION

Problem: long-context inference for 27B-class models is presented as a
datacenter capability; budget-hardware users are told to truncate context or
rent compute. Access framing: the people this helps most are often not
equipped with 100+ GB rigs (multilingual release motivation,
PUBLICATION-LANG-PLAN-2026-08-23.md).

Why it is hard on this hardware class:
- 16 GB VRAM must hold weights AND a 200k-token KV cache simultaneously.
- On Windows/AMD the practical backends differ from CUDA: HIP holds ~455-464
  GB/s on attention shapes where Windows Vulkan was measured collapsed to
  ~98 GB/s for hidden>=4096 (llama.cpp issue #26663 - our canonical external
  anchor); the depth wall is therefore kernel-side, not silicon-side.
- Speculative decoding (MTP) changes the cost anatomy: every decode step
  scans the full KV once per verify batch, making attention batch-linear in
  context length.

The claim (one sentence): on ONE 16 GB RDNA4 card under Windows/Vulkan, a
placement-aware 2-bit quantization recipe plus a new selective-attention
tile-skip mechanism deliver 200k-resident generation at ~45 tokens/s with
quality gates intact (candidate; certification battery listed).

Contributions:
C1. The pareto quantization recipe (q2_K concentrated on FFN gate/up) and its
    measured frontier (speed x quality x VRAM fit).
C2. A selective-attention tile-skip mechanism for Vulkan FA (mask_opt path
    engaged at decode shapes), with pre-registered KEEP ladder, keep-all
    control, and quality gating - extending prior CN-community selective-
    attention research (Quest/NSA lineage) to an end-to-end consumer-Vulkan
    validation at 200k, which we find unpublished anywhere (sweep F6).
C3. A measured depth anatomy and pass-budget model for MTP decoding on this
    class (attention ~72% of pass, batch-linear; W=20.7 ms weights; flat
    ~13 ms draft term), predictive to ~1.5 tokens/s.
C4. An honest engineering record: closed config space, bistable driver
    states, and four falsified mechanisms with receipts.

Non-claims (explicit): we do NOT claim novelty for draft-attn-window (merged
in vllm-ascend #10023, speculators #523, sparkinfer #751, ik_llama.cpp #2021);
we measured it neutral on this model anyway (W-patch-receipt-2026-08-23.md -
Qwen3.8's attention is global, windows are not honored). Our GSM8K protocol is
self-consistent, not leaderboard-comparable. All numbers are n=1 rig.

## 2. METHODS

### 2.1 Hardware / software stack
- GPU: AMD RX 9070 XT (gfx1201, RDNA4, 16 GB, 64 CUs). CPU: 16-thread desktop.
  OS: Windows. Stack: llama.cpp fork lineage v0.2.0 (8a35040e0) with local
  builds (rowfuse r5/r6 line for selection); shader toolchain glslc 2026.4-dev.
- Model: Qwen3.8-27B (hybrid: 48 GDN layers + 16 full-attention layers,
  GQA 24:4, head_dim 256, vocab 248,320), MTP nextn layer(s)=1 as draft
  (--spec-draft-n-max 2, p_min 0.60), draft KV q4_0 (DRAFTQ4 config).
- Production blob: pareto-bf16.gguf, 8,867.73 MiB on disk (2.72 BPW):
  Q2_K on the 99 FFN gate/up tensors ONLY, IQ2_XXS elsewhere, 31 Q4_K FFN,
  Q8_0 ssm guards, single-quantized from the verified unsloth BF16 base
  (GATE-MET.md).

### 2.2 The pareto quantization recipe (C1)
Mechanism: on this stack, decode-time matmul throughput is dominated by
whether weights reach the integer-dot MMVQ kernels (K-quants do; i-quants
do not). Placing the aggressive 2-bit type exactly on the tensors that own
most decode time (FFN gate/up) buys the fast path where it pays, while
keeping IQ2_XXS (better per-bit quality, float-dequant path) on the
remaining tensors. Measured frontier (all receipts in GATE-MET.md +
research/q2k-experiment.md): all-Q2_K blob = fastest (60.3 t/s @64k) but
GSM8K 101/150 (rejected); gate-types-only = quality but slow; pareto =
both (56.6 Stage A 4th-best, 58.5 Stage B sustained @40k resident, 256k
alloc; GSM8K 142/150).

### 2.3 Selective-attention tile-skip (C2)
What it does: during generation, a selector computes a keep-set over 128-
token KV blocks per (layer, kv-head); blocks outside the keep-set are
SKIPPED in the flash-attention tile loop (K/V loads never issued) via the
shader's existing mask_opt path (flash_attn_cm2.comp L251-270,
MASK_OPT_ALL_NEG_INF). Keep-set composition: structural protection (sink
blocks + recent window) UNION query-relevant blocks by first-order bounds
(Quest-style min/max metadata; trigram-match blocks maintained at append
time). GGML_VK_FA_SELECT_KEEP=<pct> sets rho; MINKV gates shallow contexts.
What it does NOT do: it does not recompute or approximate attention scores
(no proxy scoring inside the kernel); it does not evict or destroy cache
(dense restore = flip the env off); it does not use trained sparsity.
Enabling work required: relaxing the host decode gate
(nem0 >= block_cols*16 fails at verify shapes, ggml-vulkan.cpp L10914) so
the existing skip path engages at decode; writing the selective bitmap into
data_mask_opt (build-request-select-phase0-2026-08-23.md).
Intellectual credit: the selective-attention lineage (Quest, NSA, SnapKV/
H2O protection results, COBS second-order analysis, opencoti-llamafile's
CUDA block-skip) is the ancestor; the CN-community sweep
(cn-selective-attention-sweep-2026-08-23.md, F1-F11) is our sourced map.
Our contribution is the end-to-end consumer-Vulkan validation at 200k with
pre-registered quality gating - unpublished territory per sweep F6.
Control: keep-all (KEEP=100 through the selector machinery) measures zero
fixed tax (+0.13 t/s vs stock; keepall-control-verdict-2026-08-23.md), so
measured speedups are attributable to skipped tiles, and the residual
budget miss at K30 (~1.5 t/s) is tile-skip inefficiency (frontier partial
tiles), not dispatch overhead.

### 2.4 Benchmark protocol
- Cache-hit protocol v2: fill to depth via chunked thermal loader (<=91 C
  hard abort, <=60 C pre-chunk gates), measure generation on prompt_n<=100
  follow-ups (true cache-hit; v1 artifact documented and fixed,
  ladder-v1-protocol-bug-v2-fix-2026-08-23.md).
- Pre-registration: verdict bands declared before cells
  (ox-select-decision-2026-08-23.md); kill rules fired or explicitly not
  fired, logged.
- Quality gates: multi-hop retrieval at true depth (two facts at 12%/88%,
  combined answer; used as smoke - see Limitations), GSM8K 150-problem
  fixed sample (self-consistent protocol; NOT leaderboard-comparable),
  acceptance tracking per arm (draft accept rate).
- Honesty rules: same-session anchors bracket every comparison; cross-boot
  spread +/-2.25 t/s treated as noise band; fast/slow bistable driver state
  classified per session (StateAudit-2026-08-22.md).

## 3. RESULTS

### 3.1 The selective-skip ladder at 199.7k resident (headline)
Same boot, same fill, sequential arms, stock anchor included
(ladder-v2-verdict-45-cleared-2026-08-23.md; receipts bench-r6l2-*):

| Rung | KEEP | gen mean t/s | vs stock | multi-hop | acceptance p |
|---|---|---|---|---|---|
| STOCK | dense | 32.87 | - | 3/3 | 0.9394 |
| K75 | 75% | 33.80 | +2.8% | 3/3 | 0.9481 |
| K50 | 50% | 38.53 | +17.2% | 3/3 | 0.9427 |
| K40 | 40% | 41.88 | +27.4% | 3/3 | 0.9075 |
| **K30** | **30%** | **46.16** | **+40.4%** | **3/3** | **0.9374** |
| K25 | 25% | 46.90 | +42.7% | **2/3 (miss)** | 0.9143 |

Operating point = K30 (K25 adds 0.7 t/s for the first retrieval failure).
Keep-all control: 33.00 t/s = +0.13 vs stock => zero fixed tax.
Budget model fit: eta=0.80, residual = tile-skip inefficiency; model
predicted 47.6 at K30, measured 46.16 (within 1.5).

### 3.2 Operating-point battery at K30 (fresh boot)
operating-battery-k30-verdict-2026-08-23.md: multi-hop 3/3 (mean 44.6 t/s,
cross-boot drift inside the noise band), acceptance 0.9167 (-0.023, in-gate),
**GSM8K 144/150 (96.0%)** vs 145/150 dense baseline - two arithmetic slips,
attributed to decode stochasticity (no temperature/seed in the runner).

### 3.3 The depth curve and pass anatomy
Figure 1 (done, fig-depth-curve.png/.csv): decode t/s vs resident depth.
Pass anatomy at 200k (f200-measurement-2026-08-23.md): pass = 75.3 ms =
weights 20.7 + draft ~13.0 (flat, depth-independent) + attention 41.6
(batch-linear in KV) [D_fast ~= 0]. Attention is ~55% of pass here and
~72% of the logged-op dump at 250k (250k-profile-2026-08-22.md) - the
selective-skip result follows directly from attacking the batch-linear term.

### 3.4 Working-depth headline (the earlier gate)
At 40k resident / 256k alloc, the pareto recipe alone delivers Stage A
4th-best 56.6 t/s and Stage B 20-min sustained floor 58.5 t/s with GSM8K
142/150 (GATE-MET.md) - i.e., the recipe is the production config today.

### 3.5 The failure record (evidence of rigor; table for F6)
| mechanism | outcome | receipt |
|---|---|---|
| draft-attn-window patch (W) | NEUTRAL at 200k (model attention is GLOBAL - windows not honored); also not novel (merged upstream) | W-patch-receipt-2026-08-23.md |
| row-fused FA kernel (fa_rf) | REGRESSED -33% vs stock same-build | K2-VERDICT-2026-08-23.md |
| token-packed FA (PT packing) | ENGAGED and NEGATIVE (-33%; b2>b3 tile inversion) | BUILD-STATE-v02.md + audit |
| scalar routing at batch 3 (ten-liner) | NULL (no amortization) | s0-mini receipts |
| MAX_NODES dispatch theory | bistable-state mirage (+82% withdrawn after controlled rerun) | notes 08-23 05:08 |
Each failure redirected the campaign (to the dead-time anatomy, then to
selective-skip) - the decision-log method is itself part of the contribution.

## 4. FIGURES LIST (data sources + generator)

- F1 Depth curve [DONE]: decode t/s vs resident context; fig-depth-curve.png/.csv.
- F2 Ladder table/graph: t/s and acceptance vs KEEP at 199.7k; data
  bench-r6l2-*; render as dual-axis line+markers (us, node-canvas).
- F3 Speed-vs-quality scatter vs peers: our configs (pareto@40k, K30@200k)
  vs verified external anchors (dual-GPU rig PER-GPU numbers ~21 t/s/GPU
  @98k per SCALE-UP-NOTE; AMD day-0 51.8 t/s R9700 MTP2; CUDA reference
  12.6 t/s @250k) - axes t/s (per-GPU normalized) vs context depth; sources
  cited inline; WE generate (numbers need care).
- F4 t/s-per-VRAM-GB bars: pareto blob vs alternatives (clean-bf16, all-q2k,
  UD variants) at matched depth; data q2k-experiment.md + GATE-MET.md; us.
- F5 Sustained-floor trace at K30 (20-min rolling median): PENDING ship
  battery; us.
- F6 Failure/closure table: the section 3.5 table as a graphic; us.
- F7 (optional) Pass-anatomy stacked bar: 75.3 ms decomposition at 200k
  (W/draft/attention/D); data f200-measurement doc; us.

## 5. DISCUSSION

For local AI: the combination (placement-aware quant + selective skip) moves
a 27B long-context workload from "needs a datacenter rental" to "runs on the
card under a gaming monitor's power budget". The selective-attention path is
training-free, env-gated, dense-restorable, and costs nothing when fully kept
- properties that make it deployable rather than merely publishable. The
budget model (eta~0.8, zero fixed tax) says further gains come from better
gather efficiency at low rho (frontier partial tiles), and the quality cliff
at KEEP<0.30 says the selector's keep-set composition (structural protection
+ first-order bounds) is currently the binding constraint - second-order
scores (COBS) are the identified upgrade if harder retrieval demands it.
Scaling note (SCALE-UP-NOTE): adding GPUs via layer-split scales fill speed
(1350-1826 tok/s) but not decode per GPU; per-GPU, this single-card result
exceeds a known dual-RDNA4 rig at comparable depth (~21 t/s/GPU @98k).

## 6. LIMITATIONS

1. Candidate, not certified: the FULL ship battery (two sessions, ABAB
   interleave, Stage-B 20-min floor >=45, 4th-best-of-5) has not run yet;
   every headline number above is one-session.
2. Cross-boot spread +-2.25 t/s and launch-order drift +-5 t/s exist;
   sequential arms are bracketed but not interleaved.
3. Bistable driver states (fast/slow ~1.36-1.84x swings by path) exist on
   this rig; sessions were FAST-consistent but state classification must
   accompany every future number.
4. Multi-hop test is a SMOKE test: fixed question, lexically-overlapping
   facts, substring checker, N=3 (Clopper-Pearson upper bound on failure
   ~63%). Chained N>=50 with paraphrase-gap needles is mandatory
   certification and may fail where the smoke passes.
5. GSM8K protocol is self-consistent (own prompt/extraction/fixed 150);
   never leaderboard-comparable.
6. The custom quantization is not leaderboard-comparable either (different
   blob than public releases).
7. Single rig, single GPU vendor, single OS; RDNA4/gfx1201-specific paths
   (coopmat1 FA, mask_opt availability) may differ elsewhere.
8. Selector keep-set is first-order (Quest-class bounds); hard multi-hop
   under-selection is the known failure mode (COBS F2).
9. MTP acceptance depends on the draft seeing consistent context; selector
   masking policy for the draft path is documented but the interaction is
   only bounded by acceptance receipts, not exhaustively mapped.

## 7. RELATED WORK (verified anchors only)

Selective/sparse attention lineage (credited ancestors): Quest block min/max
bounds; SnapKV/H2O/StreamingLLM protection results (scoring-only collapses at
high eviction - F1, arXiv 2605.18053); COBS second-order block covariance
(F2, arXiv 2607.09052); opencoti-llamafile #551 (CUDA-only llama.cpp-family
block-skip; coverage thresholds at 24-48k - F3); FlashPrefill thresholding at
256k (F5, arXiv 2603.06199). Upstream llama.cpp has NO training-free
selective attention (F4): DSA #23346 needs a trained indexer; MiniMax-M3 MSA
#24908 is trained-semantics; SparQ streams all KV. Draft-window precedents
(NOT our claim): vllm-ascend #10023, speculators #523, sparkinfer #751,
ik_llama.cpp #2021. Kernel-side context: llama.cpp #26663 (Windows Vulkan
bandwidth collapse at hidden>=4096), vLLM #47763 (verify-as-decode re-stream).
MagicDec (arXiv 2408.11049) for selective drafting at depth.

## 8. VERIFY-BEFORE-SUBMISSION CHECKLIST

[ ] Ship battery: two sessions, ABAB-interleaved STOCK/K30 arms, Stage-B
    20-min rolling-median floor >= 45 at true 199.7k, 4th-best-of-5 >= 45.
[ ] Chained 2-hop retrieval N >= 50 at K30 (certification; smoke is not
    sufficient) + paraphrase-gap needle variant.
[ ] Depth-GSM8K: 30 suffixes answered AT 199.7k (context-utilization proof,
    not just short-prompt GSM8K).
[ ] State classification attached to every quoted number; fast/slow
    disclosed.
[ ] Peer-table check: every external number re-opened at its source
    (dual-GPU numbers quoted PER-GPU; AMD day-0 config confirmed MTP2).
[ ] GSM8K protocol caveat present in the main text, not a footnote.
[ ] K25 cliff reported (not hidden) with the miss transcript.
[ ] Keep-all zero-tax receipt referenced wherever speedups are attributed.
[ ] Draft-window non-novelty sentence present (cite the four merged PRs).
[ ] CN-community credit line finalized (PUBLICATION-LANG-PLAN section 6).
[ ] Receipt files archived alongside the submission (hashes).


---

## A note from the author

This is my only GPU, and I am still paying it off. It is also my school
and everything rig, so I could not go as deep into this as I wanted.
I left all the data and receipts here so people can improve on this and
do not waste time on ground we already covered. I hope this helps people
who cannot afford a big rig, and I welcome feedback and replication. If
something here is wrong, show me the receipt and I will fix it.

---

## Receipts

Every number above traces to a data file in this repo's ../results/ (see esults/README.md for the field explainer). Companion decision-log documents referenced by name (verdicts, audits, negative results) live in the private research archive and will be published alongside the arXiv submission; the data files themselves are all public here.

