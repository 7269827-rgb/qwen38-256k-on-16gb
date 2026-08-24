# 256k context on a 16 GB GPU - Qwen3.8-27B, quality-gated, measured

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper: CC-BY-4.0](https://img.shields.io/badge/paper-CC--BY--4.0-lightgrey.svg)](paper/LICENSE)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Vulkan-8A2BE2.svg)](https://github.com/ggml-org/llama.cpp)

Qwen3.8-27B (27B params, 262k native context) at **50+ tokens/s at working
depth with a full 256k allocation**, and **~42-46 t/s at 200k resident with
a selective-attention skip**, on a **16 GB AMD RX 9070 XT** (Windows,
llama.cpp Vulkan) - while holding quality on real long-context tasks. Full
measurement data and a reproducible recipe are in this repo.

- **Community:** [Contributing](.github/CONTRIBUTING.md) ·
  [Code of Conduct](.github/CODE_OF_CONDUCT.md) ·
  [Security](.github/SECURITY.md) · [Author's note](#a-note-from-the-author)
- **AI agents / tools:** read [AI-README.md](AI-README.md) first for a
  compressed briefing and the rules for quoting these numbers.
- **Status:** measurements complete, repo maintained, paper draft in
  review. Replication reports and corrections are welcome (see
  Contributing).

| What | Number | How it was measured |
|---|---|---|
| Gate speed (256k allocated, 40k resident) | **56.6 t/s** (4th-best of 5) | `results/pareto-screen-c1..c5.json` |
| Sustained 20-min floor | **58.5 t/s** | `results/sustained-pareto256.json` |
| **200k resident (selective-skip, MTP on)** | **~42-46 t/s** (46.16 measured, 44.6 fresh boot, ~42 sustained) | `results/bench-r6l2-mh-K30.json`, `results/bench-r6op-mh.json`, `results/bench-depth-gsm8k.json` |
| 200k stock (no skip) | **~33 t/s** | `results/bench-r6l2-mh-STOCK.json` |
| Quality at 200k (depth-GSM8K, selection active) | **29/30 (96.7%)** | `results/bench-depth-gsm8k.json` |
| LongBench v2 (short/medium subset, 25 q) | **14/25 (56%)** | `results/bench-longbench-v2.json` |
| Long-context retrieval | **PASS at 200k** (multi-hop 3/3, smoke-scale) | `results/bench-r6l2-mh-K30.json` |
| llama-bench raw curve (MTP off) | 45.3@4k -> 28.3@200k | `results/bench-lb-depth-*.json` |
| Model footprint | **8.66 GiB, 2.72 BPW** | `configs/pareto-map.txt` |

Numbers are the sanctioned set; result files outrank prose. Every number
in the table has its receipt in `results/`.

## The journey: what we found, what we changed, what came out

Every claim below is a measured before/after, not an opinion. Each row
has its receipt in `results/`.

### 1. Selective tile-skip (the 200k mechanism)

We found the depth wall is the attention re-read, not the math: every
decode step scans the full KV once per verify batch, and the attention
kernel ran at ~250 GB/s vs the card's ~640 GB/s peak (the thesis: the
bottleneck is software, not physics). We changed the flash-attention
tile loop to skip K/V tiles outside a relevance keep-set
(GGML_VK_FA_SELECT_KEEP=30). Before/after at 199.7k resident,
server/MTP path:

| config | t/s | delta | quality |
|---|---|---|---|
| stock (dense) | 32.87 | - | multi-hop 3/3 |
| keep-all control (KEEP=100) | 33.00 | +0.13 (zero fixed tax) | multi-hop 3/3 |
| **K30 (the operating point)** | **46.16** | **+40.4%** | multi-hop 3/3, depth-GSM8K 29/30 |

Receipts: `results/bench-r6l2-mh-STOCK.json`, `results/bench-r6-keepall.json`,
`results/bench-r6l2-mh-K30.json`, `results/bench-depth-gsm8k.json`.

### 2. Quantization placement (how it fits AND keeps quality)

We found a full-2-bit blob is fast but collapses quality, and a
quality-first blob is too slow. We changed the map so the aggressive 2-bit
type sits only on the FFN gate/up tensors (the integer-dot MMVQ fast
path) with iq2_xxs elsewhere. The frontier:

| blob | t/s @64k | GSM8K | note |
|---|---|---|---|
| all-Q2_K | 60.3 | 101/150 | fastest, quality rejected |
| gate-types-only (i-quant mix) | 49.2-50.3 | 145/150 | quality, slower |
| **pareto (2.72 BPW)** | **56.6** | **142/150** | both, fits 16 GB |

Receipts: `results/pareto-screen-c1..c5.json`, `results/sustained-pareto256.json`,
`results/gsm8k-pareto64.json`.

### 3. The MTP x allocation interaction (backend choice inverts a feature)

We found MTP speculative decoding is not uniformly good: on HIP its
benefit inverts with allocation. We ran the clean 2x2 and chose the
Vulkan backend.

| backend | alloc | MTP on | MTP off | MTP delta |
|---|---|---|---|---|
| HIP | 64k | 38.3 | 26.7 | +43% |
| HIP | 256k | 12.6 | 26.6 | -53% |
| **Vulkan** | **256k** | **37.4** | **33.6** | **+11%** |

Receipt: this is a backend bake-off, not in `results/`; see the paper
methods and the repo's measurement record.

### 4. The failure record (what we tried that did not work)

Five mechanisms were falsified with same-build measurements; each one
redirected the work:

| mechanism | measured outcome | receipt ref |
|---|---|---|
| draft-attention-window patch | NEUTRAL at 200k (~31.4 t/s, same band as stock; model attention is global, windows not honored) | W-patch-receipt (research archive) |
| row-fused flash-attention kernel | REGRESSED -33% at 200k (20.2 vs stock, same build/day) | K2-VERDICT (research archive) |
| token-packed flash attention | ENGAGED and NEGATIVE (-33%) | research archive |
| scalar routing at batch 3 | NULL (no amortization) | research archive |
| MAX_NODES dispatch theory | bistable-state mirage (+82% withdrawn after controlled rerun) | research archive |

The decision-log method (publish the failures, not just the wins) is
part of the contribution.

### How it fits inside the card

16 GB must hold the weights AND the KV cache at 256k allocation. The
math: the model is a hybrid, so only 16 of its 64 layers store
full-attention KV (48 are Gated-DeltaNet with O(1) state per token).
That structural fact is what makes 256k physically possible on 16 GB;
the pareto quant (8.66 GiB, 2.72 BPW) plus q4_0 KV cache keeps the
allocation inside free VRAM with a ~180 MiB margin at ngl 99. Detail
in `configs/launch-flags.md` and `configs/blob-identity.md`.

## Why this is interesting

A 27B model with 262k context normally needs a 24 GB+ card. It fits on
16 GB for a structural reason: this model stores KV memory for only 16 of
its 64 layers (48 are Gated-DeltaNet linear attention). The rest is a
speed/quality engineering problem, which this repo documents.

Four measured findings:

1. **The selective-attention tile-skip (the 200k mechanism).** Attention
   skips K/V tiles outside a keep-set during generation
   (GGML_VK_FA_SELECT_KEEP=30). At 200k resident this holds ~42-46 t/s
   vs ~33 stock on the server/MTP path, with zero fixed overhead
   (keep-all 33.00 vs stock 32.87), and depth-GSM8K 29/30 at 200k. In the
   MTP-off llama-bench raw curve the skip is neutral (45.3 vs 45.2 etc) -
   the gain is measured on the speculative-decode path. Builds on the
   Quest/NSA selective-attention research lineage (Chinese open-source
   community).
2. **Profile-driven quant allocation (the frontier).** q2_K only on the
   FFN gate/up tensors (where decode time lives) buys most of a full-2-bit
   blob's speed without its quality collapse (60.3 t/s @ 101/150 vs 56.6
   @ 142/150), with context capacity as the objective.
3. **The MTP x allocation interaction.** On HIP, MTP is +43% at 64k alloc
   but -53% at 256k; on Vulkan it stays +11%. Backend choice can invert a
   feature's benefit.
4. **The bandwidth model (in-model fit).** t/s = C / (W + 18KB x
   resident), C = 288.6 GB/s - predicts the residency curve; explains why
   the attention re-read (not the math) is the depth wall.

## Reproduce it

1. `repro/01-download.sh` - BF16 base (verified against LFS oids) + imatrix.
2. `repro/02-quantize.sh` - llama-quantize --tensor-type-file pareto-map.txt.
3. `repro/03-run.sh` - llama-server with the exact flags + K30 env
   (GGML_VK_FA_SELECT_KEEP=30 GGML_VK_FA_SELECT_MINKV=32768).
4. `repro/04-probe.sh` - reproduce the gate cells + the 200k K30 ladder.

## A note from the author

This is my only GPU, and I am still paying it off. It is also my school
and everything rig, so I could not go as deep into this as I wanted.
I left all the data and receipts here so people can improve on this and
do not waste time on ground I already covered. I hope this helps people
who cannot afford a big rig, and I welcome feedback and replication. If
something here is wrong, show me the receipt and I will fix it.

## Honest caveats (read before quoting numbers)

- One rig, self-consistent (RX 9070 XT, driver 32.0.31035.1003, llama.cpp
  custom build). Not leaderboard-comparable; llama-bench done, RULER
  pending.
- 50+ t/s is at working depth (<=64k resident). At 200k the honest number
  is ~42-46 with the skip (~33 stock); at full 250k resident ~27-32,
  limited by an upstream Vulkan attention-bandwidth issue (#26663 class).
- Two number sets, both real: server/MTP (real usage, ~2.4-2.6x
  amplification) and llama-bench MTP-off (raw decode, comparable to other
  cards). Never conflate them.
- Multi-hop N=3 is a smoke test, not a certification; large-N is pending.
- 2-bit is a real quality cliff; I hold quality by localizing the damage.
- Bistable fast/slow state (44.57 vs 32.67 = 1.36x) + ~20% sustained
  drift; all comparisons same-day, interleaved, cache-hit.

## Files

- `AI-README.md` - compressed briefing for LLM agents / AI coding tools
  (bootstraps fast, includes the hard rules for quoting these numbers)
- `paper/` - the writeup (paper.md) + figures + raw CSVs
- `results/` - every JSON behind the table (results/README.md = field explainer)
- `configs/` - the exact quant map + launch flags + K30 env + blob hash
- `repro/` - 01 download, 02 quantize, 03 run, 04 probe
- `notes/caveats.md` - the honest box

## License

MIT (code/configs, see LICENSE) · CC-BY-4.0 (paper text + figures, see
paper/LICENSE). Built with llama.cpp (MIT), Qwen3.8-27B (Apache-2.0),
Unsloth's GGUF pipeline. Credit: the human (the thesis + oversight),
four AI assistants by role, and the Chinese selective-attention research
community.

## How to cite

If this work is useful to you, a citation or a link back is appreciated
(no formal DOI yet; the repo is the canonical reference).

Plain-text citation:

```
Kindadodgy. (2026). 256k context on a 16 GB GPU - Qwen3.8-27B,
quality-gated, measured. GitHub repository.
https://github.com/7269827-rgb/qwen38-256k-on-16gb
```

BibTeX:

```bibtex
@misc{kindadodgy2026qwen38256k,
  title = {256k context on a 16 GB GPU - {Qwen3.8-27B}, quality-gated, measured},
  author = {Kindadodgy},
  year = {2026},
  howpublished = {\url{https://github.com/7269827-rgb/qwen38-256k-on-16gb}},
  note = {Single-rig measurements; all receipts in the repository}
}
```

If you publish numbers from this repo, please state the two number
regimes (server/MTP-on vs llama-bench MTP-off) as labeled here, and do
not present a single best run as the sustained bar.
