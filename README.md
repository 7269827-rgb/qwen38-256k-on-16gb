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
| Sustained proxy at 200k (30 depth-GSM8K problems) | **42.21 t/s mean** (median 41.99, min 39.07, max 46.46) | `results/bench-depth-gsm8k.json` |
| Long-context retrieval | **PASS at 200k** (multi-hop 3/3, smoke-scale) | `results/bench-r6l2-mh-K30.json` |
| Supporting: LongBench v2 subset (short/medium, 25 q) | **14/25 (56%)** | `results/bench-longbench-v2.json` |
| llama-bench raw curve (MTP off) | 45.3@4k -> 28.3@200k | `results/bench-lb-depth-*.json` |
| Model footprint | **8.66 GiB, 2.72 BPW** | `configs/pareto-map.txt` |

LongBench v2 is a supporting signal (hard bilingual MC benchmark, easy
subset, N=25). The primary quality story is depth-GSM8K at 200k plus
the multi-hop smoke test; see the paper for full context.

Numbers are the sanctioned set; result files outrank prose. Every number
in the table has its receipt in `results/`.

## How it fits inside the card

16 GB must hold the weights AND the KV cache at 256k allocation. The
math: the model is a hybrid, so only 16 of its 64 layers store
full-attention KV (48 are Gated-DeltaNet with O(1) state per token).
That structural fact is what makes 256k physically possible on 16 GB;
the pareto quant (8.66 GiB, 2.72 BPW) plus q4_0 KV cache keeps the
allocation inside free VRAM with a ~180 MiB margin at ngl 99. Detail
in `configs/launch-flags.md` and `configs/blob-identity.md`.

## Daily driver (how people will actually use it)

Two paths, both honest about what you get.

**Path A: stock llama.cpp - works everywhere, no custom build.**

Download the GGUF from Hugging Face, then:

```
llama-server --model pareto-bf16.gguf \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.60 \
  --n-gpu-layers 99 --host 127.0.0.1 --port 8080 --ctx-size 262144 \
  -fa on -ctkd q4_0 -ctvd q4_0
```

You get the MTP boost (~18-24 t/s on short contexts), the full 256k
allocation, and the quality gates intact, on any recent llama.cpp
(b10537-era or newer). At deep context you get stock speeds (~33 t/s at
200k) because the selective-skip is not active on the stock build.

**Path B: the selective-skip build - the 42-46 t/s number.**

Same command plus two env vars (GGML_VK_FA_SELECT_KEEP=30,
GGML_VK_FA_SELECT_MINKV=32768). Requires the custom llama.cpp build
(commit bf0040e15) whose mechanism is documented in `patches/`. Gives
~42-46 t/s at 200k resident with quality intact.

Most users will start with Path A; Path B is the frontier config.

## Why this is interesting

A 27B model with 262k context normally needs a 24 GB+ card. It fits on
16 GB for a structural reason: this model stores KV memory for only 16 of
its 64 layers (48 are Gated-DeltaNet linear attention). The rest is a
speed/quality engineering problem, which this repo documents.

Four measured findings:

1. **The selective-attention tile-skip (the 200k mechanism).** Attention
   skips K/V tiles outside a keep-set during generation
   (GGML_VK_FA_SELECT_KEEP=30). At 200k resident this holds ~42-46 t/s
   vs ~33 stock on the server/MTP path, with zero measurable fixed overhead
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
4. `repro/04-probe.sh` - reproduce the gate runs + the 200k K30 sweep.

## A note from the author

This is my only GPU, and I am still paying it off. It is also my school
and everything rig, so I could not go as deep into this as I wanted.
I left all the data and receipts here so people can improve on this and
do not waste time on ground I already covered. I hope this helps people
who cannot afford a big rig, and I welcome feedback and replication. If
something here is wrong, show me the receipt and I will fix it.

## Honest caveats

- One rig, self-consistent (RX 9070 XT, driver 32.0.31035.1003; llama.cpp
  custom build, commit bf0040e15 / b10537-era, with the selective-skip
  selector; full identity in configs/blob-identity.md). Not
  leaderboard-comparable; llama-bench done, RULER pending.
- 50+ t/s is at working depth (<=64k resident). At 200k the honest number
  is ~42-46 with the skip (~33 stock); at full 250k resident ~27-32,
  limited by an upstream Vulkan attention-bandwidth issue (#26663 class).
- Two number sets, both real: server/MTP (real usage, ~2.4-2.6x
  amplification) and llama-bench MTP-off (raw decode, comparable to other
  cards). Never conflate them.
- Multi-hop N=3 is a smoke test, not a certification; a larger-N run
  (N>=50 with paraphrase-gap needles) is planned.
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
- `patches/` - reconstructed source for the selective-attention tile-skip
  (see the honesty note inside: the build-machine diff was not archived)
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

## How to quote these numbers

Three rules keep these numbers honest; the full brief is in
[AI-README.md](AI-README.md) (the "Hard rules" section).

1. **Always label the regime.** server/MTP-on (real usage) and
   llama-bench MTP-off (comparability) are different numbers. Never
   conflate them.
2. **Never quote 46.16 as the sustained bar.** The honest set is 46.16
   measured best, 44.6 fresh boot, ~42 sustained.
3. **Quality claims at 200k come from depth-GSM8K 29/30 and the
   multi-hop smoke test.** The base GSM8K 144/150 is a build check, not
   a selectivity receipt, and LongBench 14/25 is supporting data on the
   easy subset.

Every number in the table above traces to a receipt in `results/`.
Result files outrank prose.
