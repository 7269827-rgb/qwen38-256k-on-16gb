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

| What | Number | How it was measured |
|---|---|---|
| Gate speed (256k allocated, 40k resident) | **56.6 t/s** (4th-best of 5) | `results/pareto-screen-c1..c5.json` |
| Sustained 20-min floor | **58.5 t/s** | `results/sustained-pareto256.json` |
| **200k resident (selective-skip, MTP on)** | **~42-46 t/s** (46.16 measured, 44.6 fresh boot, ~42 sustained) | `results/bench-r6l2-mh-K30.json`, `results/bench-r6op-mh.json` |
| 200k stock (no skip) | **~33 t/s** | `results/bench-r6l2-mh-STOCK.json` |
| Quality at 200k (depth-GSM8K, selection active) | **29/30 (96.7%)** | `results/bench-depth-gsm8k.json` |
| Long-context retrieval | **PASS at 200k** (multi-hop 3/3, smoke-scale) | `results/bench-r6l2-mh-K30.json` |
| llama-bench raw curve (MTP off) | 45.3@4k -> 28.3@200k | `results/bench-lb-depth-*.json` |
| Model footprint | **8.66 GiB, 2.72 BPW** | `configs/pareto-map.txt` |

Numbers are the sanctioned set; result files outrank prose. Every number
in the table has its receipt in `results/`.

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
  custom r6 build). Not leaderboard-comparable; llama-bench done, RULER
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
