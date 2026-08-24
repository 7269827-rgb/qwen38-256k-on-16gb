# AI README (for LLM agents and AI coding tools)

This file is a compressed briefing for an AI agent (Copilot, Claude Code,
Cursor, an autonomous researcher, etc.) that has been pointed at this repo.
Read this first, then `README.md` for the human-facing story and
`results/README.md` for the raw data. If this repo is used to write a
report, post, or paper, the "Hard rules" section at the end is mandatory.

## What this repo is, in one paragraph

A single 16 GB AMD RX 9070 XT (Windows 11, llama.cpp Vulkan, custom r6
build, commit bf0040e15 / b10537) runs Qwen3.8-27B (27B params, 48
Gated-DeltaNet layers + 16 full-attention layers, 262k native context) at
a 256k allocation: ~56-58 t/s at working depth (<=64k resident) with
quality gates passing, and ~42-46 t/s at 200k resident via a
selective-attention tile-skip mechanism. The repo contains the measured
numbers, the exact quant map, the launch flags, and the reproduce scripts.
Everything here is one rig, n=1 sessions.

## The numbers (sanctioned set; do not alter)

Two regimes exist and must never be conflated:

1. Server path, MTP speculative decoding ON (how the model is actually
   used, the headline):
   - Gate (256k alloc, 40k resident): 56.6 t/s 4th-best-of-5
     (`results/pareto-screen-c1..c5.json`), 20-min floor 58.5 t/s
     (`results/sustained-pareto256.json`).
   - 200k resident: 46.16 t/s measured best (ladder), 44.6 fresh boot,
     ~42 sustained. Stock (no skip) ~33 t/s.
     Receipts: `results/bench-r6l2-mh-K30.json`, `bench-r6op-mh.json`,
     `bench-r6l2-mh-STOCK.json`.
2. llama-bench path, MTP OFF (community raw-decode curve, comparable to
   other cards' published results): stock 45.3@4k -> 28.3@200k; K30
   identical within 0.1 t/s at every depth. The selective-skip gain does
   NOT appear on this path. Receipts: `results/bench-lb-depth-*.json`,
   `results/bench-lb-k30-depth-*.json`.

Quality at 200k (selection active): depth-GSM8K 29/30 (96.7%),
`results/bench-depth-gsm8k.json`. Base GSM8K (selection inactive): 144/150
`results/bench-r6op-gsm8k.json` (a different file lives in the research
archive; the repo copy is the K30 operating battery). Multi-hop retrieval
at 200k: 3/3 PASS, but N=3 is a smoke test, not a certification. Draft
acceptance 0.91-0.94.

Keep-all control: 33.00 t/s vs 32.87 stock = zero fixed tax
(`results/bench-r6-keepall.json`).

## How to get the model

Option A: download the exact blob. Filename `pareto-bf16.gguf`, 9,298,488,288
bytes, SHA256 `57DDA505ECD2B0731341C002FF03EDE1526740DC899E6F840AE1DFB7F1B3FA81`.
(Not published on Hugging Face at the time of writing; contact the author
or rebuild per Option B.)

Option B: rebuild from source (all inputs public, `repro/01-03`):
1. From `unsloth/Qwen3.8-27B-GGUF` on Hugging Face, get the BF16 base
   (`BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf` + `-00002-of-00002.gguf`,
   LFS oids `b9966e82...` / `92e3943c...`), the imatrix
   (`imatrix_unsloth.gguf`, oid `0ee5b10b...`), and optionally the MTP
   head (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, oid `50d9ce5a...`).
2. `llama-quantize <BF16-shard-1> pareto-bf16.gguf --allow-requantize
   --imatrix imatrix_unsloth.gguf --tensor-type-file configs/pareto-map.txt`
   (llama.cpp b10537-era; single quantization, no requantize of the base).
3. Verify SHA256 against the value above.

## How to run it

Server (200k K30 configuration, from `configs/launch-flags.md` +
`repro/03-run.sh`):

```
export GGML_VK_FA_SELECT_KEEP=30
export GGML_VK_FA_SELECT_MINKV=32768
llama-server --model pareto-bf16.gguf \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.60 \
  --n-gpu-layers 99 --host 127.0.0.1 --port 8090 --ctx-size 262144 \
  -fa on -ctkd q4_0 -ctvd q4_0
```

The selector is an environment variable read at runtime, not a compile
flag: unset it (or KEEP=100) for dense attention. MINKV=32768 gates
shallow contexts so skipping never engages below 32k of KV.

llama-bench (MTP off, for the comparability curve):

```
llama-bench -m pareto-bf16.gguf -p 512 -n 128 -d <depth> -b 512 -ub 256 \
  -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -r 3 -o json
# depths: 4096 16384 32768 65536 131072 200000
```

Reproduce the gate cells and the K30 ladder: `repro/04-probe.sh`.

## What each directory is for

- `results/` - every JSON behind the table; `results/README.md` is the
  field explainer (what genPerS, docTokens, avg_ts mean).
- `configs/` - `pareto-map.txt` (exact per-tensor quant map),
  `launch-flags.md` (exact commands), `blob-identity.md` (hash + build
  identity + env-var semantics).
- `repro/` - 01 download, 02 quantize, 03 run, 04 probe.
- `paper/` - the arXiv-track paper draft (`paper.md`) + figures and the
  raw CSVs behind them.
- `notes/caveats.md` - the honest box; read before quoting numbers.

## Truth hierarchy

1. `results/*.json` outrank prose. If a doc disagrees with a result file,
   the result file is right.
2. `configs/pareto-map.txt` is the exact quant map used; the blob was
   built from it with the imatrix listed above.
3. Numbers are single-rig, n=1, same-day-interleaved where compared.
   There is a documented bistable fast/slow driver state (~1.36x spread)
   and a ~20% sustained drift. Cross-day deltas are not attributable to a
   version.

## Hard rules (if generating any public text from this repo)

- No em dashes. Use commas, periods, parentheses.
- Label the two number regimes every time: (a) server/MTP on = headline,
  (b) llama-bench MTP off = comparability. Never conflate them.
- Report 46.16 / 44.6 / ~42 sustained as three distinct values. Never
  present 46.16 as the sustained bar.
- Multi-hop is N=3 smoke only; say larger-N is in progress.
- Base GSM8K ran selection-inactive; depth-GSM8K is the selection-active
  gate. Do not cite 144/150 as a selectivity receipt.
- Use "under documented conditions", "we measured", "to our knowledge".
  No superlatives.
- The 2-bit quant is a real quality cliff; quality is held by localizing
  the damage (q2_K only on FFN gate/up), not by claiming 2-bit is
  lossless.
- Credit: the human (thesis: GPU bandwidth underutilization, ~250 vs 640
  GB/s peak, software not physics; oversight), DeepSeek-flash (execution),
  Claude/Fable (analysis, source-checking), Opus (early work, superseded),
  Ox Alpha (pre-registered decision logic), and the Chinese
  selective-attention research community (Quest/NSA lineage).
