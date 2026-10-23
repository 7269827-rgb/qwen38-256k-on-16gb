# Historical result records

These files preserve author-reported measurements from one AMD RX 9070 XT
system. The raw JSON records have not been changed by the 2026-10-22
documentation audit. Checking their arithmetic is not independent reproduction
or authentication of the experiments. Full prompts, cache-fill logs, evaluation
harnesses, and the exact custom selector source are not all available here.

The records cover different builds and measurement regimes. The gate campaign
used a b10502-lineage build with q8_0 draft KV; later selector runs used a
b10537-era build with q4_0 draft KV. The paper describes sequential comparison
arms and possible order effects, so the earlier blanket statement that all
comparisons were interleaved was too strong.

## Reading the fields

- `genPerS` is the recorded generation rate in tokens per second for a server
  probe. The historical server experiments used MTP/speculative decoding.
- `resident` and `depthCached` record the author's context-depth attribution.
  Several retrieval files contain `docTokens: 244000` while the notes attribute
  those runs to approximately 199.7k resident tokens. That field alone does not
  demonstrate the actual occupied KV-cache depth; the full fill logs are absent.
- `endpoint` is a loopback address used by the local server, not a public service.
- In the llama-bench files, `avg_ts` is the average token rate for an entry; an
  entry with `n_gen > 0` supplies the decode result. These runs used MTP disabled.
- Timestamps and configuration fields are retained as reported, where present.
  They do not constitute a complete runtime or hardware-state manifest.

## Quantization labels

The llama-bench records print `qwen35 27B IQ2_XXS - 2.0625 bpw`. This is a
quantization-type label, not the effective density of the mixed artifact. The
published file uses F32, IQ2_XXS, Q2_K, IQ3_XXS, Q8_0, Q4_K, IQ4_XS, and Q3_K
tensors. The file-wide calculation is
9,298,488,288 bytes × 8 / 27,320,697,856 tensor elements = approximately
**2.72 bits per element**, including file overhead. See
[artifact identity](../configs/blob-identity.md) and the
[tensor map](../configs/pareto-map.txt). The filename `pareto-bf16.gguf` does
not mean the published output contains BF16 tensors.

## Server-path records

| File | Recorded result and its limits |
|---|---|
| `pareto-screen-c1.json` through `pareto-screen-c5.json` | Five launches attributed to approximately 40k resident tokens with 262,144 allocated. Each `cacheHitMean` averages two cached probes. Sorted means are 60.4, 58.1, 57.3, 56.6, and 54.5 tokens/s; **56.6 is the fourth-highest of those five means**, not a single-probe minimum. These are gate-era results, not a measurement of the later selector build. |
| `sustained-pareto256.json` | Twenty samples in a record labeled 20 minutes. Reported `rollingMedianFloor` is **58.5 tokens/s**; the minimum individual sample is **56.12 tokens/s**. The original rolling-window procedure is not specified in this file. This working-depth record is not a 200k-resident sustained floor. |
| `bench-r6l2-mh-K30.json` | Three repetitions of a fixed retrieval question, attributed to approximately 199.7k resident tokens and KEEP=30. **46.16 tokens/s is the arithmetic mean of three rates**, not the maximum. Three recorded `ok: true` outcomes are a repeated smoke check, not three independent retrieval tasks. |
| `bench-r6l2-mh-STOCK.json` | Three stock/no-selector retrieval repetitions, mean **32.87 tokens/s**, with three recorded successful checks. |
| `bench-r6-keepall.json` | Five keep-all retrieval repetitions, mean **33.00 tokens/s**. The small difference from the separate 32.87 stock mean does not establish zero overhead under all conditions. |
| `bench-r6op-mh.json` | A separate run described as a fresh-boot K30 check: three retrieval repetitions, mean **44.60 tokens/s**, three successful checks. Not a fresh-launch performance guarantee. |
| `bench-depth-gsm8k.json` | A 30-problem selection-active check attributed to approximately 199.7k resident tokens: **29/30 (96.7%)** recorded correct. The unweighted arithmetic mean of its 30 per-problem rates is **42.21 tokens/s**. This is a workload proxy, not total-token/total-time throughput or a measured 20-minute floor. It is not the full official GSM8K benchmark. |
| `bench-r6op-gsm8k.json` | Summary-only 150-problem result: **144/150 (96.0%)**. The historical notes describe bare questions with selection inactive. It does not establish selector quality at depth. |
| `gsm8k-dense-baseline.json` | Summary-only reference result: **145/150 (96.7%)**. The JSON lacks the full input-model identity, prompts, individual answers, and harness required to independently verify a controlled comparison. |
| `gsm8k-pareto64.json` | Summary-only gate-era result: **142/150 (94.7%)**, described as bare questions with selection inactive. It does not establish selector quality at depth. |
| `bench-longbench-v2.json` | **14/25 (56%)** on a recorded subset: **22 short and 3 medium** examples, comprising **10 easy and 15 hard** examples. It is neither an easy-only subset nor a full LongBench v2 score, and contains no examples labeled long. Historical notes describe thinking off and temperature 0; the result file alone is not a full reproducible harness. |

The repeated retrieval outputs use the same Okafor/14 answer. Their recorded
success flags and small sample size cannot establish general retrieval or
long-context reasoning quality. The published record contains no completed
RULER evaluation.

## Raw llama-bench records, MTP disabled

The stock series is `bench-lb-depth-<depth>.json`; the K30-environment series is
`bench-lb-k30-depth-<depth>.json`. Both cover depths 4,096, 16,384, 32,768, 65,536,
131,072, and 200,000.

Rounded decode rates in tokens/s are:

| Depth | Stock | K30 environment |
|---|---:|---:|
| 4,096 | 45.3 | 45.2 |
| 16,384 | 43.5 | 43.5 |
| 32,768 | 41.4 | 41.5 |
| 65,536 | 37.9 | 37.8 |
| 131,072 | 32.5 | 32.4 |
| 200,000 | 28.3 | 28.2 |

These records show no material speed improvement from the K30 environment in
this raw-decode series. The reported selector improvement occurred in the
separate MTP-enabled server campaign. That observation does not establish a
general causal result across other builds or workloads. Cross-system benchmark
comparisons require matched models, settings, software, hardware, and protocol;
using llama-bench alone does not make results directly comparable.

## Later checks that did not reproduce headline speeds

- `protocol-vs-state-2026-08-24.json`: six shallow probes, three empty-cache and
  three loaded-cache, span **23.0–29.3 tokens/s**. The record is marked
  **UNCLASSIFIED**.
- `working-depth-clean-cell-2026-08-24.json`: six listed probes span
  **25.2–30.5 tokens/s**, with mean approximately **27.2 tokens/s**. Its protocol
  description says three probes, but the saved array contains six; the numbers
  here use the array. This record is also **UNCLASSIFIED**.

These are unresolved replication limitations. The saved record does not settle
whether slower hardware/driver state or other unreproduced conditions caused
the gap. It should not be attributed conclusively to the driver. See the
[main status and limitations](../README.md), the
[incomplete selector sketch](../patches/README.md), and the
[historical script inventory](../repro/README.md) before attempting independent
replication.
