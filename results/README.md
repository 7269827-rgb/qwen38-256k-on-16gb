# results/ - what every file is

Every number quoted in the README table has its receipt here. All runs are
single-rig, same-day-interleaved where comparisons are made, cache-hit
protocol v2 (fill to depth, then measure generation on short follow-ups).

## Conventions

- `genPerS` = generation tokens per second measured on the server path
  (MTP speculative decoding ON, the real-usage configuration).
- `docTokens` = verified resident context length at fill time (244000 =
  ~199.7k tokens: the fill is measured in tokens, not allocation).
- `endpoint` = the loopback address the local llama-server listened on.
  Not a public service.
- llama-bench files: `avg_ts` = average tokens per second for that phase;
  the entry with `n_gen > 0` is the generation (decode) number = "TG".
  llama-bench protocol runs with MTP OFF by design.
- `test_time` = UTC timestamp of the run.

## A note on the "2.0625 bpw" you will see in the llama-bench files

The llama-bench JSONs carry `model_type: "qwen35 27B IQ2_XXS - 2.0625
bpw"`. That is llama-bench printing the GGUF's declared dominant quant
type name (IQ2_XXS), not the actual file-wide bit rate. The pareto blob
is a per-tensor mix: q2_K on the 99 FFN gate/up tensors, IQ2_XXS
elsewhere, 31 q4_K FFN tensors, q8_0 guards. Measured on the real file:
9,298,488,288 bytes x 8 / 27,320,697,856 params = **2.72 BPW**, which is
what the README reports. Both numbers are true; they measure different
things (declared type name vs actual mixed-file density).

## Server-path files (MTP on)

| File | What it is |
|---|---|
| `pareto-screen-c1..c5.json` | The gate screen: 5 fresh launches at 40k resident / 256k alloc. Cache-hit means 60.4/58.1/57.3/56.6/54.5; 4th-best-of-5 = 56.6 t/s (the README headline). `cacheHitMean` = mean of the cache-hit runs in that file. |
| `sustained-pareto256.json` | 20-minute sustained floor at working depth: rolling-median floor 58.5 t/s, pass. |
| `bench-r6l2-mh-K30.json` | The K30 ladder winner: 3 multi-hop runs at 199.7k resident with the selective-skip at KEEP=30. 46.16 t/s peak, 3/3 retrieval PASS. |
| `bench-r6l2-mh-STOCK.json` | Same ladder arm with selection OFF (dense): ~33 t/s, 3/3 PASS. The stock anchor. |
| `bench-r6-keepall.json` | Keep-all control: selector machinery with KEEP=100 (no skipping). 33.00 t/s mean vs 32.87 stock = zero fixed tax. |
| `bench-r6op-mh.json` | Operating-point battery at K30 on a fresh boot: 44.6 t/s mean, 3/3 PASS. |
| `bench-depth-gsm8k.json` | Depth-GSM8K at 199.7k resident, selection ACTIVE: 29/30 (96.7%). Per-problem `genPerS` included (39-46 t/s band). |
| `bench-r6op-gsm8k.json` | K30 operating-battery GSM8K (selection inactive on bare questions): 144/150 (96.0%). The strongest base-GSM8K receipt. NOT a selectivity receipt. |
| `gsm8k-dense-baseline.json` | Dense (unquantized-family) reference GSM8K: 145/150 (96.7%). The baseline the paper compares the quant against. Same 150-problem self-consistent protocol. |
| `bench-longbench-v2.json` | LongBench v2, short/medium subset (25 q, seed-42 deterministic sample, thinking off, temperature 0): 14/25 (56%). Single rig, K30 operating point. Hard bilingual MC benchmark; contexts up to ~46k words. |
| `gsm8k-pareto64.json` | Gate-era base GSM8K (selection inactive, bare questions): 142/150 (94.7%). NOT a selectivity receipt. |

## llama-bench files (MTP off, community protocol)

`bench-lb-depth-{4096,16384,32768,65536,131072,200000}.json` = stock curve.
`bench-lb-k30-depth-{...}.json` = same depths with the K30 env set.

Generation TG (t/s): stock 45.3 / 43.5 / 41.4 / 37.9 / 32.5 / 28.3;
K30 45.2 / 43.5 / 41.5 / 37.8 / 32.4 / 28.3. Identical within 0.1 at every
depth: the selective-skip gain is measured only on the MTP-on server path,
and I say exactly that in the README.

## Why two number sets

- llama-bench runs a fixed protocol with speculative decoding off, so its
  numbers are directly comparable to other cards' published llama-bench
  results.
- The server path is how the model is actually used (MTP on), and the
  selective-skip only helps there. Real-usage speed is higher than the raw
  curve; I report both, labeled, and never mix them.
