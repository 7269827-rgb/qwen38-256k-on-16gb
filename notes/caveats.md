# Caveats for the legacy research record

Updated 2026-10-22. These limitations apply to the README, paper, historical
figures, and any public summary of the results.

- **Status:** retained for research reference; not recommended for new
  deployments. No new inference benchmark was run during this documentation
  audit, and no verified comparison establishes a particular newer model as its
  replacement.
- **Provenance:** one author's records from one RX 9070 XT (16 GB), Windows,
  and custom llama.cpp Vulkan builds. Checking arithmetic and file consistency
  does not authenticate or independently reproduce the experiments.
- **Builds differ:** the 56.6/58.5 working-depth records used the gate-era
  b10502-lineage configuration. The later selector records identify a
  b10537-era/bf0040e15 build. Draft-cache settings also differ.
- **Capacity is not occupancy:** 262,144 allocated tokens do not imply
  262,144 occupied tokens. Server resident depths are author-recorded; complete
  cache-fill logs are absent from the saved results.
- **Rate definitions matter:** 56.6 tokens/s is the fourth-highest of five
  per-launch cache-hit means. 58.5 is a reported 20-minute rolling-median floor,
  not the individual-sample minimum (56.12). At reported approximately 200k
  resident, 46.16 and 44.6 are separate three-run means, while 42.21 is a mean of
  30 per-problem rates, not a measured 20-minute floor.
- **Different inference paths:** server/MTP-enabled rates and
  `llama-bench` MTP-disabled raw-decode rates are not interchangeable.
  The recorded selector gain appears on the server path; the raw-decode curves
  are nearly unchanged.
- **Reproduction remains unresolved:** later shallow probes returned
  23.0–29.3 tokens/s, with a separate clean-cell mean of 27.2
  (range 25.2–30.5). Those checks were UNCLASSIFIED and did not reproduce the
  headline speeds. A driver-state explanation was proposed but not established.
  Sequential arms and possible order effects rule out claiming that every
  comparison was interleaved.
- **Quality checks are limited:** retrieval was three repetitions of one fixed
  question with a substring checker. Depth-GSM8K was 29/30; base GSM8K
  144/150 did not test selection at deep occupancy. LongBench v2 was 14/25 on
  22 short and three medium examples. These are not broad quality certification,
  official full-benchmark scores, or a completed RULER evaluation.
- **Quantization is mixed:** the historical `pareto-bf16.gguf` filename does
  not describe its output precision. The inspected header has no BF16 tensors.
  Q2_K appears on 99 FFN gate/up tensors and `token_embd.weight`, alongside
  seven other tensor types. Neither uniform 2-bit precision nor lossless
  quantization is established.
- **Runtime source is incomplete:** the exact measured selector diff was not
  preserved, and the supplied reconstruction has unfinished selection/mask
  logic. The reproduction scripts also omit manual steps and harness pieces.
  No byte-identical rebuild, behavior-equivalent selector, or guaranteed speed
  reproduction is established.
- **Historical figures are unchanged:** their labels and selected campaign
  points should be interpreted with these caveats and the corrected
  [technical note](../paper/paper.md), not as present-day rankings.
- **No promised follow-up:** larger tests and engineering ideas mentioned in
  earlier notes were proposals, not completed or scheduled work.

See [the evidence index](../results/README.md),
[artifact notes](../configs/blob-identity.md), and
[the root README](../README.md).
