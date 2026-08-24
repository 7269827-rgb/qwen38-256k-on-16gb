# Caveats (read before quoting numbers)

- One rig, self-consistent: a single RX 9070 XT (16 GB), Windows 11,
  driver 32.0.31035.1003, llama.cpp Vulkan custom build. Not
  leaderboard-comparable. llama-bench is the comparability harness; RULER
  is pending (needs Python).
- 50+ t/s is at working depth (<= 64k resident). At 200k resident the
  honest number is ~42-46 with the selective-skip (~33 stock). At full
  250k resident: ~27-32, limited by an upstream Vulkan attention-bandwidth
  issue (llama.cpp #26663 class).
- Two number sets, both real: server/MTP (real usage, ~2.4-2.6x
  amplification from speculative decoding) and llama-bench MTP-off (raw
  decode, comparable to other cards). Never conflate them.
- The selective-skip gain is measured on the server/MTP path. In the
  MTP-off llama-bench curve, K30 == stock (45.3 vs 45.2 etc). Say this
  plainly; do not claim a raw-decode win.
- Multi-hop N=3 is a smoke test, not a certification. Large-N is pending.
- 2-bit quantization is a real quality cliff; quality is held by
  localizing the damage (q2_K only on FFN gate/up), not by pretending it
  does not exist.
- The rig has a bistable fast/slow state (clean pair 44.57 vs 32.67 =
  1.36x) and ~20% sustained drift. All comparisons are same-day,
  interleaved, cache-hit.
- The 144/150 base GSM8K ran on bare questions (selection inactive); the
  depth-GSM8K 29/30 at 200k is the selection-active quality gate.
