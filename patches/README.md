# Selective-attention tile-skip — incomplete historical sketch

> **Status correction — 2026-10-22:** `selective-skip.diff` is an incomplete
> reconstruction from design notes. It is not the source used to produce the
> reported measurements, a working selector implementation, or a verified
> behavior-equivalent replacement for the measured binary.

The original notes identify a custom `llama-server-v02-select-r6.exe` build and
a b10537-era llama.cpp base, commit `bf0040e15`. The exact build-machine source
diff was not preserved. The supplied patch is retained unchanged as a historical
artifact; its own comments and older claims must be read with this correction.
The result files are author-reported records, not proof that this sketch
implements the measured behavior.

## What the sketch actually contains

- Environment-variable and dispatch-gate changes intended to enable selection
  during decode with at most four query rows.
- A `ggml_vk_fa_select_blocks` function that only returns
  `(int)(keep_pct * n_blocks)`. It does not compute block scores, select top
  blocks, retain sink/recent blocks, or populate `keep_mask`.
- A `ggml_vk_fa_select_write_mask` function whose body contains comments only.
  It does not write the proposed bitmap into the attention-mask data path.
- Calls referring to query and key-bound inputs whose complete integration and
  maintenance are not supplied here.

Consequently, setting the documented environment variables or applying this
file does not establish a functioning selective-attention implementation. A
successful patch application or compilation would not, by itself, establish
correctness. The earlier description of this reconstruction as "faithful" and
"behavior-equivalent" was not supported by the supplied code.

## Intended mechanism in the historical notes

The proposed mechanism scores 128-token KV blocks per KV head, retains an
attention-sink block and a recent window, and selects additional blocks by a
first-order bound. A two-bit tile bitmap would then use the existing
flash-attention mask path to skip other blocks. This describes the design
intent, not completed logic in this repository. Claims of unchanged stock
dispatch when disabled also require implementation-specific verification.

## Independent implementation and evaluation

Anyone investigating this design needs to implement and review the missing
logic, pin the full source and build configuration, and run fresh correctness,
quality, memory-safety, and performance evaluations. Keep-all equivalence and
disabled-selector behavior need explicit tests against the same stock build.

The historical checks included a keep-all control, a small depth-GSM8K sample,
and repeated retrieval of a fixed question. Their reported outcomes are
documented in [the results index](../results/README.md); they are not sufficient
acceptance criteria for a new implementation or a broad quality certification.
The [reproduction notes](../repro/README.md) describe the remaining workflow
gaps. No completed reconstruction or new benchmark is supplied by this
documentation correction.