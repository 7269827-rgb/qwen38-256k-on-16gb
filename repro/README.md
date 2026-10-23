# Historical reproduction fragments

> **Status clarification — 2026-10-22:** the numbered shell files are preserved
> historical fragments, not an automated, complete reproduction workflow. They
> have not been repaired or validated by the documentation audit. Running them
> in sequence does not reproduce the published model or benchmark campaign.

Read the [current project limitations](../README.md) and
[artifact identity notes](../configs/blob-identity.md) first. No inference or
quantization was run for this documentation update.

## What each script currently does

| Script | Actual contents and missing steps |
|---|---|
| `01-download.sh` | Creates and enters `unsloth-bf16`, then prints a note. Download and hash-verification commands are comments. It does not download weights or verify any checksum. The imatrix identifier is only the prefix `0ee5b10b...`; the referenced `.gitattributes` is not a full immutable input hash manifest. |
| `02-quantize.sh` | Calls a locally available `llama-quantize` with assumed input filenames, then prints a computed checksum and an expected checksum. It does not acquire or pin a compatible quantizer, prepare the source shards, verify complete input hashes, or fail on an output checksum mismatch. Its expected hash is a historical artifact identifier, not a guarantee that this command recreates those bytes. |
| `03-run.sh` | Exports selector variables and invokes `llama-server`, but passes an empty `--model ""` argument. Its `MODEL` variable is unused and points to a different location from the quantizer output. It does not supply the custom selector binary or its missing source. This is not a working canonical launch as written. |
| `04-probe.sh` | Prints gate and 200k-sweep descriptions; those sections contain no cache-fill, server-probe, or scoring implementation. The executable portion is a six-depth `llama-bench` loop using a locally available binary and model. It is a separate MTP-disabled raw-decode protocol, not the reported MTP-enabled server evaluation. |

The shell files remain unchanged to preserve the published record. Comments
such as "verified," "canonical," or "reproduce" in them are historical claims,
not assertions that the present repository supplies the required workflow.

## Requirements for a new independent replication

A replication would need complete source-model and imatrix identities, an
immutable toolchain/build manifest, usable model paths, and explicit checksum
comparisons. Reproducing the selector experiment also requires a complete
implementation: the [supplied patch](../patches/README.md) contains unfinished
selection and mask-writing logic and is not the measured binary's exact source.
Setting its environment variables on upstream llama.cpp does not add that
implementation.

The server evaluation would additionally need the missing cache-fill and probe
harness, prompts and sampling settings, measured occupied context depth,
run-order and hardware-state records, and suitable quality checks. Allocation
size alone is not occupied context depth. A new experiment should report its
own results and limitations, keeping MTP-enabled server measurements separate
from MTP-disabled raw decode; it should not assume the historical rates will
recur. See the [results index](../results/README.md) for the saved evidence and
the later checks that did not reproduce the headline rates.