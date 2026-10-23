# Historical launch flags — legacy research release

**Status: 2026-10-22. Not recommended as a current deployment recipe.** These
are recorded command-line settings from the original single-rig experiments.
They have not been revalidated with current llama.cpp, other hardware, or other
operating systems. See the [repository overview](../README.md) for the status and evidence
limitations. Nothing in this document guarantees the historical throughput.

Model: `pareto-bf16.gguf`, a mixed-quantization Qwen3.8-27B GGUF, approximately
2.72 effective bits per parameter. Despite its historical filename, it is not
a BF16 model.

## Recorded deep-context server configuration

The measured selector path used a custom b10537-era Vulkan binary, with the
reported underlying commit `bf0040e15`. That upstream commit alone does not
identify or reconstruct the custom modifications.

The recorded command was:

```text
llama-server.exe --model pareto-bf16.gguf --cache-type-k q4_0 --cache-type-v q4_0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.60 --n-gpu-layers 99 --host 127.0.0.1 --port 8090 --ctx-size 262144 -fa on -ctkd q4_0 -ctvd q4_0
```

The custom binary's recorded environment settings were:

```text
GGML_VK_FA_SELECT_KEEP=30
GGML_VK_FA_SELECT_MINKV=32768
GGML_VK_FA_TIMING=1
```

`TIMING` was optional. These variables describe the experimental build; setting
them does not add the selector to upstream llama.cpp. The exact measured source
diff was not preserved, and the public reconstructed patch omits operative
selection/mask-writing logic. This document therefore does not provide a
complete executable reproduction of that path.

## Recorded raw-decode benchmark configuration

MTP was disabled on this separate measurement path. The historical benchmark
shape was `-p 512 -n 128 -b 512 -ub 256 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -r 3 -o json`,
with `-d` set to 4096, 16384, 32768, 65536, 131072, or 200000. A complete example
for one recorded depth is:

```text
llama-bench -m pareto-bf16.gguf -p 512 -n 128 -d 200000 -b 512 -ub 256 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -r 3 -o json
```

This is a historical example, not a claim that every current version accepts
the same flags. Server/MTP and raw-decode results must be reported separately.

## Correction: the working-depth gate was a different build

The original 56.6/58.5 tokens/s measurements used a **b10502-lineage gate-era
binary**, approximately 40k reported resident tokens, 262144-token allocation,
and q8_0 draft KV. They were not measurements made by the later b10537-era
selector binary with q4_0 draft KV.

The previous paragraph claiming 40k was below `MINKV=32768` was incorrect:
**40,000 is greater than 32,768**. That paragraph cannot establish whether the
selector was active, nor can it transfer the gate-era speeds to the later binary.

The later clean working-depth cell records a 27.2 tokens/s mean at approximately
4.6k reported resident tokens and explicitly does not reproduce the earlier fast
campaign speeds. It is not a 40k-resident replication. See the pinned
[GitHub disclosure](https://github.com/7269827-rgb/qwen38-256k-on-16gb/blob/c7abad32e4f2465617da890429831ddf534795db/README.md#what-you-should-expect-to-reproduce-read-this-first).
