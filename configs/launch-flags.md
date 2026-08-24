# Launch flags (exact, reproducible)

Model: pareto-bf16.gguf (Qwen3.8-27B, 2.72 BPW)
Build: llama.cpp Vulkan, custom build with the selective-attention
selector (b10537-era, commit bf0040e15)

## Server (the 200k K30 configuration)

llama-server.exe \
  --model pareto-bf16.gguf \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.60 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 --port 8090 --ctx-size 262144 \
  -fa on -ctkd q4_0 -ctvd q4_0

env:
  GGML_VK_FA_SELECT_KEEP=30      # the selective-attention keep fraction
  GGML_VK_FA_SELECT_MINKV=32768  # below this KV depth, no skipping
  GGML_VK_FA_TIMING=1            # optional: per-op timing

## llama-bench (MTP OFF, community standard)

llama-bench -m pareto-bf16.gguf -p 512 -n 128 -d <depth> -b 512 -ub 256 \
  -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -r 3 -o json

depths: 4096 16384 32768 65536 131072 200000

## The gate config (256k alloc, working depth)

Same server line with ctx 262144. The gate numbers (56.6/58.5 at 40k
resident) ran with the K30 selector configured but BELOW its engagement
threshold: MINKV=32768 means the skip stays off until 32k of KV, so at
40k resident the gate is effectively dense (selector on but never
skipping). It is the same K30 build, not a different binary. The
keep-all control (KEEP=100 through the same selector machinery) measured
+0.13 t/s vs stock (33.00 vs 32.87), so a dormant selector costs nothing
measurable - that is why the gate numbers are valid on this build.
(Note: the original 56.6/58.5 gate measurement was made on the gate-era
b10502-lineage build; the selective-skip build's own working-depth cell
is pending - see README build qualifier.)
