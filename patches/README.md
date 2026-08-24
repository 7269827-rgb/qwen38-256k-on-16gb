# Selective-attention tile-skip - source reconstruction

This directory contains a faithful reconstruction of the source change
behind the selective-attention tile-skip used for the K30 operating
point, written from the project's build and design notes
(PIONEER-SELECTIVE.md, R5-SELECT-NOTES.md, the Phase-0 build request,
and the paper methods section).

IMPORTANT HONESTY NOTE: the exact source diff from the build machine
was NOT preserved. The selector shipped as an assembled binary
(`llama-server-v02-select-r6.exe`); only its parts manifests and SHA256
survive on disk. The binary is the source of record for every number in
`results/` - the measurements stand on their own regardless of this
file. This reconstruction is behavior-equivalent to the documented
mechanism but is NOT the literal build diff: it is written from the
project's build and design notes (PIONEER-SELECTIVE.md, R5-SELECT-NOTES.md,
the Phase-0 build request, and the paper methods section) against the
same llama.cpp base (b10537-era, commit bf0040e15), using the documented
code locations and mechanism. It is intended for maintainers and
reviewers to verify the approach. Anyone rebuilding from this
reconstruction MUST re-run the quality gates (below) before trusting it;
the numbers here were produced by the shipped binary, not by this
reconstruction. Do not diff this file against the binary and expect a
literal match.

## What it does (one paragraph)

During decode (query rows N <= 4), a host-side selector scores every
128-token KV block per kv-head and writes a 2-bits-per-tile bitmap into
the flash-attention mask_opt path. Tiles outside the keep-set are marked
ALL_NEG_INF, and the existing shader skip path never issues their K/V
loads. The keep-set is: tile 0 (attention sink) + the newest
GGML_VK_FA_SELECT_RECENT tiles (default 2) + top-(KEEP%) tiles by a
first-order bound score. Env-gated: unset KEEP = byte-identical stock
dispatch.

## The three changes

1. Env parsing + selector (ggml-vulkan.cpp, host side)
2. Relax the decode gate so mask_opt engages at N<=4 (ggml-vulkan.cpp,
   the use_mask_opt condition near the FA dispatch)
3. Selector output written into data_mask_opt (the fa_mask_opt
   pipeline; the shader itself needed no change - scalar/cm1/cm2 all
   honor the mask path)

## Files

- `selective-skip.diff` - the reconstructed patch (unified diff)
- `README.md` - this file

## Quality gates to re-run after any rebuild

- KEEP=100 sanity: output must match stock to ULP class (bitmap inert
  when everything is selected)
- multi-hop retrieval at true >=199.7k resident: 3/3 PASS
- depth-GSM8K at 199.7k with selection active: 29/30
- acceptance |delta| <= 0.05 vs stock at the operating point
