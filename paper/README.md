# Legacy technical note and historical figures

> **Updated 2026-10-22.** This folder preserves the written account of a
> single-GPU inference experiment. It does not establish an arXiv publication,
> peer review, independent replication, or a current deployment recommendation.

- [paper.md](paper.md): revised technical note separating recorded measurements
  from hypotheses and missing reproduction evidence.
- `figures/fig-depth-curve.png` and `.csv`: historical server/MTP rates against
  author-reported resident depth.
- `figures/fig-k30-vs-stock-mtp.png` and `.csv`: historical selector/stock
  server-path comparison.
- `figures/fig-llamabench.png` and `.csv`: raw-decode curves with MTP disabled;
  selector and stock are nearly unchanged.

The figures and CSVs remain unchanged as historical records. Read their
selected campaign points alongside [the caveats](../notes/caveats.md) and
[the evidence index](../results/README.md). In particular, the approximately
42 tokens/s per-problem mean at reported 200k depth is not a measured
20-minute sustained floor, and the 58.5 tokens/s working-depth value is a
reported rolling-median floor rather than the minimum individual sample.

The exact measured selector implementation and a complete benchmark harness
were not archived. Some original narrative claims lacked supporting files;
the revised note no longer presents them as independently established facts.
Earlier wording remains accessible in Git history.

Paper text and figures retain the [CC-BY-4.0 license](LICENSE).
