# Contributing

Thanks for taking an interest. This is a single-rig measurement repo, so
contributions look a little different from a typical software project.

## What is welcome

1. **Replication reports.** Run the same config on your hardware and
   report what you get. Same card, different card, different OS: all
   useful. Open an issue titled `Replication: <hardware>` with the
   numbers and your launch flags.
2. **Corrections to the measurement record.** If a result file or the
   README table disagrees with what you can reproduce, tell us. Result
   files outrank prose; we will fix the prose.
3. **New measurement points.** Different depths, different KEEP values,
   different quant maps, different backends. Add the JSON receipt under
   `results/` in the same shape as the existing files and document how
   you ran it.
4. **Docs.** Clearer repro scripts, flag explanations, figures.

## What is not welcome

- Benchmark numbers without their receipt (no JSON, no flags, no
  protocol). Numbers without receipts get closed.
- Unverifiable superlatives. Everything here is "under documented
  conditions"; keep that standard.

## Ground rules

- No em dashes in any file. Use commas, periods, parentheses.
- Keep the two number regimes labeled: server/MTP-on (real usage) and
  llama-bench MTP-off (comparability). Never conflate them.
- Never present a single best run (46.16 t/s) as the sustained bar. The
  honest set is 46.16 measured best, 44.6 fresh boot, ~42 sustained.
- Single-rig, n=1, same-day-interleaved methodology. State your rig and
  session conditions in any report.
- No absolute paths, no machine names, no credentials anywhere.

## Process

1. Open an issue first for anything bigger than a typo.
2. Fork, branch, commit with a clear message.
3. Open a pull request referencing the issue. Keep it small and
   receipt-backed.

## License

By contributing you agree your contributions are licensed under the
same terms as the repo: MIT for code and configs, CC-BY-4.0 for paper
text and figures (see LICENSE and paper/LICENSE).
