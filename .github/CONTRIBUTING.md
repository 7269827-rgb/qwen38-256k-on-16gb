# Contributing

This is a legacy research record of a single-GPU quantization and inference
experiment. It is not recommended as a current deployment configuration.
Contributions that improve the accuracy and usefulness of the record are welcome;
there is no maintenance schedule or response-time commitment.

## What is welcome

1. **Independent replication reports.** Read the reproduction limitations in
   the README first: the exact measured selector diff was not preserved, and the
   scripts contain manual and unimplemented steps. If you evaluate a configuration,
   open an issue titled `Replication: <hardware>` with the software revisions,
   launch flags, procedure, and results. Identify any differences from the
   historical experiment.
2. **Corrections to the measurement record.** If a result file or the
   README table disagrees with a saved record, identify the discrepancy. A new
   measurement that differs from an old one should be documented separately;
   it is not a reason to overwrite a historical result.
3. **New measurement points.** Different depths, different KEEP values,
   different quant maps, different backends. Add the JSON receipt under
   `results/` in the same shape as the existing files and document how
   you ran it. Give new records distinct filenames and dates, and preserve
   the original records.
4. **Docs.** Clearer repro scripts, flag explanations, figures.

## What is not welcome

- Benchmark claims without supporting measurements, configuration, and protocol.
- Unverifiable superlatives. Everything here is "under documented
  conditions"; keep that standard.

## Ground rules

- Keep the two number regimes labeled: server/MTP-on and
  llama-bench MTP-off (comparability). Never conflate them.
- Label the historical 46.16 t/s and separate fresh-boot 44.6 t/s figures
  as three-run means. The 42.21 t/s figure is a mean of 30 per-problem
  rates, not a measured 20-minute sustained floor. The working-depth
  58.5 t/s figure is a reported rolling-median floor, not the lowest
  individual sample or a 200k-resident measurement.
- State hardware, software, occupied context depth, timing method, and session
  conditions. Distinguish sequential and interleaved comparisons. The archived
  record does not establish that every comparison was interleaved.
- Remove credentials, private machine identifiers, and personal filesystem paths
  from new submissions. Mark any redactions without changing the measured values.

## Process

1. Open an issue first for anything bigger than a typo.
2. Fork, branch, commit with a clear message.
3. Open a pull request referencing the issue. Keep it small and
   receipt-backed.

## License

By contributing you agree your contributions are licensed under the
same terms as the repo: MIT for code and configs, CC-BY-4.0 for paper text and figures
(see [LICENSE](../LICENSE) and [paper/LICENSE](../paper/LICENSE)).
