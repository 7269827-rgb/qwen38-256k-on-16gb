---
name: Issue
about: Report a problem, a replication result, or a correction
title: ""
labels: ""
assignees: ""
---

Before opening: read the README's legacy status and reproduction limitations.
If a number looks wrong, quote the supporting result file and explain the
discrepancy. Historical records are evidence of the reported measurements,
not independent verification or a guarantee of reproducibility.

## What kind of issue is this?

- [ ] Replication report (I ran the config on my hardware)
- [ ] Correction (a number or claim does not match the receipts)
- [ ] Bug (a repro script or command fails)
- [ ] Question / suggestion

## The details

**Hardware and OS** (for replication reports):

**Software revisions, MTP setting, allocated context, and occupied cache depth:**

**What you ran** (exact command, or point to the repro script):

**What you expected:**

**What happened** (paste output, include numbers and timestamps):

**Receipts** (result files, logs, screenshots):

## Ground rules (from CONTRIBUTING.md)

- Two number regimes stay labeled: server/MTP-on and llama-bench MTP-off.
- Distinguish individual samples, run means, workload means, and rolling floors.
- Do not overwrite historical measurements to match a new run.
- Remove credentials, private machine identifiers, and personal filesystem paths.
- For security-sensitive reports, read SECURITY.md before posting details.
