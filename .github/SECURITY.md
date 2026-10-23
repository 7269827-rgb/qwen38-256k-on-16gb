# Security Policy

## Supported versions

This legacy repository preserves measurements, documentation, shell-script
fragments, and an incomplete experimental source patch. It is not recommended
for new deployments and has no supported version matrix or security-update
commitment. The default branch is the current documentation reference, not a
supported runtime release. Review any script or patch before using it.

## Reporting a vulnerability

Do not put credentials, private data, or sensitive exploit details in a public
issue. If the repository's Security tab offers **Report a vulnerability**, use
that private reporting feature. Its availability depends on the repository's
settings; this document does not establish that it is enabled.

If that option is unavailable, this repository has no documented private
reporting channel. You may open a minimal public issue requesting one, without
disclosing sensitive details. Ordinary GitHub repository issues are public;
there is no "private issue" option for this public repository.

In a private report, include:

- The file and line involved
- What the issue allows an attacker to do
- A minimal reproduction if possible

## Relevant reports

- Anything in the repro scripts that could execute unexpected commands
- Credential or path handling in documented commands
- Documentation that encourages unsafe handling of external artifacts

Measurement corrections without security-sensitive information can be reported
as ordinary issues. There is no guaranteed triage or resolution timeframe.

## Notes

- The benchmark JSONs contain loopback endpoints (127.0.0.1) that were
  local test servers, not public services.
- The model blob is a research artifact; its published SHA256 and provenance
  limitations are recorded in [configs/blob-identity.md](../configs/blob-identity.md).
