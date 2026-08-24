# Security Policy

## Supported versions

This repository publishes measurements and documentation, not a
long-lived software product. The only executable artifact is the set of
repro shell scripts in `repro/`, which wrap llama.cpp and llama-bench
commands. There is no supported version matrix; the current state of the
default branch is the reference.

## Reporting a vulnerability

If you believe you have found a security issue in anything in this
repository, do not open a public issue. Report it privately:

- Use GitHub's private vulnerability reporting for this repository
  (Security tab, "Report a vulnerability"), or
- Open a private issue mentioning "security" in the title.

Please include:

- The file and line involved
- What the issue allows an attacker to do
- A minimal reproduction if possible

## What we take seriously

- Anything in the repro scripts that could execute unexpected commands
- Credential or path handling in documented commands
- Misleading measurement claims that could be used to deceive

## Notes

- The benchmark JSONs contain loopback endpoints (127.0.0.1) that were
  local test servers, not public services.
- The model blob is a research artifact; hash verification instructions
  are in `configs/blob-identity.md`.
