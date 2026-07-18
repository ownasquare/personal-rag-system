# Support

## Questions and setup help

Start with the [README](README.md), [configuration guide](docs/configuration.md), and
[operations guide](docs/operations.md). If the problem remains, open a GitHub issue using the bug
report form.

Include only sanitized diagnostics:

- operating system and architecture;
- Python, `uv`, Docker, and Compose versions;
- the failing command;
- service names and health states;
- the safe application error code; and
- the smallest reproducible sequence.

Do not post `.env` contents, API keys, private document names/content, conversation text, database
files, provider payloads, or full logs that may contain private paths.

## Supported scope

Maintainer support covers the checked-in single-user, single-host application and its documented
Docker Compose or local-development workflows. Multi-user hosting, public internet deployment,
custom identity systems, OCR pipelines, and high availability are architecture extensions rather
than setup support.

Use the feature-request form for a proposed product change. Use the private process in
[SECURITY.md](SECURITY.md) for vulnerabilities.
