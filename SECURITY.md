# Security policy

## Supported versions

Security fixes are applied to the latest release on the default branch. Older releases are not
maintained unless a release note says otherwise.

## Report a vulnerability privately

Please use GitHub's **Security** tab and **Report a vulnerability** for this repository. That keeps
the report private while a fix is prepared. If private vulnerability reporting is unavailable, do
not open a public issue; contact the repository owner through a private method listed on their
GitHub profile.

Include:

- the affected version or commit;
- a minimal reproduction;
- expected and observed impact;
- whether document content, credentials, or host access may be exposed; and
- any safe mitigation already tested.

Do not include real API keys, private documents, complete database files, or provider responses.
Use synthetic data and redact tokens.

We will acknowledge a complete report, investigate it, coordinate a fix and disclosure, and credit
the reporter if requested. Response timing depends on severity and maintainer availability; no
fixed service-level agreement is promised.

The technical threat model and deployment boundary are documented in
[docs/security.md](docs/security.md). This project is designed for a private, single-user host.
Internet exposure requires TLS and an identity-aware access layer; the built-in bearer token alone
is not represented as an internet-facing identity system.
