from __future__ import annotations

from pathlib import Path

from scripts.check_public_repo import audit_tracked_files, relative_markdown_targets, scan_text


def test_scan_text_rejects_internal_paths_without_repeating_them() -> None:
    machine_path = "/" + "Users/example/private"
    internal_path = "internal-company" + ".example/private-handoffs/item.mdc"
    findings = scan_text(
        "docs/note.md",
        f"Repository: {machine_path} and {internal_path}",
    )

    assert findings == [
        "docs/note.md: contains a machine-specific home path",
        "docs/note.md: contains an internal handoff reference",
    ]
    assert "example" not in repr(findings)


def test_relative_markdown_targets_ignores_remote_and_anchor_links() -> None:
    assert relative_markdown_targets(
        "[Local](docs/guide.md#start) [Remote](https://example.com) [Anchor](#usage)"
    ) == {"docs/guide.md"}


def test_audit_uses_tracked_truth_for_required_files_and_links(tmp_path: Path) -> None:
    required = {
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "SUPPORT.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/pull_request_template.md",
        "docs/guide.md",
    }
    for relative_path in required:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Public\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[Guide](docs/guide.md)\n", encoding="utf-8")

    assert audit_tracked_files(tmp_path, required) == []


def test_audit_rejects_tracked_internal_directories(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "handoffs" / "private.mdc"
    path.parent.mkdir(parents=True)
    path.write_text("private\n", encoding="utf-8")

    findings = audit_tracked_files(tmp_path, {"docs/handoffs/private.mdc"})

    assert "docs/handoffs/private.mdc: internal execution artifact must not be tracked" in findings


def test_audit_scans_python_and_extensionless_public_text(tmp_path: Path) -> None:
    private_reference = "internal-company" + ".example/private-handoffs/item.mdc"
    script = tmp_path / "example.py"
    script.write_text(private_reference, encoding="utf-8")
    makefile = tmp_path / "Makefile"
    makefile.write_text("public target\n", encoding="utf-8")

    findings = audit_tracked_files(tmp_path, {"example.py", "Makefile"})

    assert "example.py: contains an internal handoff reference" in findings
