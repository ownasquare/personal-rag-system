"""Fail closed when the tracked repository contains private execution artifacts."""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404
from pathlib import Path

REQUIRED_PUBLIC_FILES = frozenset(
    {
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
    }
)
PRIVATE_PREFIXES = ("docs/handoffs/", "docs/superpowers/", "docs/personal-rag/")
MACHINE_PATH_PATTERNS = (
    re.compile("/" + "Users/" + r"[^/\s]+/"),
    re.compile("/" + "home/" + r"[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\" + "Users" + r"\\[^\\\s]+\\"),
)
INTERNAL_REFERENCE_PATTERNS = (
    re.compile(re.escape("internal-company" + ".example/private-handoffs"), re.IGNORECASE),
    re.compile(re.escape("." + "codex/memories"), re.IGNORECASE),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def scan_text(path: str, text: str) -> list[str]:
    """Return sanitized findings without repeating matched private text."""

    findings: list[str] = []
    if any(pattern.search(text) for pattern in MACHINE_PATH_PATTERNS):
        findings.append(f"{path}: contains a machine-specific home path")
    if any(pattern.search(text) for pattern in INTERNAL_REFERENCE_PATTERNS):
        findings.append(f"{path}: contains an internal handoff reference")
    return findings


def relative_markdown_targets(text: str) -> set[str]:
    """Collect repository-relative Markdown link targets from one document."""

    targets: set[str] = set()
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "/")):
            continue
        targets.add(target)
    return targets


def audit_tracked_files(root: Path, tracked_paths: set[str]) -> list[str]:
    """Audit public files and relative README links using Git-tracked truth."""

    findings = [
        f"{path}: internal execution artifact must not be tracked"
        for path in sorted(tracked_paths)
        if path.startswith(PRIVATE_PREFIXES)
    ]
    for required in sorted(REQUIRED_PUBLIC_FILES - tracked_paths):
        findings.append(f"{required}: required public project file is missing")

    for relative_path in sorted(tracked_paths):
        path = root / relative_path
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            findings.append(f"{relative_path}: tracked file could not be read")
            continue
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Binary artifacts need not be UTF-8, but every decodable public surface is scanned.
            continue
        findings.extend(scan_text(relative_path, text))

    readme = root / "README.md"
    if "README.md" in tracked_paths and readme.is_file():
        for target in sorted(relative_markdown_targets(readme.read_text(encoding="utf-8"))):
            normalized = (readme.parent / target).resolve()
            try:
                relative_target = normalized.relative_to(root.resolve()).as_posix()
            except ValueError:
                findings.append(f"README.md: relative link escapes the repository: {target}")
                continue
            if relative_target not in tracked_paths:
                findings.append(f"README.md: relative link target is not tracked: {target}")
    return findings


def git_tracked_paths(root: Path) -> set[str]:
    """Return the exact tracked path set without invoking a shell."""

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required for public repository checks")
    result = subprocess.run(  # nosec B603
        [git, "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = audit_tracked_files(root, git_tracked_paths(root))
    if findings:
        print("Public repository checks failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Public repository checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
