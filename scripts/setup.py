#!/usr/bin/env python3
"""Create or validate a private local configuration without printing secrets."""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

INTERNAL_SECRET_NAMES = ("RAG_API_KEY", "RAG_QDRANT_API_KEY")
PROVIDER_SECRET_NAMES = ("RAG_OPENAI_API_KEY", "RAG_VOYAGE_API_KEY")
REQUIRED_TEMPLATE_NAMES = (*INTERNAL_SECRET_NAMES, "RAG_OPENAI_API_KEY")
MINIMUM_INTERNAL_SECRET_LENGTH = 24
MINIMUM_PROVIDER_SECRET_LENGTH = 12


class ConfigurationError(ValueError):
    """A configuration contract failed without exposing a secret value."""


def generate_internal_tokens() -> tuple[str, str]:
    """Return distinct application and Qdrant tokens from 48 random bytes each."""

    application_token = secrets.token_urlsafe(48)
    qdrant_token = secrets.token_urlsafe(48)
    while qdrant_token == application_token:
        qdrant_token = secrets.token_urlsafe(48)
    return application_token, qdrant_token


def _is_placeholder(value: str) -> bool:
    return value.strip().casefold().startswith(("change-me", "replace-me"))


def _validated_openai_key(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ConfigurationError("RAG_OPENAI_API_KEY is required.")
    if _is_placeholder(cleaned):
        raise ConfigurationError("RAG_OPENAI_API_KEY must not contain a placeholder.")
    if len(cleaned) < MINIMUM_PROVIDER_SECRET_LENGTH:
        raise ConfigurationError(
            f"RAG_OPENAI_API_KEY must contain at least {MINIMUM_PROVIDER_SECRET_LENGTH} characters."
        )
    return cleaned


def _render_template(template_text: str, replacements: dict[str, str]) -> str:
    rendered: list[str] = []
    replaced: set[str] = set()
    for line in template_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rendered.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in replacements:
            rendered.append(f"{key}={replacements[key]}")
            replaced.add(key)
        else:
            rendered.append(line)

    missing = sorted(set(REQUIRED_TEMPLATE_NAMES) - replaced)
    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(f"The template is missing required key names: {names}.")
    return "\n".join(rendered).rstrip("\n") + "\n"


def create_environment(
    template: Path,
    destination: Path,
    *,
    openai_key: str,
) -> Path:
    """Create ``destination`` exclusively from ``template`` with mode ``0600``."""

    if destination.exists():
        raise FileExistsError("Configuration already exists; refusing to overwrite it.")

    validated_openai_key = _validated_openai_key(openai_key)
    application_token, qdrant_token = generate_internal_tokens()
    rendered = _render_template(
        template.read_text(encoding="utf-8"),
        {
            "RAG_API_KEY": application_token,
            "RAG_QDRANT_API_KEY": qdrant_token,
            "RAG_OPENAI_API_KEY": validated_openai_key,
        },
    )

    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError("Configuration already exists; refusing to overwrite it.") from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            destination.unlink(missing_ok=True)
        raise

    os.chmod(destination, 0o600)
    return destination


def _read_assignments(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            errors.append(f"Configuration line {line_number} is not a key=value assignment.")
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            errors.append(f"Configuration line {line_number} has an empty key name.")
            continue
        if normalized_key in values:
            errors.append(f"{normalized_key} is defined more than once.")
            continue
        values[normalized_key] = value.strip()
    return values, errors


def validate_environment(path: Path) -> list[str]:
    """Return value-free validation errors for one explicit configuration file."""

    if not path.is_file():
        return ["Configuration file does not exist."]

    values, errors = _read_assignments(path)
    for name in INTERNAL_SECRET_NAMES:
        value = values.get(name, "")
        if not value:
            errors.append(f"{name} is required.")
        elif _is_placeholder(value):
            errors.append(f"{name} still contains a placeholder.")
        elif len(value) < MINIMUM_INTERNAL_SECRET_LENGTH:
            errors.append(
                f"{name} must contain at least {MINIMUM_INTERNAL_SECRET_LENGTH} characters."
            )

    openai_key = values.get("RAG_OPENAI_API_KEY", "")
    if not openai_key:
        errors.append("RAG_OPENAI_API_KEY is required.")
    elif _is_placeholder(openai_key):
        errors.append("RAG_OPENAI_API_KEY still contains a placeholder.")
    elif len(openai_key) < MINIMUM_PROVIDER_SECRET_LENGTH:
        errors.append(
            f"RAG_OPENAI_API_KEY must contain at least {MINIMUM_PROVIDER_SECRET_LENGTH} characters."
        )

    voyage_key = values.get("RAG_VOYAGE_API_KEY", "")
    if voyage_key:
        if _is_placeholder(voyage_key):
            errors.append("RAG_VOYAGE_API_KEY still contains a placeholder.")
        elif len(voyage_key) < MINIMUM_PROVIDER_SECRET_LENGTH:
            errors.append(
                "RAG_VOYAGE_API_KEY must contain at least "
                f"{MINIMUM_PROVIDER_SECRET_LENGTH} characters."
            )

    application_token = values.get("RAG_API_KEY", "")
    qdrant_token = values.get("RAG_QDRANT_API_KEY", "")
    if application_token and application_token == qdrant_token:
        errors.append("RAG_API_KEY and RAG_QDRANT_API_KEY must be different values.")

    embedding_provider = values.get("RAG_EMBEDDING_PROVIDER", "openai").casefold()
    if embedding_provider == "voyage" and not voyage_key:
        errors.append("RAG_VOYAGE_API_KEY is required for Voyage embeddings.")

    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        errors.append("Configuration file permissions must be 0600.")
    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a private .env file or validate it without printing values."
    )
    parser.add_argument("--template", type=Path, default=Path(".env.example"))
    parser.add_argument("--output", type=Path, default=Path(".env"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the output file and print only value-free status messages.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        errors = validate_environment(args.output)
        if errors:
            print("Configuration check failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("Configuration check passed.")
        return 0

    if args.output.exists():
        print("Setup stopped: configuration already exists and was not changed.", file=sys.stderr)
        return 1

    try:
        openai_key = getpass.getpass("OpenAI API key (input hidden): ")
        create_environment(args.template, args.output, openai_key=openai_key)
    except (ConfigurationError, FileNotFoundError, PermissionError) as error:
        print(f"Setup stopped: {error}", file=sys.stderr)
        return 1
    except FileExistsError:
        print("Setup stopped: configuration already exists and was not changed.", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("Setup cancelled; no configuration was written.", file=sys.stderr)
        return 130

    print("Configuration created with private file permissions.")
    print("Next: run docker compose up --build -d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
