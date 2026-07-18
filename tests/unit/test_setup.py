from __future__ import annotations

import stat
from pathlib import Path

import pytest

from scripts import setup


def _template(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "# Minimal configuration",
                "RAG_API_KEY=change-me-generate-a-long-random-value",
                "RAG_QDRANT_API_KEY=change-me-generate-a-different-long-random-value",
                "RAG_OPENAI_API_KEY=",
                "RAG_VOYAGE_API_KEY=",
                "RAG_EMBEDDING_PROVIDER=openai",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_generate_internal_tokens_uses_distinct_48_byte_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = iter(("repeated", "repeated", "different"))
    byte_counts: list[int] = []

    def fake_token_urlsafe(byte_count: int) -> str:
        byte_counts.append(byte_count)
        return next(generated)

    monkeypatch.setattr(setup.secrets, "token_urlsafe", fake_token_urlsafe)

    application_token, qdrant_token = setup.generate_internal_tokens()

    assert application_token != qdrant_token
    assert byte_counts == [48, 48, 48]


def test_create_environment_replaces_placeholders_and_uses_private_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _template(tmp_path / "template.env")
    destination = tmp_path / ".env"
    monkeypatch.setattr(
        setup,
        "generate_internal_tokens",
        lambda: ("application-token-for-test", "qdrant-token-for-test"),
    )

    result = setup.create_environment(
        template,
        destination,
        openai_key="openai-provider-key-for-test",
    )

    assert result == destination
    text = destination.read_text(encoding="utf-8")
    assert "change-me" not in text
    assert "RAG_OPENAI_API_KEY=openai-provider-key-for-test" in text
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_create_environment_refuses_to_overwrite_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _template(tmp_path / "template.env")
    destination = tmp_path / ".env"
    destination.write_text("existing configuration\n", encoding="utf-8")
    monkeypatch.setattr(
        setup,
        "generate_internal_tokens",
        lambda: ("application-token-for-test", "qdrant-token-for-test"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        setup.create_environment(
            template,
            destination,
            openai_key="openai-provider-key-for-test",
        )

    assert destination.read_text(encoding="utf-8") == "existing configuration\n"


def test_validate_environment_reports_placeholders_and_missing_openai_key(
    tmp_path: Path,
) -> None:
    environment = tmp_path / ".env"
    environment.write_text(
        "\n".join(
            (
                "RAG_API_KEY=change-me-application-token",
                "RAG_QDRANT_API_KEY=qdrant-token-at-least-24-characters",
                "RAG_OPENAI_API_KEY=",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment.chmod(0o600)

    errors = setup.validate_environment(environment)

    assert errors == [
        "RAG_API_KEY still contains a placeholder.",
        "RAG_OPENAI_API_KEY is required.",
    ]


def test_check_mode_never_prints_secret_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    application_token = "application-token-at-least-24-characters"
    qdrant_token = "qdrant-token-different-at-least-24-characters"
    provider_key = "openai-provider-key-for-test"
    environment = tmp_path / ".env"
    environment.write_text(
        "\n".join(
            (
                f"RAG_API_KEY={application_token}",
                f"RAG_QDRANT_API_KEY={qdrant_token}",
                f"RAG_OPENAI_API_KEY={provider_key}",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment.chmod(0o600)

    exit_code = setup.main(["--check", "--output", str(environment)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Configuration check passed." in captured.out
    combined_output = captured.out + captured.err
    assert application_token not in combined_output
    assert qdrant_token not in combined_output
    assert provider_key not in combined_output


def test_failed_check_never_prints_other_configured_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    qdrant_token = "qdrant-token-different-at-least-24-characters"
    provider_key = "openai-provider-key-for-test"
    environment = tmp_path / ".env"
    environment.write_text(
        "\n".join(
            (
                "RAG_API_KEY=change-me-application-token",
                f"RAG_QDRANT_API_KEY={qdrant_token}",
                f"RAG_OPENAI_API_KEY={provider_key}",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment.chmod(0o600)

    exit_code = setup.main(["--check", "--output", str(environment)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "RAG_API_KEY" in captured.err
    assert qdrant_token not in captured.out + captured.err
    assert provider_key not in captured.out + captured.err


def test_interactive_setup_uses_hidden_prompt_and_never_echoes_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    template = _template(tmp_path / "template.env")
    destination = tmp_path / ".env"
    provider_key = "openai-provider-key-for-test"
    prompts: list[str] = []

    def fake_getpass(prompt: str) -> str:
        prompts.append(prompt)
        return provider_key

    monkeypatch.setattr(setup.getpass, "getpass", fake_getpass)
    monkeypatch.setattr(
        setup,
        "generate_internal_tokens",
        lambda: ("application-token-for-test", "qdrant-token-for-test"),
    )

    exit_code = setup.main(["--template", str(template), "--output", str(destination)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert prompts == ["OpenAI API key (input hidden): "]
    assert provider_key not in captured.out + captured.err
    assert destination.is_file()
