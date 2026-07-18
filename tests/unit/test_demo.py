from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts import demo
from tests.browser.fake_api import TOKEN


def test_child_environment_is_fixture_scoped_and_removes_provider_keys(
    tmp_path: Path,
) -> None:
    inherited: Mapping[str, str] = {
        "PATH": "/safe/bin",
        "OPENAI_API_KEY": "ambient-provider-value",
        "VOYAGE_API_KEY": "ambient-provider-value",
        "RAG_OPENAI_API_KEY": "ambient-provider-value",
        "RAG_VOYAGE_API_KEY": "ambient-provider-value",
        "UNRELATED_SETTING": "preserved",
    }

    child = demo.build_child_environment(inherited, home=tmp_path)

    assert child["RAG_ENVIRONMENT"] == "development"
    assert child["RAG_API_URL"] == "http://127.0.0.1:8012"
    assert child["RAG_API_KEY"] == TOKEN
    assert child["RAG_DEMO_MODE"] == "true"
    assert child["RAG_UI_PORT"] == "8512"
    assert child["HOME"] == str(tmp_path)
    assert child["RAG_OPENAI_API_KEY"] == ""
    assert child["RAG_VOYAGE_API_KEY"] == ""
    assert "OPENAI_API_KEY" not in child
    assert "VOYAGE_API_KEY" not in child
    assert "UNRELATED_SETTING" not in child


def test_demo_commands_use_fixed_loopback_ports() -> None:
    api_command, ui_command = demo.build_commands(Path("/python"))

    assert api_command == [
        "/python",
        "-m",
        "uvicorn",
        "tests.browser.fake_api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8012",
        "--no-access-log",
    ]
    assert ui_command == [
        "/python",
        "-m",
        "streamlit",
        "run",
        "src/personal_rag/ui/app.py",
        "--server.address=127.0.0.1",
        "--server.port=8512",
        "--server.headless=true",
    ]


class _FakeProcess:
    def __init__(self, *, exits_after_terminate: bool) -> None:
        self.exits_after_terminate = exits_after_terminate
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self.killed or (self.terminated and self.exits_after_terminate):
            return 0
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if self.poll() is None:
            raise demo.subprocess.TimeoutExpired("demo", timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


def test_stop_process_terminates_then_kills_after_timeout() -> None:
    cooperative = _FakeProcess(exits_after_terminate=True)
    stubborn = _FakeProcess(exits_after_terminate=False)

    demo.stop_process(cooperative)
    demo.stop_process(stubborn)

    assert cooperative.terminated is True
    assert cooperative.killed is False
    assert stubborn.terminated is True
    assert stubborn.killed is True


def test_run_demo_explains_no_provider_reset_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processes: list[_FakeProcess] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        del command, kwargs
        process = _FakeProcess(exits_after_terminate=True)
        processes.append(process)
        return process

    monkeypatch.setattr(demo.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(demo, "wait_for_service", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        demo.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    exit_code = demo.run_demo()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No provider key or external model call" in captured.out
    assert "reset the next time you start it" in captured.out
    assert "discarding temporary changes" in captured.out
    assert len(processes) == 2
    assert all(process.terminated for process in processes)
