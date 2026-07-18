#!/usr/bin/env python3
"""Run the deterministic Personal Library product tour without provider calls."""

from __future__ import annotations

import http.client
import os
import subprocess  # nosec B404
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from tests.browser.fake_api import TOKEN as FIXTURE_TOKEN

API_HOST = "127.0.0.1"
API_PORT = 8012
UI_HOST = "127.0.0.1"
UI_PORT = 8512
STARTUP_TIMEOUT_SECONDS = 30.0
SAFE_INHERITED_NAMES = (
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "WINDIR",
)


class DemoError(RuntimeError):
    """The deterministic demo could not start or stay running."""


class ProcessLike(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


def build_child_environment(
    inherited: Mapping[str, str],
    *,
    home: Path,
) -> dict[str, str]:
    """Build a minimal child environment with fixture-only credentials."""

    child = {name: inherited[name] for name in SAFE_INHERITED_NAMES if name in inherited}
    child.update(
        {
            "HOME": str(home),
            "PYTHONUNBUFFERED": "1",
            "RAG_ENVIRONMENT": "development",
            "RAG_API_URL": f"http://{API_HOST}:{API_PORT}",
            "RAG_API_KEY": FIXTURE_TOKEN,
            "RAG_QDRANT_API_KEY": "",
            "RAG_OPENAI_API_KEY": "",
            "RAG_VOYAGE_API_KEY": "",
            "RAG_DEMO_MODE": "true",
            "RAG_UI_PORT": str(UI_PORT),
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )
    return child


def build_commands(python_executable: Path) -> tuple[list[str], list[str]]:
    """Return fixed loopback commands for the fixture API and Streamlit UI."""

    python = str(python_executable)
    api_command = [
        python,
        "-m",
        "uvicorn",
        "tests.browser.fake_api:app",
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
        "--no-access-log",
    ]
    ui_command = [
        python,
        "-m",
        "streamlit",
        "run",
        "src/personal_rag/ui/app.py",
        f"--server.address={UI_HOST}",
        f"--server.port={UI_PORT}",
        "--server.headless=true",
    ]
    return api_command, ui_command


def wait_for_service(
    process: ProcessLike,
    *,
    host: str,
    port: int,
    path: str,
    label: str,
    timeout_seconds: float = STARTUP_TIMEOUT_SECONDS,
) -> None:
    """Wait for one local HTTP service or fail when its process exits."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise DemoError(f"{label} exited before it became ready.")
        connection = http.client.HTTPConnection(host, port, timeout=0.5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            if 200 <= response.status < 500:
                return
        except (ConnectionError, OSError, TimeoutError, http.client.HTTPException):
            pass
        finally:
            connection.close()
        time.sleep(0.1)
    raise DemoError(f"{label} did not become ready within {timeout_seconds:g} seconds.")


def stop_process(process: ProcessLike) -> None:
    """Stop one direct server process and escalate after a bounded grace period."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def run_demo() -> int:
    """Launch both demo services and keep them alive until interrupted."""

    repository_root = Path(__file__).resolve().parents[1]
    api_command, ui_command = build_commands(Path(sys.executable))
    processes: list[subprocess.Popen[bytes]] = []

    print("Starting Personal Library with deterministic sample data.")
    print("No provider key or external model call is used by this demo.")
    print("Demo changes are temporary and reset the next time you start it.")

    result = 0
    with tempfile.TemporaryDirectory(prefix="personal-library-demo-") as temporary_home:
        try:
            child_environment = build_child_environment(
                os.environ,
                home=Path(temporary_home),
            )
            # Both commands contain only fixed literals and the current Python executable.
            api_process = subprocess.Popen(  # nosec B603
                api_command,
                cwd=repository_root,
                env=child_environment,
                start_new_session=True,
            )
            processes.append(api_process)
            wait_for_service(
                api_process,
                host=API_HOST,
                port=API_PORT,
                path="/health/live",
                label="Demo API",
            )

            ui_process = subprocess.Popen(  # nosec B603
                ui_command,
                cwd=repository_root,
                env=child_environment,
                start_new_session=True,
            )
            processes.append(ui_process)
            wait_for_service(
                ui_process,
                host=UI_HOST,
                port=UI_PORT,
                path="/_stcore/health",
                label="Demo UI",
            )

            print(f"Open http://{UI_HOST}:{UI_PORT}")
            print("Press Ctrl+C to stop the demo and discard its temporary changes.")
            while True:
                for label, process in (("Demo API", api_process), ("Demo UI", ui_process)):
                    if process.poll() is not None:
                        raise DemoError(f"{label} stopped unexpectedly.")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping the demo and discarding temporary changes.")
        except DemoError as error:
            print(f"Demo stopped: {error}", file=sys.stderr)
            result = 1
        except OSError:
            print("Demo stopped: a local demo service could not be launched.", file=sys.stderr)
            result = 1
        finally:
            for process in reversed(processes):
                stop_process(process)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        print("This command does not accept arguments.", file=sys.stderr)
        return 2
    return run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
