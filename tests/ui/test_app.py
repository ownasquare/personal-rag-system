"""Mock-backed Streamlit AppTest coverage for the primary UI states."""

from __future__ import annotations

from typing import TYPE_CHECKING

from personal_rag.models import DependencyState, DocumentStatus, SystemStatus
from personal_rag.ui.client import ApiClientError

from .conftest import FakeRagClient, make_document, ready_status

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest

    from personal_rag.config import Settings


def _button(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(button for button in app.button if button.label == label)


def _text_input(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.text_input if widget.label == label)


def _text_area(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.text_area if widget.label == label)


def _visible_text(app: AppTest) -> str:
    values: list[str] = []
    for element_type in (
        app.markdown,
        app.caption,
        app.info,
        app.success,
        app.warning,
        app.error,
    ):
        values.extend(str(element.value) for element in element_type)
    return "\n".join(values)


def test_three_area_shell_and_empty_library_onboarding(app_test: AppTest) -> None:
    result = app_test.run()

    assert not result.exception
    assert [tab.label for tab in result.tabs] == ["Chat", "Library", "Settings & Status"]
    assert "Add your first document" in _visible_text(result)
    assert _button(result, "Add to library").disabled is True


def test_provider_missing_state_is_explicit_and_disables_chat(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.system_status = SystemStatus(
        status="needs_setup",
        collection="personal_knowledge",
        document_count=0,
        ready_document_count=0,
        chunk_count=0,
        queued_job_count=0,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
        chat_provider="openai",
        chat_model="gpt-4.1-mini",
        dependencies=[DependencyState(name="providers", status="not_configured")],
    )

    result = app_test.run()

    assert not result.exception
    assert "Provider setup required" in _visible_text(result)
    assert not any(area.label == "Ask about your library" for area in result.text_area)


def test_files_are_uploaded_only_after_explicit_multi_file_action(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = app_test.run()
    result.file_uploader[0].set_value(
        [
            ("alpha.md", b"alpha", "text/markdown"),
            ("beta.txt", b"beta", "text/plain"),
        ]
    )
    result.run()

    assert fake_client.upload_calls == []
    assert _button(result, "Add to library").disabled is False

    _button(result, "Add to library").click()
    result.run()

    assert fake_client.upload_calls == ["alpha.md", "beta.txt"]
    assert set(result.session_state["tracked_jobs"]) == {"upload-job-1", "upload-job-2"}
    assert "2 documents accepted" in _visible_text(result)


def test_library_search_and_filename_confirmed_delete(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [
        make_document("alpha-notes.md", document_id="doc-alpha"),
        make_document("beta-plan.md", document_id="doc-beta"),
    ]
    fake_client.system_status = ready_status(document_count=2)
    result = app_test.run()

    confirm = _text_input(result, 'Type "alpha-notes.md" to confirm deletion')
    delete_button = _button(result, "Delete alpha-notes.md")
    assert delete_button.disabled is True

    confirm.set_value("alpha-notes.md")
    result.run()
    _button(result, "Delete alpha-notes.md").click()
    result.run()

    assert fake_client.delete_calls == ["doc-alpha"]
    assert "delete-doc-alpha" in result.session_state["tracked_jobs"]

    _text_input(result, "Search documents").set_value("beta")
    result.run()
    expander_labels = [expander.label for expander in result.expander]
    assert any("beta-plan.md" in label for label in expander_labels)
    assert not any("alpha-notes.md" in label for label in expander_labels)


def test_deletion_failure_offers_retry_delete_not_reindex(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    failed = make_document(status=DocumentStatus.DELETION_FAILED).model_copy(
        update={
            "active_version": 1,
            "chunk_count": 4,
            "error_code": "vector_delete_failed",
        }
    )
    fake_client.documents = [failed]
    fake_client.system_status = ready_status(document_count=1)

    result = app_test.run()

    assert _button(result, "Reindex").disabled is True
    assert _button(result, "Retry deletion").disabled is True
    assert "Deletion is incomplete" in _visible_text(result)

    _text_input(result, 'Type "field-notes.md" to confirm deletion').set_value("field-notes.md")
    result.run()
    _button(result, "Retry deletion").click()
    result.run()

    assert fake_client.delete_calls == ["doc-1"]


def test_chat_history_top_k_and_backend_citation_cards(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    result = app_test.run()

    next(slider for slider in result.slider if slider.label == "Sources to retrieve").set_value(3)
    _text_area(result, "Ask about your library").set_value("What is the launch key?")
    result.run()
    _button(result, "Ask library").click()
    result.run()

    assert not result.exception
    assert fake_client.chat_calls[-1].top_k == 3
    assert len(result.session_state["chat_history"]) == 2
    assert _text_area(result, "Ask about your library").value == ""
    assert "The launch key is cobalt [S1]." in _visible_text(result)
    assert any(
        "S1" in expander.label and "field-notes.md" in expander.label
        for expander in result.expander
    )
    assert "The Atlas launch key is cobalt." in _visible_text(result)


def test_retryable_chat_error_preserves_question_and_existing_history(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.chat_error = ApiClientError(
        code="provider_unavailable",
        message="The answer provider is temporarily unavailable.",
        status_code=503,
        retryable=True,
        request_id="request-2",
    )
    app_test.session_state["chat_history"] = [
        {"role": "user", "content": "Earlier question", "citations": []},
        {"role": "assistant", "content": "Earlier answer", "citations": []},
    ]
    result = app_test.run()
    question = "Please retry this exact question"
    _text_area(result, "Ask about your library").set_value(question)
    result.run()
    _button(result, "Ask library").click()
    result.run()

    assert len(result.session_state["chat_history"]) == 2
    assert _text_area(result, "Ask about your library").value == question
    assert "temporarily unavailable" in _visible_text(result)
    assert "You can retry" in _visible_text(result)


def test_zero_history_limit_sends_no_prior_messages(
    app_test: AppTest, fake_client: FakeRagClient, ui_settings: Settings
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    app_test.session_state["_rag_settings"] = ui_settings.model_copy(
        update={"max_history_messages": 0}
    )
    app_test.session_state["chat_history"] = [
        {"role": "user", "content": "Prior question", "citations": []},
        {"role": "assistant", "content": "Prior answer", "citations": []},
    ]
    result = app_test.run()
    _text_area(result, "Ask about your library").set_value("Fresh question")
    result.run()
    _button(result, "Ask library").click()
    result.run()

    assert fake_client.chat_calls[-1].history == []


def test_document_text_is_not_rendered_as_unsafe_html_and_status_hides_secrets(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    hostile_name = "<script>window.bad=true</script>.md"
    fake_client.documents = [make_document(hostile_name)]
    fake_client.system_status = ready_status(document_count=1)
    result = app_test.run()

    assert not result.exception
    assert hostile_name in "\n".join(expander.label for expander in result.expander)
    assert "server-secret" not in _visible_text(result)
    assert "OPENAI_API_KEY" not in _visible_text(result)


def test_processing_jobs_refresh_from_server_truth(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    pending = make_document("pending.md", status=DocumentStatus.QUEUED)
    fake_client.documents = [pending]
    receipt = fake_client.upload_document("tracked.md", b"tracked", "text/markdown")
    app_test.session_state["tracked_jobs"] = {
        receipt.job.id: {
            "id": receipt.job.id,
            "document_name": receipt.document.display_name,
            "action": "Adding",
            "status": receipt.job.status.value,
            "stage": receipt.job.stage.value,
            "progress": receipt.job.progress,
        }
    }

    result = app_test.run()

    assert not result.exception
    assert "Adding tracked.md" in _visible_text(result)
    assert fake_client.get_job(receipt.job.id).status.value == "queued"
