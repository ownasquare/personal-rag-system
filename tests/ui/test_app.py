"""Mock-backed Streamlit AppTest coverage for the Phase 2 workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from personal_rag.models import (
    ConversationTurnStatus,
    DependencyState,
    DocumentStatus,
    JobKind,
    JobStage,
    JobStatus,
    SystemStatus,
)
from personal_rag.ui.client import ApiClientError

from .conftest import (
    FakeRagClient,
    make_conversation,
    make_document,
    make_job,
    make_turn,
    ready_status,
)

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest


def _button(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(button for button in app.button if button.label == label)


def _text_input(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.text_input if widget.label == label)


def _text_area(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.text_area if widget.label == label)


def _navigate(app: AppTest, section: str) -> AppTest:
    navigation = next(widget for widget in app.segmented_control if widget.label == "Workspace")
    navigation.set_value(section)
    return app.run()


def _visible_text(app: AppTest) -> str:
    values: list[str] = []
    for element_type in (
        app.markdown,
        app.caption,
        app.info,
        app.success,
        app.warning,
        app.error,
        app.header,
        app.subheader,
    ):
        values.extend(str(element.value) for element in element_type)
    return "\n".join(values)


def _ready_app(app_test: AppTest, fake_client: FakeRagClient) -> AppTest:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    return app_test.run()


def test_empty_library_opens_a_complete_onboarding_flow(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = app_test.run()

    assert not result.exception
    assert "Bring in something you already use" in _visible_text(result)
    assert "Add a PDF, document, Markdown file" in _visible_text(result)
    assert _button(result, "Add to library").disabled is True
    assert fake_client.list_conversations_calls == 0
    assert fake_client.health_live_calls == 0
    assert fake_client.health_ready_calls == 0


def test_ready_workspace_defaults_to_ask_and_hides_operator_controls(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)

    assert not result.exception
    assert "Ask your library" in _visible_text(result)
    assert "A fresh conversation" in _visible_text(result)
    assert "Sources to retrieve" not in _visible_text(result)
    assert any(expander.label == "Search options" for expander in result.expander)
    assert fake_client.health_live_calls == 0
    assert fake_client.health_ready_calls == 0


def test_conditional_navigation_avoids_hidden_system_health_calls(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)
    fake_client.get_status_calls = 0
    fake_client.health_live_calls = 0
    fake_client.health_ready_calls = 0

    result = _navigate(result, "Documents")

    assert "Your library" in _visible_text(result)
    assert fake_client.get_status_calls == 0
    assert fake_client.health_live_calls == 0
    assert fake_client.health_ready_calls == 0

    result = _navigate(result, "System")

    assert "System details" in _visible_text(result)
    assert fake_client.health_live_calls == 1
    assert fake_client.health_ready_calls == 1


def test_multi_upload_reruns_into_immediately_visible_activity(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _navigate(app_test.run(), "Documents")
    result.file_uploader[0].set_value(
        [
            ("alpha.md", b"alpha", "text/markdown"),
            ("beta.txt", b"beta", "text/plain"),
        ]
    )
    result = result.run()

    assert fake_client.upload_calls == []
    assert _button(result, "Add to library").disabled is False
    _button(result, "Add to library").click()
    result = result.run()

    assert fake_client.upload_calls == ["alpha.md", "beta.txt"]
    assert "2 documents accepted" in _visible_text(result)
    assert "Adding alpha.md" in _visible_text(result)
    assert "Adding beta.txt" in _visible_text(result)
    assert fake_client.list_jobs_calls > 0


def test_partial_upload_keeps_success_and_failure_feedback_together(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.upload_errors["broken.pdf"] = ApiClientError(
        code="invalid_document",
        message="This PDF could not be read.",
        status_code=422,
    )
    result = _navigate(app_test.run(), "Documents")
    result.file_uploader[0].set_value(
        [
            ("good.md", b"good", "text/markdown"),
            ("broken.pdf", b"broken", "application/pdf"),
        ]
    )
    result = result.run()
    _button(result, "Add to library").click()
    result = result.run()

    assert "1 document accepted" in _visible_text(result)
    assert "broken.pdf: This PDF could not be read." in _visible_text(result)
    assert "Adding good.md" in _visible_text(result)


def test_saved_conversation_is_restored_and_new_turn_uses_durable_api(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation()]
    fake_client.turns = {"conversation-1": [make_turn()]}

    result = app_test.run()

    assert "The launch key is cobalt [S1]." in _visible_text(result)
    assert any(expander.label == "Source · 1" for expander in result.expander)
    assert "Relevance" not in _visible_text(result)

    _text_area(result, "Your question").set_value("What should I do next?")
    result = result.run()
    _button(result, "Ask library").click()
    result = result.run()

    assert len(fake_client.turn_calls) == 1
    assert fake_client.turn_calls[0].message == "What should I do next?"
    assert len(fake_client.turns["conversation-1"]) == 2
    assert result.session_state["pending_client_turn_id"] is None


def test_new_conversation_creates_server_truth_on_first_question(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation()]
    fake_client.turns = {"conversation-1": [make_turn()]}
    result = app_test.run()

    _button(result, "New conversation").click()
    result = result.run()
    assert "A fresh conversation" in _visible_text(result)
    _text_area(result, "Your question").set_value("Give me a clean summary")
    result = result.run()
    _button(result, "Ask library").click()
    result = result.run()

    assert len(fake_client.conversations) == 2
    selected_id = result.session_state["selected_conversation_id"]
    assert isinstance(selected_id, str)
    assert fake_client.turns[selected_id][0].question == "Give me a clean summary"


def test_retryable_turn_error_preserves_question_and_client_turn_id(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)
    fake_client.chat_error = ApiClientError(
        code="provider_unavailable",
        message="The answer service is temporarily unavailable.",
        status_code=503,
        retryable=True,
        request_id="request-2",
    )
    question = "Please retry this exact question"
    _text_area(result, "Your question").set_value(question)
    result = result.run()
    _button(result, "Ask library").click()
    result = result.run()

    first_turn_id = fake_client.turn_calls[-1].client_turn_id
    assert _text_area(result, "Your question").value == question
    assert result.session_state["pending_client_turn_id"] == first_turn_id
    assert "temporarily unavailable" in _visible_text(result)

    fake_client.chat_error = None
    _button(result, "Ask library").click()
    result = result.run()

    assert fake_client.turn_calls[-1].client_turn_id == first_turn_id
    assert result.session_state["pending_client_turn_id"] is None


def test_editing_a_retry_rotates_the_client_turn_id(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)
    fake_client.chat_error = ApiClientError(
        code="provider_unavailable",
        message="The answer service is temporarily unavailable.",
        status_code=503,
        retryable=True,
    )
    _text_area(result, "Your question").set_value("First wording")
    result = result.run()
    _button(result, "Ask library").click()
    result = result.run()
    first_client_turn_id = fake_client.turn_calls[-1].client_turn_id

    fake_client.chat_error = None
    _text_area(result, "Your question").set_value("Edited wording")
    result = result.run()
    _button(result, "Ask library").click()
    result = result.run()

    assert fake_client.turn_calls[-1].client_turn_id != first_client_turn_id
    assert fake_client.turn_calls[-1].message == "Edited wording"
    assert "idempotency" not in _visible_text(result).casefold()


def test_persisted_retryable_turn_can_be_retried_after_a_fresh_session(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation()]
    failed = make_turn(status=ConversationTurnStatus.FAILED, answer=None)
    fake_client.turns = {"conversation-1": [failed]}

    result = app_test.run()
    _button(result, "Try again").click()
    result = result.run()

    assert fake_client.turn_calls[-1].client_turn_id == failed.client_turn_id
    assert fake_client.turn_calls[-1].message == failed.question
    assert fake_client.turns["conversation-1"][0].status is ConversationTurnStatus.COMPLETED
    assert "The launch key is cobalt" in _visible_text(result)


def test_persisted_pending_turn_can_be_checked_after_a_fresh_session(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation()]
    pending = make_turn(status=ConversationTurnStatus.PENDING, answer=None)
    fake_client.turns = {"conversation-1": [pending]}

    result = app_test.run()
    _button(result, "Check or retry").click()
    result = result.run()

    assert fake_client.turn_calls[-1].client_turn_id == pending.client_turn_id
    assert fake_client.turns["conversation-1"][0].status is ConversationTurnStatus.COMPLETED


def test_no_answer_has_a_distinct_recovery_state(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.no_answer = True
    result = _ready_app(app_test, fake_client)
    _text_area(result, "Your question").set_value("What is not in these files?")
    result = result.run()
    _button(result, "Ask library").click()
    result = result.run()

    assert "There was not enough support" in _visible_text(result)
    assert fake_client.turns["conversation-1"][0].no_answer is True


def test_document_search_refresh_and_confirmed_removal(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [
        make_document("alpha-notes.md", document_id="doc-alpha"),
        make_document("beta-plan.md", document_id="doc-beta"),
    ]
    fake_client.system_status = ready_status(document_count=2)
    result = _navigate(app_test.run(), "Documents")

    _text_input(result, "Find a document").set_value("beta")
    result = result.run()
    assert any("beta-plan.md" in expander.label for expander in result.expander)
    assert not any("alpha-notes.md" in expander.label for expander in result.expander)

    _button(result, "Refresh document").click()
    result = result.run()
    assert fake_client.reindex_calls == ["doc-beta"]
    assert "Refreshing beta-plan.md" in _visible_text(result)

    result = _navigate(result, "Documents")
    _text_input(result, 'Type "beta-plan.md" to confirm removal').set_value("beta-plan.md")
    result = result.run()
    _button(result, "Remove beta-plan.md permanently").click()
    result = result.run()
    assert fake_client.delete_calls == ["doc-beta"]
    assert "Removing beta-plan.md" in _visible_text(result)


def test_deletion_failure_offers_clear_retry_path(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    failed = make_document(status=DocumentStatus.DELETION_FAILED).model_copy(
        update={"error_code": "vector_delete_failed"}
    )
    fake_client.documents = [failed]
    fake_client.system_status = ready_status(document_count=1)
    result = _navigate(app_test.run(), "Documents")

    assert "Removal is incomplete" in _visible_text(result)
    assert _button(result, "Refresh document").disabled is True
    assert _button(result, "Retry removal of field-notes.md").disabled is True

    _text_input(result, 'Type "field-notes.md" to confirm removal').set_value("field-notes.md")
    result = result.run()
    _button(result, "Retry removal of field-notes.md").click()
    result = result.run()

    assert fake_client.delete_calls == ["doc-1"]


def test_activity_restores_jobs_after_a_fresh_ui_session(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document("pending.md", status=DocumentStatus.QUEUED)]
    fake_client.jobs = {
        "job-active": make_job(job_id="job-active", stage=JobStage.EMBEDDING),
        "job-done": make_job(
            job_id="job-done",
            status=JobStatus.SUCCEEDED,
            stage=JobStage.COMPLETE,
            kind=JobKind.REINDEX,
        ),
    }

    result = _navigate(app_test.run(), "Activity")

    assert not result.exception
    assert "In progress" in _visible_text(result)
    assert "Adding pending.md" in _visible_text(result)
    assert "Refreshing pending.md" in _visible_text(result)
    assert fake_client.list_jobs_calls == 1


def test_activity_separates_active_failed_and_completed_work(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document("notes.md")]
    fake_client.jobs = {
        "active": make_job(job_id="active"),
        "failed": make_job(
            job_id="failed",
            status=JobStatus.FAILED,
            stage=JobStage.FAILED,
        ),
        "complete": make_job(
            job_id="complete",
            status=JobStatus.SUCCEEDED,
            stage=JobStage.COMPLETE,
        ),
    }

    result = _navigate(app_test.run(), "Activity")

    headings = [str(element.value) for element in result.subheader]
    assert headings == ["In progress", "Needs attention", "Completed"]
    assert fake_client.list_jobs_calls == 1


def test_activity_escapes_markdown_control_characters_in_document_names(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    hostile_name = "**not a heading** [click](https://invalid.example).md"
    fake_client.documents = [make_document(hostile_name)]
    fake_client.jobs = {"active": make_job(job_id="active")}

    result = _navigate(app_test.run(), "Activity")
    activity_markdown = "\n".join(str(element.value) for element in result.markdown)

    assert r"\*\*not a heading\*\*" in activity_markdown
    assert r"\[click\]\(https://invalid.example\)" in activity_markdown
    assert not result.exception


def test_provider_missing_state_explains_setup_without_a_question_form(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = SystemStatus(
        status="needs_setup",
        collection="personal_knowledge",
        document_count=1,
        ready_document_count=1,
        chunk_count=4,
        queued_job_count=0,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
        chat_provider="openai",
        chat_model="gpt-4.1-mini",
        dependencies=[DependencyState(name="providers", status="not_configured")],
    )

    result = app_test.run()

    assert "One setup step remains" in _visible_text(result)
    assert not any(area.label == "Your question" for area in result.text_area)


def test_api_outage_explains_that_saved_work_is_untouched(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.status_error = ApiClientError(
        code="api_unavailable",
        message="The knowledge service is unavailable.",
        retryable=True,
    )

    result = app_test.run()

    assert "knowledge service is unavailable" in _visible_text(result)
    assert "saved work is untouched" in _visible_text(result)
    assert not result.exception


def test_hostile_document_text_is_never_inserted_into_unsafe_html(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    hostile_name = "<script>window.bad=true</script>.md"
    fake_client.documents = [make_document(hostile_name)]
    fake_client.system_status = ready_status(document_count=1)

    result = _navigate(app_test.run(), "Documents")

    assert hostile_name in "\n".join(expander.label for expander in result.expander)
    assert "server-secret" not in _visible_text(result)
    assert "OPENAI_API_KEY" not in _visible_text(result)
    assert not result.exception


def test_saved_conversation_can_be_hard_deleted_without_touching_documents(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation()]
    fake_client.turns = {"conversation-1": [make_turn(status=ConversationTurnStatus.COMPLETED)]}
    result = app_test.run()

    _button(result, "Delete…").click()
    result = result.run()
    assert "Delete this conversation" in _visible_text(result)
    _button(result, "Delete permanently").click()
    result = result.run()

    assert fake_client.conversations == []
    assert len(fake_client.documents) == 1
    assert "A fresh conversation" in _visible_text(result)


def test_conversation_picker_loads_beyond_the_first_api_page(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [
        make_conversation(
            conversation_id=f"conversation-{index:03d}",
            title=f"Conversation {index:03d}",
            turn_count=0,
        )
        for index in range(101)
    ]

    result = app_test.run()
    picker = next(item for item in result.selectbox if item.label == "Previous conversations")

    assert len(picker.options) == 101
    assert "Conversation 100" in picker.options
    assert fake_client.list_conversations_calls == 2


def test_long_conversation_shows_the_latest_turn_window(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation(turn_count=101)]
    fake_client.turns = {
        "conversation-1": [
            make_turn(
                turn_id=f"turn-{index:03d}",
                client_turn_id=f"client-turn-{index:03d}",
                question=f"Question {index:03d}",
            )
            for index in range(101)
        ]
    }

    result = app_test.run()
    visible = _visible_text(result)

    assert "Question 100" in visible
    assert "Question 000" not in visible
    assert "100 most recent of 101 saved questions" in visible


def test_clean_upload_clears_an_older_batch_error(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    failure = ApiClientError(
        code="bad_upload",
        message="That file could not be added.",
        status_code=422,
        retryable=False,
    )
    fake_client.upload_errors = {"bad.md": failure}
    result = _navigate(app_test.run(), "Documents")
    result.file_uploader[0].set_value([("bad.md", b"bad", "text/markdown")])
    result = result.run()
    _button(result, "Add to library").click()
    result = result.run()
    assert "could not be added" in _visible_text(result)

    fake_client.upload_errors = {}
    result.file_uploader[0].set_value([("good.md", b"good", "text/markdown")])
    result = result.run()
    _button(result, "Add to library").click()
    result = result.run()

    assert "1 document accepted" in _visible_text(result)
    assert "could not be added" not in _visible_text(result)
