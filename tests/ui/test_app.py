"""Mock-backed Streamlit AppTest coverage for the Phase 2 workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from personal_rag.config import Settings
from personal_rag.models import (
    ConversationTurnStatus,
    DependencyState,
    DocumentSort,
    DocumentStatus,
    JobKind,
    JobStage,
    JobStatus,
    SortOrder,
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


def _selectbox(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.selectbox if widget.label == label)


def _toggle(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(widget for widget in app.toggle if widget.label == label)


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
    assert "Add your first documents" in _visible_text(result)
    assert "Add files" in _visible_text(result)
    assert _button(result, "Add to library").disabled is True
    assert fake_client.list_conversations_calls == 0
    assert fake_client.health_live_calls == 0
    assert fake_client.health_ready_calls == 0


def test_demo_mode_is_clearly_labeled(
    app_test: AppTest, fake_client: FakeRagClient, ui_settings: Settings
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    app_test.session_state["_rag_settings"] = ui_settings.model_copy(update={"demo_mode": True})

    result = app_test.run()

    assert "Demo mode" in _visible_text(result)
    assert "Processing and answers are simulated" in _visible_text(result)
    assert "Changes reset" in _visible_text(result)


def test_provider_setup_blocks_upload_everywhere(
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
    assert "Finish setup" in _visible_text(result)
    assert not result.file_uploader
    assert _button(result, "Open system status")

    result = _navigate(result, "Library")
    assert "Finish setup" in _visible_text(result)
    assert not result.file_uploader
    assert not any(item.label == "Find a document" for item in result.text_input)


def test_ready_workspace_defaults_to_ask_and_hides_operator_controls(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)

    assert not result.exception
    assert "Ask your library" in _visible_text(result)
    navigation = next(widget for widget in result.segmented_control if widget.label == "Workspace")
    assert navigation.options == ["Ask", "Library", "Activity"]
    assert "New conversation" in _visible_text(result)
    assert "Sources to retrieve" not in _visible_text(result)
    assert any(expander.label == "Search options" for expander in result.expander)
    examples = next(expander for expander in result.expander if expander.label == "Try an example")
    assert examples.proto.expanded is False
    assert any(expander.label == "Saved conversations" for expander in result.expander)
    button_labels = [button.label for button in result.button]
    assert button_labels.index("Ask library") < button_labels.index("Summarize the main points")
    assert fake_client.health_live_calls == 0
    assert fake_client.health_ready_calls == 0


def test_suggestion_seeds_a_new_question_widget_after_the_primary_form(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)

    _button(result, "Summarize the main points").click()
    result = result.run()

    assert not result.exception
    assert result.session_state["question_draft_version"] == 1
    assert _text_area(result, "Your question").value == "Summarize the main points"
    button_labels = [button.label for button in result.button]
    assert button_labels.index("Ask library") < button_labels.index("Summarize the main points")


def test_saved_conversation_management_is_secondary_but_reachable(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation()]
    fake_client.turns = {"conversation-1": [make_turn()]}

    result = app_test.run()

    assert any(expander.label == "Saved conversations" for expander in result.expander)
    assert "Current conversation" in _visible_text(result)
    previous = _selectbox(result, "Previous conversations")
    assert previous.value == "conversation-1"
    assert previous.options == ["Atlas launch notes"]


def test_newest_answer_is_immediate_and_older_answers_are_collapsed(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.conversations = [make_conversation(turn_count=2)]
    fake_client.turns = {
        "conversation-1": [
            make_turn(turn_id="turn-old", question="Older question"),
            make_turn(turn_id="turn-new", question="Newest question"),
        ]
    }

    result = app_test.run()

    assert "Newest question" in _visible_text(result)
    assert "Older question" not in _visible_text(result)
    previous = _toggle(result, "Previous answers (1)")
    assert previous.value is False

    previous.set_value(True)
    result = result.run()
    assert "Older question" in _visible_text(result)


def test_conditional_navigation_avoids_hidden_system_health_calls(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)
    fake_client.get_status_calls = 0
    fake_client.health_live_calls = 0
    fake_client.health_ready_calls = 0

    result = _navigate(result, "Library")

    assert "Library" in _visible_text(result)
    assert fake_client.get_status_calls == 1
    assert fake_client.health_live_calls == 0
    assert fake_client.health_ready_calls == 0

    _button(result, "System status").click()
    result = result.run()

    assert "System status" in _visible_text(result)
    assert fake_client.health_live_calls == 1
    assert fake_client.health_ready_calls == 1

    result = _navigate(result, "Ask")
    assert "Ask your library" in _visible_text(result)


def test_system_status_can_return_to_the_already_selected_primary_section(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _ready_app(app_test, fake_client)

    _button(result, "System status").click()
    result = result.run()

    navigation = next(widget for widget in result.segmented_control if widget.label == "Workspace")
    assert navigation.value is None
    assert "System status" in _visible_text(result)

    result = _navigate(result, "Ask")
    assert "Ask your library" in _visible_text(result)


def test_multi_upload_reruns_into_immediately_visible_activity(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _navigate(app_test.run(), "Library")
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
    result = _navigate(app_test.run(), "Library")
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
    assert any(expander.label == "View source (1)" for expander in result.expander)
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
    result = _navigate(app_test.run(), "Library")

    _text_input(result, "Find a document").set_value("beta")
    result = result.run()
    _button(result, "Search").click()
    result = result.run()
    document_choices = _selectbox(result, "Document")
    assert any("beta-plan.md" in option for option in document_choices.options)
    assert not any("alpha-notes.md" in option for option in document_choices.options)

    _button(result, "Reprocess for search").click()
    result = result.run()
    assert fake_client.reindex_calls == ["doc-beta"]
    assert "Reprocessing beta-plan.md" in _visible_text(result)

    result = _navigate(result, "Library")
    _text_input(result, 'Type "beta-plan.md" to confirm removal').set_value("beta-plan.md")
    result = result.run()
    _button(result, "Remove beta-plan.md permanently").click()
    result = result.run()
    assert fake_client.delete_calls == ["doc-beta"]
    assert "Removing beta-plan.md" in _visible_text(result)


def test_nonempty_documents_is_library_first_and_requests_one_page(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [
        make_document("alpha.md", document_id="doc-alpha"),
        make_document("beta.md", document_id="doc-beta"),
    ]
    fake_client.system_status = ready_status(document_count=2)
    result = app_test.run()
    ask_reads = fake_client.list_all_documents_calls

    result = _navigate(result, "Library")

    assert not result.exception
    assert str(result.header[0].value) == "Library"
    expander_labels = [expander.label for expander in result.expander]
    assert expander_labels.index("Add documents") < expander_labels.index("Filter & sort")
    manage = next(expander for expander in result.expander if expander.label == "Manage document")
    assert manage.proto.expanded is False
    assert fake_client.list_all_documents_calls == ask_reads
    assert len(fake_client.document_page_requests) == 1
    assert fake_client.document_page_requests[0]["limit"] == 10
    assert len([item for item in result.text_input if item.label.startswith('Type "')]) == 1


def test_ready_document_can_open_a_scoped_question(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document("launch.md")]
    fake_client.system_status = ready_status(document_count=1)
    result = _navigate(app_test.run(), "Library")

    _button(result, "Ask about this document").click()
    result = result.run()

    assert "Ask your library" in _visible_text(result)
    assert "Searching only: launch.md" in _visible_text(result)
    scope = next(item for item in result.multiselect if item.label == "Documents")
    assert scope.value == ["doc-1"]

    _text_area(result, "Your question").set_value("What is the launch plan?")
    result = result.run()
    _button(result, "Ask library").click()
    result = result.run()

    assert fake_client.turn_calls[-1].document_ids == ["doc-1"]
    assert "Searching only: launch.md" in _visible_text(result)

    _button(result, "New conversation").click()
    result = result.run()
    assert result.session_state["question_document_scope"] == []
    assert "Searching only:" not in _visible_text(result)


def test_document_filters_are_applied_once_with_server_contract(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [
        make_document("alpha-notes.md", document_id="doc-alpha", status=DocumentStatus.FAILED),
        make_document(
            "beta-notes.pdf",
            document_id="doc-beta",
            status=DocumentStatus.DELETION_FAILED,
        ),
        make_document("ready-notes.md", document_id="doc-ready"),
    ]
    fake_client.system_status = ready_status(document_count=3)
    result = _navigate(app_test.run(), "Library")
    initial_requests = len(fake_client.document_page_requests)

    _text_input(result, "Find a document").set_value("notes")
    _selectbox(result, "Status").set_value("Needs attention")
    _selectbox(result, "Sort by").set_value("Name A-Z")
    assert result.session_state["document_query"] == ""
    assert result.session_state["document_status_filter"] == "All"
    assert result.session_state["document_sort"] == "Recently added"
    _button(result, "Search").click()
    result = result.run()

    assert len(fake_client.document_page_requests) == initial_requests + 1
    request = fake_client.document_page_requests[-1]
    assert request["query"] == "notes"
    assert request["statuses"] == (
        DocumentStatus.FAILED,
        DocumentStatus.DELETION_FAILED,
    )
    assert request["sort"] is DocumentSort.NAME
    assert request["order"] is SortOrder.ASC
    assert request["offset"] == 0
    choices = _selectbox(result, "Document")
    assert choices.options == [
        "alpha-notes.md — Needs attention",
        "beta-notes.pdf — Needs attention",
    ]


def test_document_pagination_resets_when_filters_are_applied(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [
        make_document(f"file-{index:03d}.md", document_id=f"doc-{index:03d}") for index in range(21)
    ]
    fake_client.system_status = ready_status(document_count=21)
    result = _navigate(app_test.run(), "Library")

    _button(result, "Next page").click()
    result = result.run()
    assert fake_client.document_page_requests[-1]["offset"] == 10
    choices = _selectbox(result, "Document")
    assert result.session_state["selected_document_id"] in {
        document.id
        for document in fake_client.documents
        if any(document.display_name in option for option in choices.options)
    }

    _text_input(result, "Find a document").set_value("file-000")
    result = result.run()
    _button(result, "Search").click()
    result = result.run()

    assert fake_client.document_page_requests[-1]["offset"] == 0
    assert result.session_state["selected_document_id"] == "doc-000"


def test_inconsistent_empty_boundary_page_recovers_without_crashing(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [
        make_document(f"file-{index:03d}.md", document_id=f"doc-{index:03d}") for index in range(11)
    ]
    fake_client.system_status = ready_status(document_count=11)
    result = _navigate(app_test.run(), "Library")

    _button(result, "Next page").click()
    result = result.run()
    assert fake_client.document_page_requests[-1]["offset"] == 10
    fake_client.inconsistent_document_page_once = True
    result = result.run()

    assert not result.exception
    assert [request["offset"] for request in fake_client.document_page_requests[-2:]] == [10, 0]
    assert result.session_state["selected_document_id"] == "doc-010"


def test_deletion_failure_offers_clear_retry_path(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    failed = make_document(status=DocumentStatus.DELETION_FAILED).model_copy(
        update={"error_code": "vector_delete_failed"}
    )
    fake_client.documents = [failed]
    fake_client.system_status = ready_status(document_count=1)
    result = _navigate(app_test.run(), "Library")

    assert "Removal is incomplete" in _visible_text(result)
    assert _button(result, "Reprocess for search").disabled is True
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
    assert "Reprocessing pending.md" in _visible_text(result)
    assert "Preparing search" in _visible_text(result)
    assert "Embedding" not in _visible_text(result)
    assert fake_client.list_jobs_calls == 1


def test_completed_activity_hands_off_to_ask(app_test: AppTest, fake_client: FakeRagClient) -> None:
    fake_client.documents = [make_document("ready.md")]
    fake_client.system_status = ready_status(document_count=1)
    fake_client.jobs = {
        "complete": make_job(
            job_id="complete",
            status=JobStatus.SUCCEEDED,
            stage=JobStage.COMPLETE,
        )
    }

    result = _navigate(app_test.run(), "Activity")

    assert _button(result, "Ask your documents")
    _button(result, "Ask your documents").click()
    result = result.run()
    assert "Ask your library" in _visible_text(result)


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
    assert headings == ["In progress", "Needs attention"]
    completed = next(expander for expander in result.expander if expander.label == "Completed (1)")
    assert completed.proto.expanded is False
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

    assert "Setup required" in _visible_text(result)
    assert not any(area.label == "Your question" for area in result.text_area)


def test_api_outage_explains_that_saved_work_is_untouched(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.status_error = ApiClientError(
        code="api_unavailable",
        message="Personal Library is not responding.",
        retryable=True,
    )

    result = app_test.run()

    assert "Personal Library is not responding" in _visible_text(result)
    assert "saved work is safe" in _visible_text(result)
    assert _button(result, "Open system status")
    assert not result.exception


def test_hostile_document_text_is_never_inserted_into_unsafe_html(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    hostile_name = "<script>$x$ ~~window.bad=true~~</script>.md"
    fake_client.documents = [make_document(hostile_name)]
    fake_client.system_status = ready_status(document_count=1)

    result = _navigate(app_test.run(), "Library")

    document_choices = _selectbox(result, "Document")
    assert any(hostile_name in option for option in document_choices.options)
    detail_heading = next(heading for heading in result.subheader if "script" in str(heading.value))
    assert r"\$x\$" in str(detail_heading.value)
    assert r"\~\~window.bad=true\~\~" in str(detail_heading.value)
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
    assert "New conversation" in _visible_text(result)


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
    result = _navigate(app_test.run(), "Library")
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
