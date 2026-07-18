"""Calm Streamlit workspace for the private Personal Library."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol, cast
from uuid import uuid4

import streamlit as st

from personal_rag.config import Settings, get_settings
from personal_rag.models import (
    Citation,
    ConversationList,
    ConversationSummary,
    ConversationTurn,
    ConversationTurnCreate,
    ConversationTurnList,
    ConversationTurnStatus,
    DocumentPublic,
    DocumentStatus,
    JobList,
    JobRecord,
    JobStatus,
    SystemStatus,
    UploadReceipt,
)
from personal_rag.ui.client import ApiClientError, HealthCheck, RagApiClient
from personal_rag.ui.presentation import (
    HEADER_HTML,
    ONBOARDING_STEPS_HTML,
    STATIC_STYLES,
    document_status_label,
    format_bytes,
    job_action_label,
    job_status_label,
)

SUPPORTED_FILE_TYPES = ["pdf", "docx", "md", "txt"]
WORKSPACE_SECTIONS = ("Ask", "Documents", "Activity", "System")
TERMINAL_JOB_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED}
MAX_CONVERSATIONS = 2_000
TURN_WINDOW = 100
_MARKDOWN_CONTROL = re.compile(r"([\\`*_{}\[\]()#+!|>])")


class UiClient(Protocol):
    """The narrow API surface consumed by the UI and implemented by test fakes."""

    def health_live(self) -> HealthCheck: ...

    def health_ready(self) -> HealthCheck: ...

    def get_status(self) -> SystemStatus: ...

    def list_all_documents(self, *, max_documents: int = 2000) -> list[DocumentPublic]: ...

    def upload_document(
        self, filename: str, content: bytes, content_type: str
    ) -> UploadReceipt: ...

    def get_job(self, job_id: str) -> JobRecord: ...

    def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: JobStatus | None = None,
        document_id: str | None = None,
    ) -> JobList: ...

    def reindex_document(self, document_id: str) -> JobRecord: ...

    def delete_document(self, document_id: str) -> JobRecord: ...

    def create_conversation(self, title: str | None = None) -> ConversationSummary: ...

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> ConversationList: ...

    def delete_conversation(self, conversation_id: str) -> None: ...

    def list_conversation_turns(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ConversationTurnList: ...

    def create_conversation_turn(
        self,
        conversation_id: str,
        turn: ConversationTurnCreate,
    ) -> ConversationTurn: ...


@st.cache_resource(show_spinner=False)
def _cached_client() -> RagApiClient:
    return RagApiClient.from_settings(get_settings())


def _resolve_settings() -> Settings:
    injected = st.session_state.get("_rag_settings")
    if isinstance(injected, Settings):
        return injected
    return get_settings()


def _resolve_client() -> UiClient:
    injected = st.session_state.get("_rag_client")
    if injected is not None:
        return cast("UiClient", injected)
    return _cached_client()


def _initialize_state() -> None:
    st.session_state.setdefault("tracked_jobs", {})
    st.session_state.setdefault("selected_conversation_id", None)
    st.session_state.setdefault("starting_new_conversation", False)
    st.session_state.setdefault("confirm_delete_conversation", False)
    st.session_state.setdefault("question_draft_version", 0)
    st.session_state.setdefault("pending_client_turn_id", None)
    st.session_state.setdefault("pending_turn_payload", None)
    st.session_state.setdefault("turn_error", None)
    st.session_state.setdefault("activity_notice", None)
    st.session_state.setdefault("activity_errors", [])


def _apply_requested_section() -> None:
    requested = st.session_state.pop("_next_section", None)
    if requested in WORKSPACE_SECTIONS:
        st.session_state["workspace_section"] = requested


def _request_section(section: str) -> None:
    if section not in WORKSPACE_SECTIONS:
        raise ValueError(f"Unknown workspace section: {section}")
    st.session_state["_next_section"] = section
    st.rerun()


def _safe_status(client: UiClient) -> tuple[SystemStatus | None, ApiClientError | None]:
    try:
        return client.get_status(), None
    except ApiClientError as exc:
        return None, exc


def _safe_documents(
    client: UiClient,
) -> tuple[list[DocumentPublic], ApiClientError | None]:
    try:
        return client.list_all_documents(), None
    except ApiClientError as exc:
        return [], exc


def _safe_conversations(
    client: UiClient,
) -> tuple[ConversationList | None, ApiClientError | None]:
    try:
        page_size = 100
        first = client.list_conversations(limit=page_size, offset=0)
        items = list(first.items)
        while len(items) < min(first.total, MAX_CONVERSATIONS):
            page = client.list_conversations(
                limit=min(page_size, MAX_CONVERSATIONS - len(items)),
                offset=len(items),
            )
            if not page.items:
                break
            items.extend(page.items)
        return first.model_copy(update={"items": items}), None
    except ApiClientError as exc:
        return None, exc


def _safe_turns(
    client: UiClient,
    conversation_id: str,
) -> tuple[ConversationTurnList | None, ApiClientError | None]:
    try:
        first = client.list_conversation_turns(conversation_id, limit=TURN_WINDOW, offset=0)
        if first.total <= TURN_WINDOW:
            return first, None
        return (
            client.list_conversation_turns(
                conversation_id,
                limit=TURN_WINDOW,
                offset=first.total - TURN_WINDOW,
            ),
            None,
        )
    except ApiClientError as exc:
        return None, exc


def _safe_jobs(client: UiClient) -> tuple[JobList | None, ApiClientError | None]:
    try:
        return client.list_jobs(limit=100), None
    except ApiClientError as exc:
        return None, exc


def _providers_configured(status: SystemStatus | None) -> bool:
    if status is None or status.status == "needs_setup":
        return False
    return not any(dependency.status == "not_configured" for dependency in status.dependencies)


def _escape_markdown(value: str) -> str:
    return _MARKDOWN_CONTROL.sub(r"\\\1", value)


def _render_header() -> None:
    # These are trusted static fragments. Document-derived values never enter unsafe HTML.
    st.markdown(HEADER_HTML, unsafe_allow_html=True)


def _render_navigation() -> str:
    section = st.segmented_control(
        "Workspace",
        options=WORKSPACE_SECTIONS,
        default=None if "workspace_section" in st.session_state else "Ask",
        selection_mode="single",
        label_visibility="collapsed",
        key="workspace_section",
        width="stretch",
    )
    return section if isinstance(section, str) else "Ask"


def _track_job(job: JobRecord, *, document_name: str) -> None:
    tracked_jobs = cast("dict[str, dict[str, object]]", st.session_state["tracked_jobs"])
    tracked_jobs[job.id] = {
        "document_id": job.document_id,
        "document_name": document_name,
        "kind": job.kind.value,
        "status": job.status.value,
    }


def _render_upload(
    client: UiClient,
    settings: Settings,
    *,
    key_prefix: str,
    heading: str | None = None,
) -> None:
    if heading:
        st.subheader(heading, anchor=False)
    st.caption(
        "PDF, DOCX, Markdown, and text files are supported. Nothing is added until you confirm."
    )
    uploads = st.file_uploader(
        "Choose documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help=f"Each file can be up to {format_bytes(settings.upload_max_bytes)}.",
        key=f"{key_prefix}-files",
    )
    add_clicked = st.button(
        "Add to library",
        type="primary",
        width="stretch",
        disabled=not uploads,
        key=f"{key_prefix}-add",
    )
    if not add_clicked:
        return

    accepted = 0
    duplicates = 0
    failures: list[str] = []
    progress = st.progress(0.0, text="Preparing your documents…")
    for index, uploaded in enumerate(uploads, start=1):
        if uploaded.size > settings.upload_max_bytes:
            failures.append(f"{uploaded.name}: this file is larger than the allowed limit")
            progress.progress(index / len(uploads), text=f"Checked {index} of {len(uploads)}")
            continue
        try:
            receipt = client.upload_document(
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type or "application/octet-stream",
            )
        except ApiClientError as exc:
            failures.append(f"{uploaded.name}: {exc.message}")
        else:
            accepted += 1
            duplicates += int(receipt.duplicate)
            _track_job(receipt.job, document_name=receipt.document.display_name)
        progress.progress(index / len(uploads), text=f"Submitted {index} of {len(uploads)}")

    if accepted:
        noun = "document" if accepted == 1 else "documents"
        duplicate_note = f" {duplicates} were already in your library." if duplicates else ""
        st.session_state["activity_notice"] = (
            f"{accepted} {noun} accepted. Processing is saved and continues in the background."
            f"{duplicate_note}"
        )
    st.session_state["activity_errors"] = failures
    if accepted:
        _request_section("Activity")
    for failure in failures:
        st.error(failure)


def _render_onboarding(client: UiClient, settings: Settings) -> None:
    st.markdown('<div class="section-kicker">Start here</div>', unsafe_allow_html=True)
    st.header("Bring in something you already use", anchor=False)
    st.write("A project brief, meeting notes, a handbook, or a research PDF is enough to begin.")
    # This is a trusted static fragment; file names are rendered only through Streamlit widgets.
    st.markdown(ONBOARDING_STEPS_HTML, unsafe_allow_html=True)
    with st.container(border=True):
        _render_upload(client, settings, key_prefix="onboarding")


def _choose_conversation(
    client: UiClient,
    conversations: list[ConversationSummary],
) -> str | None:
    known_ids = [conversation.id for conversation in conversations]
    selected = st.session_state.get("selected_conversation_id")
    starting_new = bool(st.session_state.get("starting_new_conversation"))

    if not conversations:
        starting_new = True
        selected = None
        st.session_state["starting_new_conversation"] = True
        st.session_state["selected_conversation_id"] = None
    elif not starting_new and selected not in known_ids:
        selected = conversations[0].id
        st.session_state["selected_conversation_id"] = selected

    new_column, previous_column = st.columns([1, 2], vertical_alignment="bottom")
    with new_column:
        if st.button("New conversation", width="stretch", type="primary"):
            st.session_state["selected_conversation_id"] = None
            st.session_state["starting_new_conversation"] = True
            st.session_state["confirm_delete_conversation"] = False
            st.session_state["pending_client_turn_id"] = None
            st.session_state["pending_turn_payload"] = None
            st.session_state["turn_error"] = None
            st.session_state.pop("conversation_picker", None)
            st.rerun()
    with previous_column:
        if conversations:
            selected_index = None
            if not starting_new and selected in known_ids:
                selected_index = known_ids.index(cast("str", selected))
            picked = st.selectbox(
                "Previous conversations",
                options=known_ids,
                index=selected_index,
                format_func=lambda item_id: next(
                    item.title for item in conversations if item.id == item_id
                ),
                placeholder="Choose a saved conversation",
                key="conversation_picker",
            )
            if picked is not None and (starting_new or picked != selected):
                st.session_state["selected_conversation_id"] = picked
                st.session_state["starting_new_conversation"] = False
                st.session_state["confirm_delete_conversation"] = False
                st.session_state["pending_client_turn_id"] = None
                st.session_state["pending_turn_payload"] = None
                st.session_state["turn_error"] = None
                st.rerun()

    selected_value = st.session_state.get("selected_conversation_id")
    if not isinstance(selected_value, str) or st.session_state.get("starting_new_conversation"):
        return None

    selected_summary = next(
        (conversation for conversation in conversations if conversation.id == selected_value),
        None,
    )
    if selected_summary is None:
        return None

    detail_column, delete_column = st.columns([4, 1], vertical_alignment="center")
    with detail_column:
        st.caption(
            f"Saved conversation · {selected_summary.turn_count} "
            f"{'question' if selected_summary.turn_count == 1 else 'questions'}"
        )
    with delete_column:
        if st.button("Delete…", width="stretch"):
            st.session_state["confirm_delete_conversation"] = True

    if st.session_state.get("confirm_delete_conversation"):
        st.warning("Delete this conversation and its saved answers? Your documents stay intact.")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button("Delete permanently", type="primary", width="stretch"):
                try:
                    client.delete_conversation(selected_summary.id)
                except ApiClientError as exc:
                    st.error(exc.message)
                else:
                    st.session_state["selected_conversation_id"] = None
                    st.session_state["starting_new_conversation"] = True
                    st.session_state["confirm_delete_conversation"] = False
                    st.session_state.pop("conversation_picker", None)
                    st.rerun()
        with cancel:
            if st.button("Keep conversation", width="stretch"):
                st.session_state["confirm_delete_conversation"] = False
                st.rerun()
    return selected_summary.id


def _render_sources(citations: list[Citation]) -> None:
    if not citations:
        return
    source_word = "Source" if len(citations) == 1 else "Sources"
    with st.expander(f"{source_word} · {len(citations)}"):
        for index, citation in enumerate(citations):
            metadata = [citation.document_name]
            if citation.page_number is not None:
                metadata.append(f"page {citation.page_number}")
            if citation.section:
                metadata.append(citation.section)
            st.markdown(f"**{citation.label}**")
            st.caption(" · ".join(metadata))
            # Document-derived text uses Streamlit's escaped renderer.
            st.write(citation.snippet)
            if index < len(citations) - 1:
                st.divider()


def _render_turn(client: UiClient, turn: ConversationTurn) -> None:
    with st.container(border=True):
        st.caption("You asked")
        st.write(turn.question)
        if turn.status is ConversationTurnStatus.PENDING:
            st.info("This answer is still being prepared.")
            if st.button("Check or retry", key=f"retry-turn-{turn.id}"):
                _retry_persisted_turn(client, turn)
            return
        if turn.status is ConversationTurnStatus.FAILED:
            message = "This question was not completed."
            if turn.retryable:
                message += " You can try it again without rewriting it."
            st.warning(message)
            if turn.retryable and st.button("Try again", key=f"retry-turn-{turn.id}"):
                _retry_persisted_turn(client, turn)
            return
        st.markdown("**From your library**")
        if turn.answer:
            st.write(turn.answer)
        if turn.no_answer:
            st.info(
                "There was not enough support in the selected documents. "
                "Try different wording or include more documents."
            )
        _render_sources(turn.citations)


def _retry_persisted_turn(client: UiClient, turn: ConversationTurn) -> None:
    request = ConversationTurnCreate(
        client_turn_id=turn.client_turn_id,
        message=turn.question,
        top_k=turn.top_k,
        document_ids=turn.document_ids,
    )
    try:
        with st.spinner("Checking the saved request…"):
            client.create_conversation_turn(turn.conversation_id, request)
    except ApiClientError as exc:
        st.session_state["turn_error"] = {
            "message": exc.message,
            "retryable": exc.retryable,
            "request_id": exc.request_id,
        }
    else:
        st.session_state["turn_error"] = None
    st.rerun()


def _suggest_question(text: str, *, key: str, draft_key: str) -> None:
    if st.button(text, key=key, width="stretch"):
        st.session_state[draft_key] = text
        st.rerun()


def _render_question_form(
    client: UiClient,
    settings: Settings,
    ready_documents: list[DocumentPublic],
    conversation_id: str | None,
) -> None:
    error_state = st.session_state.get("turn_error")
    if isinstance(error_state, dict):
        st.error(str(error_state.get("message", "The question could not be completed.")))
        if bool(error_state.get("retryable")):
            st.caption("Your exact question is still here. Choose Ask library to try again.")

    st.subheader("Ask a question", anchor=False)
    st.caption("Choose specific documents, or leave the selection empty to use everything ready.")
    document_names = {document.id: document.display_name for document in ready_documents}
    selected_document_ids = st.multiselect(
        "Look in",
        options=list(document_names),
        format_func=lambda document_id: document_names[document_id],
        placeholder="All ready documents",
        help="Leave empty to search your entire ready library.",
    )
    top_k = min(settings.retrieval_top_k, settings.retrieval_max_top_k)
    with st.expander("Search options"):
        top_k = st.slider(
            "Number of passages to consider",
            min_value=1,
            max_value=settings.retrieval_max_top_k,
            value=top_k,
            help="The default is usually best. Increase this only when answers miss context.",
        )

    draft_version = int(st.session_state["question_draft_version"])
    draft_key = f"question_draft_{draft_version}"
    st.caption("Not sure where to start?")
    suggestions = st.columns(3)
    with suggestions[0]:
        _suggest_question(
            "Summarize the main points",
            key=f"suggest-summary-{draft_version}",
            draft_key=draft_key,
        )
    with suggestions[1]:
        _suggest_question(
            "What decisions were made?",
            key=f"suggest-decisions-{draft_version}",
            draft_key=draft_key,
        )
    with suggestions[2]:
        _suggest_question(
            "What should I follow up on?",
            key=f"suggest-followup-{draft_version}",
            draft_key=draft_key,
        )

    with st.form("library-question-form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            key=draft_key,
            placeholder="What do these documents say about…",
            max_chars=settings.max_query_characters,
            height=105,
        )
        submitted = st.form_submit_button("Ask library", type="primary", width="stretch")

    if not submitted:
        return
    cleaned_question = question.strip()
    if not cleaned_question:
        st.warning("Write a question before asking your library.")
        return

    resolved_conversation_id = conversation_id
    if resolved_conversation_id is None:
        try:
            conversation = client.create_conversation()
        except ApiClientError as exc:
            st.session_state["turn_error"] = {
                "message": exc.message,
                "retryable": exc.retryable,
            }
            st.rerun()
        resolved_conversation_id = conversation.id
        st.session_state["selected_conversation_id"] = resolved_conversation_id
        st.session_state["starting_new_conversation"] = False
        st.session_state.pop("conversation_picker", None)

    request_payload = {
        "message": cleaned_question,
        "top_k": top_k,
        "document_ids": selected_document_ids or None,
    }
    client_turn_id = st.session_state.get("pending_client_turn_id")
    pending_payload = st.session_state.get("pending_turn_payload")
    if not isinstance(client_turn_id, str) or pending_payload != request_payload:
        client_turn_id = uuid4().hex
        st.session_state["pending_client_turn_id"] = client_turn_id
        st.session_state["pending_turn_payload"] = request_payload
    request = ConversationTurnCreate(
        client_turn_id=client_turn_id,
        message=cleaned_question,
        top_k=top_k,
        document_ids=selected_document_ids or None,
    )
    try:
        with st.spinner("Reading the most relevant passages…"):
            client.create_conversation_turn(resolved_conversation_id, request)
    except ApiClientError as exc:
        st.session_state["turn_error"] = {
            "message": exc.message,
            "retryable": exc.retryable,
            "request_id": exc.request_id,
        }
        if not exc.retryable:
            st.session_state["pending_client_turn_id"] = None
            st.session_state["pending_turn_payload"] = None
        st.rerun()

    st.session_state["turn_error"] = None
    st.session_state["pending_client_turn_id"] = None
    st.session_state["pending_turn_payload"] = None
    st.session_state["question_draft_version"] = draft_version + 1
    st.rerun()


def _render_ask(client: UiClient, settings: Settings) -> None:
    status, status_error = _safe_status(client)
    documents, documents_error = _safe_documents(client)
    if documents_error is not None:
        st.error(documents_error.message)
        st.info("Your saved work is untouched. Restore the knowledge service, then refresh.")
        return
    if status_error is not None or status is None:
        st.error(status_error.message if status_error is not None else "Status is unavailable.")
        st.info("Your saved work is untouched. Restore the knowledge service, then refresh.")
        return
    if not documents:
        _render_onboarding(client, settings)
        return
    if not _providers_configured(status):
        st.header("One setup step remains", anchor=False)
        st.warning(
            "Connect the document and answer providers on the server before asking questions."
        )
        if st.button("Open system details"):
            _request_section("System")
        return

    ready_documents = [
        document for document in documents if document.status is DocumentStatus.READY
    ]
    if not ready_documents:
        st.header("Your documents are getting ready", anchor=False)
        st.write("Processing continues even if you close this page.")
        if st.button("View activity", type="primary"):
            _request_section("Activity")
        return

    st.markdown('<div class="section-kicker">Ask</div>', unsafe_allow_html=True)
    st.header("Ask your library", anchor=False)
    st.caption("Saved conversations come back after a refresh, with their source passages intact.")

    conversation_page, conversation_error = _safe_conversations(client)
    if conversation_error is not None:
        st.error(conversation_error.message)
        return
    conversations = conversation_page.items if conversation_page is not None else []
    if conversation_page is not None and len(conversations) < conversation_page.total:
        st.caption(
            f"Showing the {len(conversations)} most recent of "
            f"{conversation_page.total} saved conversations."
        )
    conversation_id = _choose_conversation(client, conversations)

    if conversation_id is None:
        st.subheader("A fresh conversation", anchor=False)
        st.caption("Your first question will give this conversation a useful title.")
        turns_page = None
        turns_error = None
    else:
        turns_page, turns_error = _safe_turns(client, conversation_id)

    _render_question_form(
        client,
        settings,
        ready_documents,
        conversation_id,
    )
    if turns_error is not None:
        st.error(turns_error.message)
    elif turns_page is not None:
        if not turns_page.items:
            st.caption("This conversation is ready for its first question.")
        else:
            st.subheader("Conversation", anchor=False)
            if turns_page.total > len(turns_page.items):
                st.caption(
                    f"Showing the {len(turns_page.items)} most recent of "
                    f"{turns_page.total} saved questions."
                )
            for turn in turns_page.items:
                _render_turn(client, turn)


def _run_document_action(
    action_call: Callable[[str], JobRecord],
    *,
    document: DocumentPublic,
) -> None:
    try:
        job = action_call(document.id)
    except ApiClientError as exc:
        st.error(exc.message)
        return
    _track_job(job, document_name=document.display_name)
    st.session_state["activity_notice"] = (
        f"{job_action_label(job.kind)} {document.display_name}. Progress is saved."
    )
    st.session_state["activity_errors"] = []
    _request_section("Activity")


def _render_documents(client: UiClient, settings: Settings) -> None:
    st.markdown('<div class="section-kicker">Documents</div>', unsafe_allow_html=True)
    st.header("Documents", anchor=False)
    with st.container(border=True):
        _render_upload(client, settings, key_prefix="documents", heading="Add documents")

    documents, documents_error = _safe_documents(client)
    st.subheader("Your library", anchor=False)
    search = st.text_input(
        "Find a document",
        placeholder="Search by filename, type, or state",
        help="Searches the bounded library records already returned by the service.",
    )
    if documents_error is not None:
        st.error(documents_error.message)
        return
    if not documents:
        st.info("Your library is empty. Add a document above to begin.")
        return

    query = search.casefold().strip()
    filtered = [
        document
        for document in documents
        if not query
        or query
        in " ".join(
            (
                document.display_name,
                document.extension,
                document_status_label(document.status),
            )
        ).casefold()
    ]
    st.caption(f"{len(filtered)} of {len(documents)} documents")
    if not filtered:
        st.info("No documents match that search.")
        return

    for document in filtered:
        status_label = document_status_label(document.status)
        with st.expander(f"{document.display_name} — {status_label}"):
            st.caption(
                f"{document.extension.removeprefix('.').upper()} · "
                f"{format_bytes(document.size_bytes)} · "
                f"Updated {document.updated_at:%b %d, %Y}"
            )
            if document.status is DocumentStatus.READY:
                st.write(f"Ready across {document.chunk_count} searchable passages.")
            elif document.status is DocumentStatus.DELETION_FAILED:
                st.warning("Removal is incomplete. Restore storage access and retry below.")
            elif document.status is DocumentStatus.FAILED:
                st.warning("This document needs attention before it can be used in answers.")
            else:
                st.info("This document is still being prepared.")

            with st.expander("Technical details"):
                st.caption(
                    f"Content type: {document.content_type} · "
                    f"Version: {document.active_version} · "
                    f"Passages: {document.chunk_count}"
                )
                if document.error_code:
                    st.caption(f"Error code: {document.error_code}")

            can_refresh = document.status in {DocumentStatus.READY, DocumentStatus.FAILED}
            if st.button(
                "Refresh document",
                key=f"refresh-document-{document.id}",
                disabled=not can_refresh,
            ):
                _run_document_action(client.reindex_document, document=document)

            st.markdown("**Remove document**")
            st.caption(
                "This removes the stored file, its search index, and saved answers that cite it."
            )
            confirmation = st.text_input(
                f'Type "{document.display_name}" to confirm removal',
                key=f"delete-confirm-{document.id}",
            )
            delete_label = (
                f"Retry removal of {document.display_name}"
                if document.status is DocumentStatus.DELETION_FAILED
                else f"Remove {document.display_name} permanently"
            )
            if st.button(
                delete_label,
                key=f"delete-document-{document.id}",
                disabled=confirmation != document.display_name,
            ):
                _run_document_action(client.delete_document, document=document)


def _activity_document_name(
    job: JobRecord,
    document_names: dict[str, str],
    tracked_jobs: dict[str, dict[str, object]],
) -> str:
    tracked = tracked_jobs.get(job.id, {})
    tracked_name = tracked.get("document_name")
    if isinstance(tracked_name, str):
        return tracked_name
    return document_names.get(job.document_id, f"Document {job.document_id[:8]}")


def _render_activity_rows(
    client: UiClient,
    documents: list[DocumentPublic],
) -> bool:
    jobs_page, jobs_error = _safe_jobs(client)
    if jobs_error is not None:
        st.error(jobs_error.message)
        st.info("Activity remains stored by the service. Refresh when the connection returns.")
        return False
    if jobs_page is None or not jobs_page.items:
        st.info("No processing activity yet. Add a document to see progress here.")
        return False

    document_names = {document.id: document.display_name for document in documents}
    tracked_jobs = cast("dict[str, dict[str, object]]", st.session_state["tracked_jobs"])
    active = [job for job in jobs_page.items if job.status not in TERMINAL_JOB_STATUSES]
    failed = [job for job in jobs_page.items if job.status is JobStatus.FAILED]
    completed = [job for job in jobs_page.items if job.status is JobStatus.SUCCEEDED]

    def render_jobs(heading: str, jobs: list[JobRecord]) -> None:
        if not jobs:
            return
        st.subheader(heading, anchor=False)
        for job in jobs:
            document_name = _activity_document_name(job, document_names, tracked_jobs)
            action = job_action_label(job.kind)
            with st.container(border=True):
                label_column, state_column = st.columns([3, 1], vertical_alignment="center")
                with label_column:
                    st.markdown(f"**{action} {_escape_markdown(document_name)}**")
                    st.caption(f"Started {job.created_at:%b %d at %H:%M UTC}")
                with state_column:
                    st.write(job_status_label(job.status, job.stage))
                if job.status not in TERMINAL_JOB_STATUSES:
                    st.progress(job.progress, text=job_status_label(job.status, job.stage))
                    st.caption("You can close this page; progress is saved by the service.")
                elif job.status is JobStatus.FAILED:
                    st.warning("This work needs attention. Open Documents to inspect and retry.")
                    if job.error_code:
                        with st.expander("Error details"):
                            st.caption(f"Error code: {job.error_code}")
                else:
                    st.success("Complete")

    if active:
        render_jobs("In progress", active)
    render_jobs("Needs attention", failed)
    render_jobs("Completed", completed)
    if (failed or completed) and not active:
        st.caption("Showing the most recent saved activity.")
    return bool(active)


def _render_activity(client: UiClient) -> None:
    st.markdown('<div class="section-kicker">Activity</div>', unsafe_allow_html=True)
    heading, refresh = st.columns([4, 1], vertical_alignment="center")
    with heading:
        st.header("Recent activity", anchor=False)
        st.caption("Processing is stored by the service and comes back after a refresh.")
    with refresh:
        if st.button("Refresh", width="stretch"):
            st.rerun()

    notice = st.session_state.pop("activity_notice", None)
    if isinstance(notice, str) and notice:
        st.success(notice)
    errors = st.session_state.pop("activity_errors", [])
    if isinstance(errors, list):
        for error in errors:
            if isinstance(error, str):
                st.warning(error)

    documents, documents_error = _safe_documents(client)
    if documents_error is not None:
        documents = []

    has_active = _render_activity_rows(client, documents)
    if has_active:
        st.caption("Choose Refresh to check the latest progress.")


def _health_label(check_call: Callable[[], HealthCheck]) -> str:
    try:
        return check_call().status
    except ApiClientError:
        return "unavailable"


def _render_system(client: UiClient) -> None:
    st.markdown('<div class="section-kicker">System</div>', unsafe_allow_html=True)
    heading, refresh = st.columns([4, 1], vertical_alignment="center")
    with heading:
        st.header("System details", anchor=False)
        st.caption("For setup and troubleshooting. Credentials never appear in this workspace.")
    with refresh:
        if st.button("Refresh", key="refresh-system", width="stretch"):
            st.rerun()

    live_state = _health_label(client.health_live)
    ready_state = _health_label(client.health_ready)
    process_column, readiness_column = st.columns(2)
    with process_column, st.container(border=True):
        st.caption("Knowledge service")
        st.write(live_state.capitalize())
    with readiness_column, st.container(border=True):
        st.caption("Document search")
        st.write(ready_state.capitalize())

    status, status_error = _safe_status(client)
    if status_error is not None:
        st.error(status_error.message)
        return
    if status is None:
        st.warning("System details are not available right now.")
        return

    if not _providers_configured(status):
        st.warning(
            "Provider setup is incomplete. Configure credentials on the server, then refresh."
        )

    st.markdown("### Library overview")
    st.write(
        f"{status.document_count} documents · {status.ready_document_count} ready · "
        f"{status.queued_job_count} waiting"
    )
    with st.expander("Models and storage"):
        st.write(f"Document matching: {status.embedding_provider} / {status.embedding_model}")
        st.write(f"Answers: {status.chat_provider} / {status.chat_model}")
        st.caption(
            f"{status.embedding_dimensions} dimensions · Collection {status.collection} · "
            f"{status.chunk_count} passages"
        )
    with st.expander("Dependency checks"):
        for dependency in status.dependencies:
            st.write(f"{dependency.name.replace('_', ' ').title()}: {dependency.status}")
            if dependency.detail:
                st.caption(dependency.detail)


def main() -> None:
    st.set_page_config(
        page_title="Personal Library",
        page_icon=":material/library_books:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(STATIC_STYLES, unsafe_allow_html=True)
    _initialize_state()
    _apply_requested_section()

    try:
        settings = _resolve_settings()
        client = _resolve_client()
    except Exception:
        st.markdown(HEADER_HTML, unsafe_allow_html=True)
        st.error("The library cannot connect because server-side setup is incomplete.")
        st.info("Configure the service URL and access token, then restart this workspace.")
        st.stop()

    _render_header()
    section = _render_navigation()
    if section == "Ask":
        _render_ask(client, settings)
    elif section == "Documents":
        _render_documents(client, settings)
    elif section == "Activity":
        _render_activity(client)
    else:
        _render_system(client)


main()
