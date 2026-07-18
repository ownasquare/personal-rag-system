"""Polished Streamlit shell for the Personal Knowledge Studio."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import streamlit as st

from personal_rag.config import Settings, get_settings
from personal_rag.models import (
    ChatHistoryMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentPublic,
    DocumentStatus,
    JobRecord,
    JobStatus,
    SystemStatus,
    UploadReceipt,
)
from personal_rag.ui.client import ApiClientError, HealthCheck, RagApiClient

SUPPORTED_FILE_TYPES = ["pdf", "docx", "md", "txt"]
TERMINAL_JOB_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED}
STATIC_STYLES = """
<style>
  .stApp { background: linear-gradient(180deg, rgba(37, 99, 235, 0.055), transparent 18rem); }
  .block-container { max-width: 1240px; padding-top: 2.25rem; padding-bottom: 4rem; }
  div[data-testid="stMetric"] {
    border: 1px solid color-mix(in srgb, currentColor 13%, transparent);
    border-radius: 0.9rem;
    padding: 0.8rem 1rem;
    background: color-mix(in srgb, var(--background-color) 94%, currentColor 6%);
  }
  div[data-testid="stFileUploader"] section { border-radius: 0.9rem; }
  div[data-testid="stExpander"] { border-radius: 0.8rem; overflow: hidden; }
  div[data-testid="stChatMessage"] { border-radius: 0.9rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
  .stTabs [data-baseweb="tab"] { border-radius: 0.75rem 0.75rem 0 0; padding-inline: 1rem; }
  @media (max-width: 640px) {
    .block-container { padding: 1.25rem 1rem 3rem; }
    .stTabs [data-baseweb="tab"] { padding-inline: 0.55rem; font-size: 0.86rem; }
  }
</style>
"""


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

    def reindex_document(self, document_id: str) -> JobRecord: ...

    def delete_document(self, document_id: str) -> JobRecord: ...

    def chat(self, request: ChatRequest) -> ChatResponse: ...


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
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("tracked_jobs", {})
    st.session_state.setdefault("chat_error", None)
    st.session_state.setdefault("chat_draft_version", 0)


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


def _providers_configured(status: SystemStatus | None) -> bool:
    if status is None or status.status == "needs_setup":
        return False
    return not any(dependency.status == "not_configured" for dependency in status.dependencies)


def _format_bytes(size_bytes: int) -> str:
    value = float(size_bytes)
    units = ("B", "KB", "MB", "GB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _track_job(job: JobRecord, *, document_name: str, action: str) -> None:
    tracked_jobs = cast("dict[str, dict[str, object]]", st.session_state["tracked_jobs"])
    tracked_jobs[job.id] = {
        "id": job.id,
        "document_name": document_name,
        "action": action,
        "status": job.status.value,
        "stage": job.stage.value,
        "progress": job.progress,
    }


def _render_job_tracker(client: UiClient, poll_seconds: float) -> None:
    tracked_jobs = cast("dict[str, dict[str, object]]", st.session_state["tracked_jobs"])
    if not tracked_jobs:
        return

    polling_active = any(
        tracked.get("status") not in {status.value for status in TERMINAL_JOB_STATUSES}
        for tracked in tracked_jobs.values()
    )

    @st.fragment(run_every=poll_seconds if polling_active else None)
    def job_tracker_fragment() -> None:
        current_jobs = cast("dict[str, dict[str, object]]", st.session_state["tracked_jobs"])
        if not current_jobs:
            return

        st.subheader("Processing activity", anchor=False)
        for job_id, tracked in list(current_jobs.items()):
            try:
                job = client.get_job(job_id)
            except ApiClientError as exc:
                st.warning(f"Could not refresh this job: {exc.message}")
                continue

            tracked.update(
                status=job.status.value,
                stage=job.stage.value,
                progress=job.progress,
            )
            document_name = str(tracked.get("document_name", "document"))
            action = str(tracked.get("action", "Processing"))
            with st.container(border=True):
                st.markdown(f"**{action} {document_name}**")
                if job.status == JobStatus.SUCCEEDED:
                    st.success("Complete. The library will reflect server truth on refresh.")
                elif job.status == JobStatus.FAILED:
                    st.error(
                        "Processing failed. Review the document details and retry if appropriate."
                    )
                    if job.error_code:
                        st.caption(f"Error code: {job.error_code}")
                else:
                    st.progress(job.progress, text=job.stage.value.replace("_", " ").title())
                    st.caption("This job is stored by the API and continues if this page closes.")
                if st.button(
                    "Dismiss",
                    key=f"dismiss-job-{job_id}",
                    disabled=job.status not in TERMINAL_JOB_STATUSES,
                ):
                    current_jobs.pop(job_id, None)
                    st.rerun()

    job_tracker_fragment()


def _render_header(status: SystemStatus | None, status_error: ApiClientError | None) -> None:
    heading, state = st.columns([4, 1], vertical_alignment="center")
    with heading:
        st.title("Personal Knowledge Studio")
        st.caption(
            "Upload private reference material, search your library, and get grounded answers "
            "with source citations."
        )
    with state:
        if status_error is not None:
            st.error("API unavailable")
        elif status is not None and status.status == "ready":
            st.success("System ready")
        elif status is not None and status.status == "needs_setup":
            st.warning("Setup needed")
        else:
            st.warning("System degraded")


def _render_provider_blocker() -> None:
    st.warning(
        "Provider setup required. Configure the embedding and answer-provider "
        "credentials on the server, then refresh this page."
    )


def _render_chat(
    client: UiClient,
    settings: Settings,
    status: SystemStatus | None,
    documents: list[DocumentPublic],
    documents_error: ApiClientError | None,
) -> None:
    st.subheader("Chat with your knowledge", anchor=False)
    st.caption("Answers are grounded in ready documents; citations come directly from the API.")

    history = cast("list[dict[str, object]]", st.session_state["chat_history"])
    toolbar_left, toolbar_right = st.columns([3, 1], vertical_alignment="center")
    with toolbar_left:
        st.caption(f"{len(history) // 2} answered question(s) in this session")
    with toolbar_right:
        if st.button("Clear chat", width="stretch", disabled=not history):
            history.clear()
            st.session_state["chat_error"] = None
            st.session_state["chat_draft_version"] += 1
            st.rerun()

    for message in history:
        role = str(message.get("role", "assistant"))
        safe_role = role if role in {"user", "assistant"} else "assistant"
        with st.chat_message(safe_role):
            st.write(str(message.get("content", "")))
            if safe_role == "assistant":
                _render_citations(message.get("citations", []))

    error_state = st.session_state.get("chat_error")
    if isinstance(error_state, dict):
        st.error(str(error_state.get("message", "The question could not be answered.")))
        if bool(error_state.get("retryable")):
            st.caption(
                "You can retry without retyping; your question and chat history are preserved."
            )

    if documents_error is not None:
        st.error(documents_error.message)
        return
    if status is None:
        st.warning("System status is unavailable. Restore API connectivity before chatting.")
        return
    if not _providers_configured(status):
        _render_provider_blocker()
        return

    ready_documents = [
        document for document in documents if document.status == DocumentStatus.READY
    ]
    if not documents:
        st.markdown("### Add your first document")
        st.info("Open Library, select one or more files, then choose Add to library.")
        return
    if not ready_documents:
        st.info("Your library is processing. Chat becomes available when a document is ready.")
        return

    document_names = {document.id: document.display_name for document in ready_documents}
    top_k = st.slider(
        "Sources to retrieve",
        min_value=1,
        max_value=settings.retrieval_max_top_k,
        value=min(settings.retrieval_top_k, settings.retrieval_max_top_k),
        help="More sources can improve recall but may add less-relevant context.",
    )
    selected_document_ids = st.multiselect(
        "Search within (optional)",
        options=list(document_names),
        format_func=lambda document_id: document_names[document_id],
        help="Leave empty to search every ready document.",
    )

    with st.form("chat-question-form", clear_on_submit=False):
        question = st.text_area(
            "Ask about your library",
            key=f"chat_draft_{st.session_state['chat_draft_version']}",
            placeholder="What do my notes say about…",
            max_chars=settings.max_query_characters,
            height=100,
        )
        submitted = st.form_submit_button("Ask library", type="primary", width="stretch")

    if not submitted:
        return
    cleaned_question = question.strip()
    if not cleaned_question:
        st.warning("Enter a question before asking the library.")
        return

    history_window = (
        history[-settings.max_history_messages :] if settings.max_history_messages else []
    )
    api_history = [
        ChatHistoryMessage(
            role="user" if str(item.get("role")) == "user" else "assistant",
            content=str(item.get("content", "")),
        )
        for item in history_window
        if str(item.get("content", "")).strip()
    ]
    request = ChatRequest(
        message=cleaned_question,
        history=api_history,
        top_k=top_k,
        document_ids=selected_document_ids or None,
    )
    try:
        with st.spinner("Searching your library and checking sources…"):
            response = client.chat(request)
    except ApiClientError as exc:
        st.session_state["chat_error"] = {
            "message": exc.message,
            "retryable": exc.retryable,
            "request_id": exc.request_id,
        }
        st.rerun()

    history.extend(
        [
            {"role": "user", "content": cleaned_question, "citations": []},
            {
                "role": "assistant",
                "content": response.answer,
                "citations": [citation.model_dump(mode="json") for citation in response.citations],
                "no_answer": response.no_answer,
            },
        ]
    )
    st.session_state["chat_error"] = None
    st.session_state["chat_draft_version"] += 1
    st.rerun()


def _render_citations(raw_citations: object) -> None:
    if not isinstance(raw_citations, list):
        return
    citations: list[Citation] = []
    for raw_citation in raw_citations:
        try:
            citations.append(Citation.model_validate(raw_citation))
        except (TypeError, ValueError):
            continue
    if not citations:
        return

    st.caption("Sources")
    for citation in citations:
        with st.expander(f"{citation.label} · {citation.document_name}"):
            metadata: list[str] = []
            if citation.page_number is not None:
                metadata.append(f"Page {citation.page_number}")
            if citation.section:
                metadata.append(citation.section)
            if citation.score is not None:
                metadata.append(f"Relevance {citation.score:.2f}")
            if metadata:
                st.caption(" · ".join(metadata))
            # Document-derived text always uses Streamlit's escaped text renderer.
            st.write(citation.snippet)


def _render_upload(client: UiClient, settings: Settings) -> None:
    st.subheader("Add documents", anchor=False)
    st.caption(
        "PDF, DOCX, Markdown, and text files are accepted. Nothing uploads until you confirm."
    )
    uploads = st.file_uploader(
        "Choose documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help=f"Each file can be up to {_format_bytes(settings.upload_max_bytes)}.",
    )
    add_clicked = st.button(
        "Add to library",
        type="primary",
        width="stretch",
        disabled=not uploads,
    )
    if not add_clicked:
        return

    accepted = 0
    duplicates = 0
    failures: list[str] = []
    progress = st.progress(0.0, text="Preparing uploads…")
    for index, uploaded in enumerate(uploads, start=1):
        if uploaded.size > settings.upload_max_bytes:
            failures.append(f"{uploaded.name}: file exceeds the configured size limit")
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
            _track_job(receipt.job, document_name=receipt.document.display_name, action="Adding")
        progress.progress(index / len(uploads), text=f"Submitted {index} of {len(uploads)}")

    if accepted:
        noun = "document" if accepted == 1 else "documents"
        duplicate_note = f" ({duplicates} already known)" if duplicates else ""
        st.success(f"{accepted} {noun} accepted{duplicate_note}. Processing continues in the API.")
    for failure in failures:
        st.error(failure)


def _render_library(
    client: UiClient,
    settings: Settings,
    documents: list[DocumentPublic],
    documents_error: ApiClientError | None,
) -> None:
    _render_upload(client, settings)
    st.divider()
    st.subheader("Your library", anchor=False)
    search = st.text_input(
        "Search documents",
        placeholder="Filename, extension, or status",
        help="Search is local to the sanitized library records already returned by the API.",
    )
    if documents_error is not None:
        st.error(documents_error.message)
        return
    if not documents:
        st.markdown("### Add your first document")
        st.info("Choose one or more files above, then select Add to library.")
        return

    query = search.casefold().strip()
    filtered = [
        document
        for document in documents
        if not query
        or query
        in " ".join((document.display_name, document.extension, document.status.value)).casefold()
    ]
    st.caption(f"Showing {len(filtered)} of {len(documents)} documents")
    if not filtered:
        st.info("No documents match that search.")
        return

    for document in filtered:
        status_label = document.status.value.replace("_", " ").title()
        with st.expander(f"{document.display_name} · {status_label}"):
            metrics = st.columns(3)
            metrics[0].metric("Size", _format_bytes(document.size_bytes))
            metrics[1].metric("Chunks", document.chunk_count)
            metrics[2].metric("Version", document.active_version)
            st.caption(
                f"{document.content_type} · Added {document.created_at:%b %d, %Y} · "
                f"Updated {document.updated_at:%b %d, %Y %H:%M UTC}"
            )
            if document.error_code:
                if document.status == DocumentStatus.DELETION_FAILED:
                    st.warning(
                        "Deletion is incomplete. Restore storage access, then retry deletion."
                    )
                else:
                    st.warning(
                        "The last processing attempt failed. You can reindex after checking "
                        "provider and worker status."
                    )
                st.caption(f"Error code: {document.error_code}")

            action_column, confirmation_column = st.columns([1, 2])
            with action_column:
                can_reindex = document.status in {
                    DocumentStatus.READY,
                    DocumentStatus.FAILED,
                }
                if st.button(
                    "Reindex",
                    key=f"reindex-{document.id}",
                    width="stretch",
                    disabled=not can_reindex,
                ):
                    _run_document_action(
                        client.reindex_document,
                        document=document,
                        action="Reindexing",
                    )
            with confirmation_column:
                confirmation = st.text_input(
                    f'Type "{document.display_name}" to confirm deletion',
                    key=f"delete-confirm-{document.id}",
                    help="Deletion is asynchronous and removes indexed chunks after readback.",
                )
                delete_label = (
                    "Retry deletion"
                    if document.status == DocumentStatus.DELETION_FAILED
                    else f"Delete {document.display_name}"
                )
                if st.button(
                    delete_label,
                    key=f"delete-{document.id}",
                    width="stretch",
                    disabled=confirmation != document.display_name,
                ):
                    _run_document_action(
                        client.delete_document,
                        document=document,
                        action="Deleting",
                    )


def _run_document_action(
    action_call: Callable[[str], JobRecord],
    *,
    document: DocumentPublic,
    action: str,
) -> None:
    try:
        job = action_call(document.id)
    except ApiClientError as exc:
        st.error(exc.message)
        return
    _track_job(job, document_name=document.display_name, action=action)
    st.success(f"{action} was queued. Progress is stored by the API.")


def _health_label(check_call: Callable[[], HealthCheck]) -> str:
    try:
        return check_call().status
    except ApiClientError:
        return "unavailable"


def _render_settings_status(
    client: UiClient,
    status: SystemStatus | None,
    status_error: ApiClientError | None,
) -> None:
    heading, refresh = st.columns([4, 1], vertical_alignment="center")
    with heading:
        st.subheader("Settings & Status", anchor=False)
        st.caption("Operational metadata is sanitized; credentials are never rendered in the UI.")
    with refresh:
        if st.button("Refresh status", width="stretch"):
            st.rerun()

    live_state = _health_label(client.health_live)
    ready_state = _health_label(client.health_ready)
    health_columns = st.columns(2)
    health_columns[0].metric("API process", live_state.title())
    health_columns[1].metric("API readiness", ready_state.title())

    if status_error is not None:
        st.error(status_error.message)
        return
    if status is None:
        st.warning("Status is not currently available.")
        return

    if not _providers_configured(status):
        _render_provider_blocker()

    counts = st.columns(4)
    counts[0].metric("Documents", status.document_count)
    counts[1].metric("Ready", status.ready_document_count)
    counts[2].metric("Chunks", status.chunk_count)
    counts[3].metric("Queued jobs", status.queued_job_count)

    st.markdown("#### Models")
    model_columns = st.columns(2)
    with model_columns[0]:
        st.write(f"Embedding: {status.embedding_provider} / {status.embedding_model}")
        st.caption(f"{status.embedding_dimensions} dimensions")
    with model_columns[1]:
        st.write(f"Answers: {status.chat_provider} / {status.chat_model}")
        st.caption(f"Collection: {status.collection}")

    st.markdown("#### Dependencies")
    dependency_rows = [
        {"Dependency": dependency.name, "State": dependency.status}
        for dependency in status.dependencies
    ]
    st.dataframe(dependency_rows, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Personal Knowledge Studio",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # This block contains trusted static CSS only. Document-derived content never enters it.
    st.markdown(STATIC_STYLES, unsafe_allow_html=True)
    _initialize_state()

    try:
        settings = _resolve_settings()
        client = _resolve_client()
    except Exception:
        st.title("Personal Knowledge Studio")
        st.error("Server-side configuration is incomplete, so the UI cannot connect safely.")
        st.info("Configure the API URL and server-side bearer token, then restart Streamlit.")
        st.stop()

    status, status_error = _safe_status(client)
    documents, documents_error = _safe_documents(client)
    _render_header(status, status_error)
    _render_job_tracker(client, settings.ui_poll_seconds)

    chat_tab, library_tab, settings_tab = st.tabs(["Chat", "Library", "Settings & Status"])
    with chat_tab:
        _render_chat(client, settings, status, documents, documents_error)
    with library_tab:
        _render_library(client, settings, documents, documents_error)
    with settings_tab:
        _render_settings_status(client, status, status_error)


main()
