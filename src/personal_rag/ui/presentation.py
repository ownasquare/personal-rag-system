"""Trusted static presentation helpers for the Streamlit workspace."""

from __future__ import annotations

from personal_rag.models import DocumentStatus, JobKind, JobStage, JobStatus

STATIC_STYLES = """
<style>
  :root {
    --paper: #f5f2eb;
    --paper-raised: #fbfaf7;
    --ink: #272922;
    --muted: #6c6d65;
    --line: #d8d3c8;
    --accent: #3f654f;
    --accent-soft: #dfe8df;
    --warning-soft: #f2e7cf;
    --danger-soft: #f4dddd;
  }

  .stApp {
    background: var(--paper);
    color: var(--ink);
  }

  .block-container {
    max-width: 1120px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
  }

  [data-testid="stHeader"] {
    background: color-mix(in srgb, var(--paper) 92%, transparent);
  }

  [data-testid="stSidebar"] {
    background: #ece8de;
    border-right: 1px solid var(--line);
  }

  .library-header {
    margin-bottom: 0.15rem;
  }

  h1.library-title {
    color: var(--ink);
    font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
    font-size: clamp(1.45rem, 3vw, 1.85rem);
    font-weight: 580;
    letter-spacing: -0.025em;
    line-height: 1.15;
    margin: 0;
  }

  .library-subtitle {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.4;
    margin: 0.25rem 0 0;
  }

  .onboarding-steps {
    align-items: center;
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-radius: 9px;
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    font-size: 0.88rem;
    gap: 0.45rem;
    margin: 0.75rem 0 1rem;
    padding: 0.68rem 0.8rem;
  }

  .onboarding-step {
    align-items: center;
    display: inline-flex;
    gap: 0.35rem;
  }

  .onboarding-number {
    color: var(--accent);
    font-weight: 750;
  }

  .onboarding-separator { color: var(--line); }

  div[data-testid="stSegmentedControl"] {
    margin: 1.3rem 0 1.65rem;
  }

  div[data-testid="stSegmentedControl"] > div {
    background: transparent;
    border-bottom: 1px solid var(--line);
    border-radius: 0;
    gap: 0.25rem;
    padding: 0;
  }

  div[data-testid="stSegmentedControl"] label {
    border-radius: 7px 7px 0 0 !important;
    min-height: 2.35rem;
  }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: color-mix(in srgb, var(--paper-raised) 88%, transparent);
    border-color: var(--line) !important;
    border-radius: 10px !important;
  }

  div[data-testid="stFileUploader"] section {
    background: var(--paper-raised);
    border-color: var(--line);
    border-radius: 10px;
    min-height: 5.5rem;
  }

  div[data-testid="stExpander"] {
    background: color-mix(in srgb, var(--paper-raised) 84%, transparent);
    border-color: var(--line);
    border-radius: 9px;
    overflow: hidden;
  }

  div[data-testid="stTextInputRootElement"],
  div[data-testid="stTextArea"] textarea,
  div[data-baseweb="select"] > div {
    background: var(--paper-raised);
    border-color: var(--line);
  }

  div[data-testid="stRadio"] [role="radiogroup"] {
    gap: 0.38rem;
  }

  div[data-testid="stRadio"] [role="radiogroup"] label {
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-radius: 8px;
    margin: 0;
    padding: 0.58rem 0.7rem;
  }

  div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: var(--accent-soft);
    border-color: var(--accent);
  }

  .stButton > button,
  .stFormSubmitButton > button {
    border-radius: 7px;
    font-weight: 620;
  }

  hr {
    border-color: var(--line) !important;
  }

  @media (max-width: 700px) {
    .block-container {
      padding: 1rem 0.85rem 3rem;
    }

    h1.library-title { font-size: 1.3rem; }

    .library-subtitle { font-size: 0.84rem; }

    div[data-testid="stSegmentedControl"] {
      margin: 0.8rem 0 1.1rem;
    }

    .onboarding-steps { font-size: 0.82rem; }

    div[data-testid="stSegmentedControl"] label {
      font-size: 0.82rem;
      padding-inline: 0.55rem !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    div[data-testid="stRadio"] [role="radiogroup"] label {
      min-height: 2.75rem;
    }
  }
</style>
"""

HEADER_HTML = """
<header class="library-header">
  <h1 class="library-title">Personal Library</h1>
  <p class="library-subtitle">Add documents, ask questions, and check the source.</p>
</header>
"""

ONBOARDING_STEPS_HTML = """
<div class="onboarding-steps">
  <div class="onboarding-step">
    <span class="onboarding-number">1</span><span>Add files</span>
  </div>
  <span class="onboarding-separator">→</span>
  <div class="onboarding-step">
    <span class="onboarding-number">2</span><span>Wait until ready</span>
  </div>
  <span class="onboarding-separator">→</span>
  <div class="onboarding-step">
    <span class="onboarding-number">3</span><span>Ask and check sources</span>
  </div>
</div>
"""


def format_bytes(size_bytes: int) -> str:
    """Format a non-negative byte count for compact document metadata."""

    value = float(size_bytes)
    units = ("B", "KB", "MB", "GB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def document_status_label(status: DocumentStatus) -> str:
    """Translate implementation states into user-facing library states."""

    if status is DocumentStatus.READY:
        return "Ready to use"
    if status in {DocumentStatus.FAILED, DocumentStatus.DELETION_FAILED}:
        return "Needs attention"
    if status in {DocumentStatus.DELETING, DocumentStatus.DELETED}:
        return "Removing"
    return "Processing"


def job_action_label(kind: JobKind) -> str:
    """Return a calm activity verb for a durable job kind."""

    return {
        JobKind.INGEST: "Adding",
        JobKind.REINDEX: "Reprocessing",
        JobKind.DELETE: "Removing",
    }[kind]


def job_status_label(status: JobStatus, stage: JobStage) -> str:
    """Return a readable activity state without exposing worker terminology."""

    if status is JobStatus.SUCCEEDED:
        return "Complete"
    if status is JobStatus.FAILED:
        return "Needs attention"
    if status is JobStatus.RETRYING:
        return "Trying again"
    if stage is JobStage.QUEUED:
        return "Waiting"
    if stage in {JobStage.VALIDATING, JobStage.EXTRACTING, JobStage.CHUNKING}:
        return "Reading document"
    if stage in {JobStage.EMBEDDING, JobStage.INDEXING, JobStage.VERIFYING}:
        return "Preparing search"
    if stage is JobStage.DELETING:
        return "Removing document"
    return "Working"
