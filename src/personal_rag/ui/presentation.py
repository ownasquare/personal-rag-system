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

  .library-wordmark {
    color: var(--accent);
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    line-height: 1.2;
    margin: 0 0 0.45rem;
    text-transform: uppercase;
  }

  h1.library-title {
    color: var(--ink);
    font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
    font-size: clamp(1.8rem, 4vw, 3rem);
    font-weight: 540;
    letter-spacing: -0.035em;
    line-height: 1.04;
    margin: 0;
    max-width: 760px;
  }

  .library-subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.55;
    margin: 0.7rem 0 0;
    max-width: 680px;
  }

  .section-kicker {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 720;
    letter-spacing: 0.11em;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
  }

  .onboarding-steps {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin: 1.15rem 0 1.5rem;
  }

  .onboarding-step {
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-radius: 10px;
    min-height: 105px;
    padding: 0.9rem 1rem;
  }

  .onboarding-number {
    color: var(--accent);
    font-family: ui-serif, Georgia, serif;
    font-size: 1.25rem;
    margin-bottom: 0.35rem;
  }

  .onboarding-copy {
    color: var(--ink);
    font-size: 0.9rem;
    line-height: 1.42;
  }

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

    h1.library-title {
      font-size: 1.28rem;
      letter-spacing: -0.02em;
      line-height: 1.18;
      max-width: none;
    }

    .library-subtitle {
      display: none;
    }

    .library-wordmark {
      margin-bottom: 0.3rem;
    }

    div[data-testid="stSegmentedControl"] {
      margin: 0.8rem 0 1.1rem;
    }

    .onboarding-steps {
      grid-template-columns: 1fr;
    }

    .onboarding-step {
      min-height: 0;
    }

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
<div class="library-wordmark">Personal Library</div>
<h1 class="library-title">Your documents, ready when you need them.</h1>
<p class="library-subtitle">
  Keep notes and reference files in one private place, then find the part that matters.
</p>
"""

ONBOARDING_STEPS_HTML = """
<div class="onboarding-steps">
  <div class="onboarding-step">
    <div class="onboarding-number">01</div>
    <div class="onboarding-copy">Add a PDF, document, Markdown file, or text note.</div>
  </div>
  <div class="onboarding-step">
    <div class="onboarding-number">02</div>
    <div class="onboarding-copy">Your library prepares it and keeps progress safely.</div>
  </div>
  <div class="onboarding-step">
    <div class="onboarding-number">03</div>
    <div class="onboarding-copy">Ask a question and open the exact passages used.</div>
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
        JobKind.REINDEX: "Refreshing",
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
        return "Waiting to start"
    return stage.value.replace("_", " ").capitalize()
