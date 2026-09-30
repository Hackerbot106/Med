"""
SQLite persistence layer for the Triage Assistant prototype.

Design notes (privacy / responsible AI):
 - We store a generated `triage_id` (e.g. TRG-2026-000123) as the primary
   reference used across the UI and any printed/exported note. Patient
   name/phone are optional and never required to create a triage note.
 - No real patient data should ever be entered into this demo system -
   see README / DISCLAIMER.
 - `audit_log` records every view/edit of a triage note for traceability.
"""

import sqlite3
import os
import json
import datetime
import uuid
import logging
from contextlib import contextmanager

logger = logging.getLogger("triage_engine.database")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "triage.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode lets concurrent readers (e.g. the dashboard) proceed while a
    # writer (e.g. intake) is mid-transaction, instead of the default
    # rollback-journal mode's whole-database write lock - meaningfully
    # reduces "database is locked" errors under any real concurrent load.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def session():
    """Context manager for a single unit of work: commits on clean exit,
    rolls back and re-raises on any exception, and always closes the
    connection - so a mid-request error can never leave a half-written
    transaction or a leaked connection behind.

        with database.session() as conn:
            conn.execute("UPDATE ... ")
            # commit happens automatically here; on an exception above,
            # the transaction is rolled back instead and the connection
            # is still closed cleanly.
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def health_check():
    """Used by the /health endpoint: returns (ok: bool, message: str)."""
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return True, "database reachable"
    except Exception as exc:  # noqa: BLE001
        logger.error("Database health check failed: %s", exc, exc_info=True)
        return False, f"database error: {exc}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,          -- demo-only plaintext; NOT for production use
    role TEXT NOT NULL,              -- intake_worker | reviewer | auditor
    display_name TEXT NOT NULL,
    facility_name TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triage_id TEXT UNIQUE NOT NULL,
    display_name TEXT,               -- optional, minimal data collection
    age TEXT,
    sex TEXT,
    facility_type TEXT NOT NULL,
    facility_name TEXT,
    preferred_language TEXT NOT NULL DEFAULT 'en',
    consent_given INTEGER NOT NULL DEFAULT 0,
    consent_timestamp TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triage_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    scenario TEXT,                    -- e.g. outpatient_queue, occupational_health, campus_fever...
    input_mode TEXT NOT NULL,         -- text | voice
    raw_symptom_text TEXT,
    chief_complaint TEXT,
    symptoms_json TEXT,               -- list of {term, duration, severity}
    negated_symptoms_json TEXT,       -- list of symptoms explicitly denied by patient
    timeline_json TEXT,               -- ordered list of {event, when}
    missing_info_json TEXT,           -- list of missing fields
    follow_up_questions_json TEXT,    -- list of questions for reviewer/health worker to ask
    vitals_json TEXT,                 -- extracted / manually entered vitals
    ai_summary TEXT,                  -- local-LLM-generated narrative summary (nullable; rules-based fallback used if absent)
    ai_summary_model TEXT,            -- which local model produced ai_summary, for transparency/audit
    ai_summary_generated_at TEXT,
    ai_followups_json TEXT,           -- extra LLM-suggested follow-up questions (nullable, legacy)
    ai_summary_translations_json TEXT, -- {lang_code: translated_text} cache for AI-translated summary
    extra_translations_json TEXT,     -- {field_key: {lang_code: translated_text}} cache for every OTHER
                                       -- free-text AI field (chief complaint, missing info, extra
                                       -- follow-ups, vision description, vision findings) - see
                                       -- app.py's _translate_field()/_translate_list_field()
    extraction_source TEXT,           -- human-readable: which engine (AI vs rules-only) produced this note
    ai_powered INTEGER NOT NULL DEFAULT 0,  -- 1 if AI model drove the extraction, 0 if rules-only fallback
    rules_only_symptoms_json TEXT,    -- what the deterministic safety net alone detected (transparency)
    rules_only_risk_tier TEXT,        -- the tier the safety net alone would have assigned
    next_followup_date TEXT,          -- reviewer-set date for maternal/chronic-care follow-up tracking
    followup_completed INTEGER NOT NULL DEFAULT 0,
    image_ai_description TEXT,        -- local vision-model description of an uploaded photo (non-diagnostic)
    image_ai_model TEXT,
    image_ai_findings_json TEXT,      -- MedGemma AI-suggested possible findings (hedged, disclaimed, for clinician confirmation)
    image_ai_findings_model TEXT,
    ocr_extracted_text TEXT,
    ocr_source_filename TEXT,
    created_by_user_id INTEGER REFERENCES users(id),  -- which intake worker/reviewer created this
                                       -- note - see app.py's triage_detail() access check: an
                                       -- intake_worker may open the reviewer view of a note they
                                       -- just created (for the "Open in Reviewer View" link right
                                       -- after intake), but not an arbitrary other patient's note
    risk_tier TEXT NOT NULL DEFAULT 'unclassified',  -- emergency | high | medium | low
    risk_score INTEGER NOT NULL DEFAULT 0,
    risk_reasons_json TEXT,
    status TEXT NOT NULL DEFAULT 'new',  -- new | in_review | reviewed | referred | closed
    reviewer_id INTEGER REFERENCES users(id),
    reviewer_notes TEXT,
    reviewed_at TEXT,
    referral_facility TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT,
    timestamp TEXT NOT NULL,
    details TEXT
);
"""



# Columns added after the initial release - kept here so an existing
# data/triage.db from before the AI-summary feature gets upgraded in place
# instead of erroring out with "no such column".
MIGRATION_COLUMNS = [
    ("triage_notes", "ai_summary", "TEXT"),
    ("triage_notes", "ai_summary_model", "TEXT"),
    ("triage_notes", "ai_summary_generated_at", "TEXT"),
    ("triage_notes", "ai_followups_json", "TEXT"),
    ("triage_notes", "ai_summary_translations_json", "TEXT"),
    ("triage_notes", "extraction_source", "TEXT"),
    ("triage_notes", "ai_powered", "INTEGER NOT NULL DEFAULT 0"),
    ("triage_notes", "rules_only_symptoms_json", "TEXT"),
    ("triage_notes", "rules_only_risk_tier", "TEXT"),
    ("triage_notes", "next_followup_date", "TEXT"),
    ("triage_notes", "followup_completed", "INTEGER NOT NULL DEFAULT 0"),
    ("triage_notes", "image_ai_description", "TEXT"),
    ("triage_notes", "image_ai_model", "TEXT"),
    ("triage_notes", "image_ai_findings_json", "TEXT"),
    ("triage_notes", "image_ai_findings_model", "TEXT"),
    ("triage_notes", "extra_translations_json", "TEXT"),
    ("triage_notes", "created_by_user_id", "INTEGER REFERENCES users(id)"),
]


def _run_migrations(conn):
    for table, column, coltype in MIGRATION_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db(reset=False):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = get_connection()
    conn.executescript(SCHEMA)
    _run_migrations(conn)
    conn.commit()
    conn.close()


def now_iso():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def new_triage_id():
    year = datetime.datetime.utcnow().year
    suffix = uuid.uuid4().hex[:6].upper()
    return f"TRG-{year}-{suffix}"


def log_audit(actor, action, target_type, target_id, details=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_log (actor, action, target_type, target_id, timestamp, details) VALUES (?,?,?,?,?,?)",
        (actor, action, target_type, str(target_id), now_iso(), json.dumps(details or {})),
    )
    conn.commit()
    conn.close()


def dict_from_row(row):
    if row is None:
        return None
    return dict(row)
