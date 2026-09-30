"""
Privacy / responsible-AI helpers: anonymised display formatting and a thin
wrapper around the audit log so every view/edit of a triage note is
traceable to a role + timestamp (auditability requirement).
"""

from . import database


def mask_name(display_name):
    if not display_name:
        return "(not provided)"
    parts = display_name.strip().split()
    masked = []
    for p in parts:
        if len(p) <= 1:
            masked.append(p)
        else:
            masked.append(p[0] + "*" * (len(p) - 1))
    return " ".join(masked)


def record(actor, action, target_type, target_id, details=None):
    database.log_audit(actor, action, target_type, target_id, details)


def get_recent_audit(limit=200):
    conn = database.get_connection()
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
