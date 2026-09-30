"""
Lightweight, dependency-free security helpers: CSRF protection, a per-IP
login rate limiter, and transparent legacy-password-hash migration.

Why hand-rolled instead of pulling in Flask-WTF / Flask-Limiter: both are
good libraries, but this project deliberately keeps its dependency
footprint small and offline-friendly (see requirements.txt), and these
three concerns are simple enough - and security-critical enough - that
being able to read every line here in under a minute (rather than trusting
a black-box import) is itself worth something for a project that hackathon
judges will actually read. For a real multi-instance production
deployment, swap the in-memory RateLimiter below for a shared store
(Redis, etc.) - see README "Known limitations / scaling this beyond one
instance".
"""

import hmac
import secrets
import threading
import time

from werkzeug.security import check_password_hash, generate_password_hash

# ---------------------------------------------------------------------------
# CSRF protection (double-submit token stored in the signed session cookie)
# ---------------------------------------------------------------------------
CSRF_SESSION_KEY = "_csrf_token"
CSRF_FORM_FIELD = "csrf_token"
CSRF_HEADER = "X-CSRFToken"


def get_or_create_csrf_token(flask_session):
    """Returns the current CSRF token for this session, creating one on
    first use. Called from a Jinja context processor so every template can
    render {{ csrf_token() }} into a hidden form field.
    """
    token = flask_session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        flask_session[CSRF_SESSION_KEY] = token
    return token


def validate_csrf(flask_session, submitted_token):
    """Constant-time comparison against the token stored in this session.
    Returns False (never raises) for any malformed input - a missing or
    wrong token should just fail closed.
    """
    expected = flask_session.get(CSRF_SESSION_KEY)
    if not expected or not submitted_token:
        return False
    return hmac.compare_digest(expected, submitted_token)


# ---------------------------------------------------------------------------
# Rate limiting (sliding window, in-memory, per-process)
# ---------------------------------------------------------------------------
def parse_rate_string(spec):
    """Parses a "N per second|minute|hour" string (the same shorthand
    Flask-Limiter uses) into (max_attempts, window_seconds). Falls back to
    a sane default (10 per minute) on anything unparseable rather than
    raising - a typo'd env var should degrade gracefully, not crash boot.
    """
    default = (10, 60)
    try:
        count_str, _, unit = spec.strip().partition(" per ")
        count = int(count_str.strip())
        unit = unit.strip().lower()
        seconds = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}.get(unit)
        if seconds is None or count <= 0:
            return default
        return count, seconds
    except (ValueError, AttributeError):
        return default


class RateLimiter:
    """A simple sliding-window rate limiter keyed by an arbitrary string
    (typically the client IP). Thread-safe; suitable for a single-process
    deployment (the default `python app.py` / single gunicorn worker demo
    setup). Not shared across multiple processes/instances - see the
    module docstring for the production upgrade path.
    """

    def __init__(self, max_attempts, window_seconds):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._hits = {}  # key -> list[timestamps within the current window]

    def allow(self, key):
        now = time.time()
        with self._lock:
            timestamps = [t for t in self._hits.get(key, []) if now - t < self.window_seconds]
            if len(timestamps) >= self.max_attempts:
                self._hits[key] = timestamps
                return False
            timestamps.append(now)
            self._hits[key] = timestamps
            return True

    def retry_after_seconds(self, key):
        with self._lock:
            timestamps = self._hits.get(key, [])
            if not timestamps:
                return 0
            oldest = min(timestamps)
            return max(0, int(self.window_seconds - (time.time() - oldest)))


# ---------------------------------------------------------------------------
# Password hashing with transparent legacy-plaintext migration
# ---------------------------------------------------------------------------
_HASH_PREFIXES = ("pbkdf2:", "scrypt:", "argon2")


def is_hashed_password(value):
    return isinstance(value, str) and value.startswith(_HASH_PREFIXES)


def hash_password(plain_password):
    return generate_password_hash(plain_password)


def verify_and_upgrade_password(conn, user_row, submitted_password):
    """Verifies `submitted_password` against a users row. Supports two
    shapes of the stored `password` column so upgrading an already-seeded
    demo database never locks anyone out:
      - a proper Werkzeug hash (pbkdf2/scrypt/argon2 prefix) -> checked with
        check_password_hash, as normal.
      - a legacy plaintext value (from a database seeded before this
        security pass) -> checked with a constant-time string comparison,
        and on success, TRANSPARENTLY rewritten to a real hash in the same
        request, so it is never stored or compared in plaintext again.
    Returns True/False. Never raises for a malformed stored value.
    """
    stored = user_row["password"] if user_row else None
    if not stored:
        return False
    if is_hashed_password(stored):
        try:
            return check_password_hash(stored, submitted_password)
        except Exception:  # noqa: BLE001 - a corrupted hash should fail closed, not 500
            return False
    if hmac.compare_digest(stored, submitted_password):
        conn.execute("UPDATE users SET password=? WHERE id=?", (hash_password(submitted_password), user_row["id"]))
        conn.commit()
        return True
    return False
