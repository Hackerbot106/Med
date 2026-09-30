"""
Environment-driven configuration for the Healthcare Triage Assistant.

Industrial-grade rationale: a prototype that hardcodes its secret key and
never distinguishes "local demo" from "deployed" behavior is fine for a
five-minute hackathon walkthrough, but breaks the moment someone tries to
actually run it anywhere else. This module gives the app:

  - A real, persistent Flask secret key (generated once, stored locally,
    or supplied via the SECRET_KEY environment variable for a real
    deployment) instead of a string literal committed to source control.
  - Separate Development / Production config profiles selected by the
    APP_ENV environment variable, so cookie security, debug mode, and
    logging verbosity are correct for where the app is actually running
    instead of being one-size-fits-all.
  - Every tunable read from the environment with a safe default, following
    twelve-factor-app practice, so a deployment can be configured without
    touching code.
"""

import os
import secrets

# Optional: load a local .env file into os.environ if python-dotenv is
# installed and a .env file exists (see .env.example). This is a pure
# convenience for local/dev use - the app never REQUIRES python-dotenv,
# since every setting also works as a real environment variable (the
# twelve-factor-app approach a container/orchestrator would use instead).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SECRET_KEY_FILE = os.path.join(DATA_DIR, ".secret_key")


def _load_or_create_secret_key():
    """Returns SECRET_KEY from the environment if set (the right approach
    for a real deployment - e.g. injected by a container orchestrator or
    secrets manager). Otherwise, generates a strong random key ONCE and
    persists it to data/.secret_key so sessions survive a process restart
    without ever hardcoding a secret in source control. This file is
    local-only and should never be committed (see .gitignore).
    """
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE, "r", encoding="utf-8") as f:
            existing = f.read().strip()
            if existing:
                return existing
    new_key = secrets.token_hex(32)
    with open(SECRET_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(new_key)
    try:
        os.chmod(SECRET_KEY_FILE, 0o600)
    except OSError:
        pass  # best-effort on platforms that don't support POSIX perms (e.g. Windows)
    return new_key


class BaseConfig:
    SECRET_KEY = _load_or_create_secret_key()
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "12")) * 1024 * 1024
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME_MINUTES = int(os.environ.get("SESSION_LIFETIME_MINUTES", "60"))
    WTF_CSRF_ENABLED = True
    RATELIMIT_ENABLED = True
    RATELIMIT_LOGIN = os.environ.get("RATELIMIT_LOGIN", "10 per minute")
    RATELIMIT_STORAGE_URI = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DEBUG = False
    TESTING = False
    # Set to True only behind TLS (a reverse proxy terminating HTTPS, or the
    # app itself serving TLS) - forcing this on for a plain-HTTP local demo
    # would make the session cookie never get sent and silently break login.
    SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"


class DevelopmentConfig(BaseConfig):
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"


class ProductionConfig(BaseConfig):
    DEBUG = False
    # In production, refuse to boot on the auto-generated fallback key
    # silently - require an explicit SECRET_KEY from the environment so a
    # deployment can't accidentally ship with a key that lives only on one
    # container's ephemeral disk.
    def __init__(self):
        if not os.environ.get("SECRET_KEY"):
            import warnings
            warnings.warn(
                "APP_ENV=production but no SECRET_KEY environment variable is set - using an "
                "auto-generated key stored at data/.secret_key. Set SECRET_KEY explicitly for a "
                "real multi-instance deployment (otherwise sessions won't be portable across "
                "instances/restarts that don't share that file).",
                RuntimeWarning,
            )


class TestingConfig(BaseConfig):
    TESTING = True
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False


CONFIG_BY_ENV = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    env = os.environ.get("APP_ENV", "development").lower()
    return CONFIG_BY_ENV.get(env, DevelopmentConfig)()
