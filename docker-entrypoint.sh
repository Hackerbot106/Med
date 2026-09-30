#!/usr/bin/env bash
# Container entrypoint: seed demo data on first boot (only if no database
# exists yet - never overwrites real data on a restart), then hand off to
# gunicorn. Kept as a tiny shell script rather than logic buried in the
# Dockerfile so it's easy to read, test, and override.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f "data/triage.db" ]; then
  echo "No database found - seeding demo data (synthetic only)..."
  python3 seed.py
fi

echo "Starting Healthcare Triage Assistant via gunicorn on port ${PORT:-5000}..."
exec gunicorn -c gunicorn.conf.py wsgi:app
