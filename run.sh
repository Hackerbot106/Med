#!/usr/bin/env bash
# Quick-start script for the Healthcare Triage Assistant prototype.
set -e
cd "$(dirname "$0")"

if [ ! -f "data/triage.db" ]; then
  echo "No database found - seeding demo data (synthetic only)..."
  python3 seed.py
fi

echo "Starting server on http://localhost:5000 ..."
python3 app.py
