"""
Production WSGI entrypoint.

`python app.py` (via Flask's built-in dev server) is fine for a hackathon
demo but is explicitly documented by Flask as unsuitable for anything else
(single-threaded-ish, no process management, no graceful reload). For a
real deployment, point a production WSGI server at this module instead:

    gunicorn -c gunicorn.conf.py wsgi:app
    # or, without the config file:
    gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 wsgi:app

`app.py` already initializes the database and kicks off the local AI
model's background preload at import time (not only inside
`if __name__ == "__main__":`), so importing `wsgi:app` here does the right
thing automatically.

Note on worker count: keep this LOW (2-4) for this app specifically. Each
gunicorn worker process loads its OWN ~1GB copy of the local AI model into
memory (llama.cpp state isn't shareable across processes), so worker count
directly multiplies RAM use - a `--preload` flag would share the *pre-fork*
memory pages via copy-on-write, but llama.cpp's internal buffers are
written to during inference regardless, so plan RAM as (worker_count x
~1.5-2GB) rather than assuming a single shared copy. `--timeout 120` (or
higher) matters because CPU-only inference on a 1.5B model can take several
seconds per request - gunicorn's default 30s worker timeout is too tight.
"""

from app import app  # noqa: F401  (imported for its side effects: DB init + AI preload)
