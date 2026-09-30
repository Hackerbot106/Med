"""
Gunicorn configuration for production deployment. Run with:

    gunicorn -c gunicorn.conf.py wsgi:app

See wsgi.py's docstring for why worker count is kept low and the timeout
kept high for this specific app (each worker loads its own local AI model).
"""

import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Deliberately conservative default: this app is CPU-bound (both request
# handling and local-LLM inference share the same CPU budget), and each
# worker process loads its own full copy of the ~1GB quantized model into
# memory. min(cpu_count, 4) balances throughput against RAM use for a
# typical single PHC/clinic-server deployment; override via WEB_CONCURRENCY
# for a bigger box.
workers = int(os.environ.get("WEB_CONCURRENCY", min(multiprocessing.cpu_count(), 4)))

# threads > 1 lets a worker keep serving fast, non-AI routes (dashboard,
# audit log, static assets) while another request on the same worker is
# blocked inside a slow CPU-bound model.create_chat_completion() call, since
# llama-cpp-python releases the GIL during the C++ inference call.
threads = int(os.environ.get("GUNICORN_THREADS", "2"))

worker_class = "gthread"

# CPU-only local-model inference can legitimately take several seconds per
# request on modest hardware - the gunicorn default (30s) is too tight and
# would kill and restart a worker mid-inference under load.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

accesslog = "-"   # stdout - captured by the container/orchestrator's log driver
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# Restart a worker after N requests as a defense against slow memory growth
# in long-running native (llama.cpp) extension code - jitter avoids all
# workers recycling at the same instant.
max_requests = 500
max_requests_jitter = 50

preload_app = False  # each worker performs its own model preload; see wsgi.py
