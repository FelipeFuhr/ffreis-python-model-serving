"""Gunicorn runtime configuration derived from environment settings."""

# pylint: disable=invalid-name
# scan-fix(pylint:C0103): gunicorn config variables follow gunicorn's own naming
# convention (lowercase module globals like `bind`, `workers`, `errorlog`) — they
# are not Python constants and UPPER_CASE would break gunicorn's config loader.

from config import Settings

settings = Settings()

bind = f"0.0.0.0:{settings.port}"
workers = settings.gunicorn_workers
threads = settings.gunicorn_threads
timeout = settings.gunicorn_timeout
graceful_timeout = settings.gunicorn_graceful_timeout
keepalive = settings.gunicorn_keepalive

worker_class = "uvicorn.workers.UvicornWorker"
accesslog = "-"
errorlog = "-"
