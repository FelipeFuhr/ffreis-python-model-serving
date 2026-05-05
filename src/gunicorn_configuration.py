"""Gunicorn runtime configuration derived from environment settings."""

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
