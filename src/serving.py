"""Runtime entrypoint for HTTP serving."""

from __future__ import annotations

import os
from logging import basicConfig as logging_basicConfig

from application import create_application
from config import Settings

settings = Settings()

logging_basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

application = create_application(settings)


def main() -> None:
    """Replace current process with Gunicorn server process."""
    os.execvp(
        "gunicorn",
        [
            "gunicorn",
            "-c",
            "python:gunicorn_configuration",
            "serving:application",
        ],
    )


if __name__ == "__main__":
    main()
