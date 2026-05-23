"""ASGI entrypoint — `uvicorn agent.main:app`."""

from .api import create_app

app = create_app()
