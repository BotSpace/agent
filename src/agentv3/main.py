"""ASGI entrypoint — `uvicorn agentv3.main:app`."""

from .api import create_app

app = create_app()
