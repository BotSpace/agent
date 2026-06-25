"""`python -m agent.mcp` — standalone MCP server (uvicorn).

Port: AGENT_MCP_PORT (default 8200). `/mcp` path'ida Streamable HTTP.
"""

from __future__ import annotations

import uvicorn

from ..config import get_settings
from ..logging import configure_logging
from .server import create_standalone_app


def main() -> None:
    s = get_settings()
    configure_logging(s.log_level)
    app = create_standalone_app()
    uvicorn.run(app, host=s.host, port=s.mcp_port, log_level=s.log_level.lower())


if __name__ == "__main__":
    main()
