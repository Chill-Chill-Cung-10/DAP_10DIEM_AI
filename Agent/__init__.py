"""Public API — import from here to avoid coupling to internals."""

from Agent.config      import OPENAI_MODEL, PG_DB, PG_HOST, PG_PORT
from Agent.config      import logger
from Agent.db.health   import test_pg_connection
from Agent.graph.builder import build_app
from Agent.utils.llm   import append_chat_history

app = build_app()

__all__ = [
    "app",
    "logger",
    "OPENAI_MODEL",
    "PG_DB",
    "PG_HOST",
    "PG_PORT",
    "test_pg_connection",
    "append_chat_history",
]