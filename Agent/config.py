"""Centralised configuration — all env vars and tuneable constants live here."""

import os
import logging

from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------
# LangSmith tracing (optional)
# ---------------------------
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "DAP-Agent"))
    logger.info("LangSmith tracing enabled (project=%s)", os.environ["LANGCHAIN_PROJECT"])
else:
    logger.info("LangSmith tracing disabled (no LANGCHAIN_API_KEY)")

# ---------------------------
# OpenAI
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ---------------------------
# PostgreSQL
# ---------------------------
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "postgres")
PG_DB   = os.getenv("PG_DB", "demo_db")

# ---------------------------
# RAG thresholds & limits
# ---------------------------
SIMILARITY_THRESHOLD    = 0.45
DIRECT_ANSWER_MIN_VEC   = 0.50
DB_CACHE_TTL_SECONDS    = 300   # 5 minutes
EMBEDDING_CACHE_MAX     = 500