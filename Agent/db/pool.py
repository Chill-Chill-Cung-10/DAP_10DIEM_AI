"""PostgreSQL connection pool management."""

import logging
from typing import Optional

import psycopg2
import psycopg2.pool

from Agent.config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_DB

logger = logging.getLogger(__name__)

_pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _ensure_pool() -> None:
    """Lazy-init pool on first use — avoids crash at import time."""
    global _pg_pool
    if _pg_pool is None or _pg_pool.closed:
        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASS,
            database=PG_DB,
        )
        logger.info("PostgreSQL pool created (%s:%s/%s)", PG_HOST, PG_PORT, PG_DB)


def get_connection():
    """Return a pooled connection."""
    _ensure_pool()
    return _pg_pool.getconn()


def put_connection(conn) -> None:
    """Return a connection back to the pool."""
    if _pg_pool and not _pg_pool.closed:
        _pg_pool.putconn(conn)