"""DB health check and one-time index bootstrap."""

import logging

import psycopg2.extras

from Agent.db.pool import get_connection, put_connection

logger = logging.getLogger(__name__)

TABLE_NAME = "dap_embeddings"


def test_pg_connection() -> bool:
    """Check connectivity, ensure pg_trgm extension, and bootstrap HNSW index."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        conn.commit()

        cur.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_name = %s"
            ")",
            (TABLE_NAME,),
        )
        exists = cur.fetchone()[0]

        if exists:
            _ensure_hnsw_index(conn, cur)

        cur.close()
        return exists

    except Exception as exc:
        logger.warning("test_pg_connection failed: %s", exc)
        return False
    finally:
        if conn:
            put_connection(conn)


def _ensure_hnsw_index(conn, cur) -> None:
    """Create HNSW vector index if it does not already exist."""
    cur.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE tablename = %s AND indexdef LIKE '%%hnsw%%'",
        (TABLE_NAME,),
    )
    if cur.fetchone() is not None:
        return  # index already present

    cur.close()
    conn.commit()
    conn.autocommit = True
    idx_cur = conn.cursor()
    try:
        idx_cur.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            f"idx_{TABLE_NAME}_embedding_hnsw "
            f"ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )
        logger.info("HNSW index created on %s", TABLE_NAME)
    except Exception as exc:
        logger.warning("HNSW index creation failed (non-fatal): %s", exc)
    finally:
        idx_cur.close()
        conn.autocommit = False