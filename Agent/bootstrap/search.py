"""Hybrid vector + full-text search against pgvector."""

import logging
import time
from typing import List, Dict, Optional

import psycopg2.extras

from Agent.config import SIMILARITY_THRESHOLD, DB_CACHE_TTL_SECONDS
from Agent.db.pool import get_connection, put_connection
from Agent.embedding.model import get_embedding
from Agent.utils.text import normalize_question_text, canonicalize_question_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document:
    """Lightweight document compatible with LangChain conventions."""

    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def pgvector_search(
    query: str,
    k: int = 15,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[Document]:
    """
    Hybrid search: cosine similarity (pgvector) + BM25-style ts_rank.
    Final score = 0.7 * cosine_sim + 0.3 * ts_rank.
    """
    query_embedding = get_embedding(query)
    vec_str = "[" + ",".join(map(str, query_embedding)) + "]"

    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT question,
                   answer,
                   1 - (embedding <=> %s::vector) AS vec_sim,
                   ts_rank(
                       to_tsvector('simple', question || ' ' || answer),
                       plainto_tsquery('simple', %s)
                   ) AS text_rank
            FROM   dapchatbot
            WHERE  1 - (embedding <=> %s::vector) >= %s
            ORDER  BY (
                0.7 * (1 - (embedding <=> %s::vector))
                + 0.3 * ts_rank(
                    to_tsvector('simple', question || ' ' || answer),
                    plainto_tsquery('simple', %s)
                )
            ) DESC
            LIMIT  %s
            """,
            (vec_str, query, vec_str, similarity_threshold, vec_str, query, k),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        put_connection(conn)

    docs: List[Document] = []
    for row in rows:
        combined = 0.7 * float(row["vec_sim"]) + 0.3 * float(row["text_rank"])
        content = f"Câu hỏi: {row['question']}\nTrả lời: {row['answer']}"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "question": row["question"],
                    "answer": row["answer"],
                    "similarity": combined,
                    "vec_sim": float(row["vec_sim"]),
                    "text_rank": float(row["text_rank"]),
                },
            )
        )

    logger.info("Hybrid search returned %d docs for: %s", len(docs), query[:80])
    return docs


# ---------------------------------------------------------------------------
# Exact-match DB lookup (cached)
# ---------------------------------------------------------------------------

_db_question_cache: Optional[List[Dict[str, str]]] = None
_db_question_cache_time: float = 0


def _get_db_question_cache() -> List[Dict[str, str]]:
    """Load and TTL-cache all DB questions with normalised/canonical keys."""
    global _db_question_cache, _db_question_cache_time

    now = time.time()
    if _db_question_cache is not None and (now - _db_question_cache_time) < DB_CACHE_TTL_SECONDS:
        return _db_question_cache

    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT question, answer FROM dapchatbot")
        rows = cur.fetchall()
        cur.close()
    finally:
        put_connection(conn)

    cache = [
        {
            "question":   row["question"],
            "answer":     row["answer"],
            "normalized": normalize_question_text(row["question"]),
            "canonical":  canonicalize_question_text(row["question"]),
        }
        for row in rows
    ]
    _db_question_cache = cache
    _db_question_cache_time = now
    logger.info("DB question cache refreshed: %d entries", len(cache))
    return cache


def exact_db_lookup(question: str) -> Optional[Dict[str, str]]:
    """Return (question, answer) dict on exact normalised/canonical match."""
    try:
        normalized = normalize_question_text(question)
        canonical  = canonicalize_question_text(question)
        for row in _get_db_question_cache():
            if row["normalized"] == normalized or row["canonical"] == canonical:
                return {"question": row["question"], "answer": row["answer"]}
        return None
    except Exception as exc:
        logger.warning("exact_db_lookup failed: %s", exc)
        return None