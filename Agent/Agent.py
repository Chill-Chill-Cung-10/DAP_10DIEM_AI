import os
import re
import json
import logging
import unicodedata
from collections import OrderedDict
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any

import hashlib

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from FlagEmbedding import BGEM3FlagModel

# ---------------------------
# Logging (quiet by default — use Agent_log.py to see full logs)
# ---------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------
# Config
# ---------------------------
load_dotenv()

# ---------------------------
# LangSmith tracing (optional — set env vars to enable)
# ---------------------------
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "DAP-Agent"))
    logger.info("LangSmith tracing enabled (project=%s)", os.environ["LANGCHAIN_PROJECT"])
else:
    logger.info("LangSmith tracing disabled (no LANGCHAIN_API_KEY)")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "postgres")
PG_DB   = os.getenv("PG_DB", "demo_db")

# ---------------------------
# LLM (OpenAI API)
# ---------------------------
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0.2,
    timeout=60,
)

# ---------------------------
# Embedding model – BGE-M3 (1024-dim)
# ---------------------------
embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# ---------------------------
# PostgreSQL connection pool (lazy init – không crash khi import)
# ---------------------------
_pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _ensure_pool():
    """Tạo pool lần đầu khi thực sự cần, tránh crash lúc import."""
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


def get_pg_connection():
    """Return a pooled psycopg2 connection."""
    _ensure_pool()
    return _pg_pool.getconn()


def put_pg_connection(conn):
    """Return a connection back to the pool."""
    if _pg_pool and not _pg_pool.closed:
        _pg_pool.putconn(conn)


def test_pg_connection() -> bool:
    """Quick connectivity & table check.  Also ensures HNSW index + pg_trgm exist."""
    conn = None
    try:
        conn = get_pg_connection()
        cur = conn.cursor()

        # Ensure pg_trgm extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        conn.commit()

        cur.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_name = 'dapchatbot'"
            ")"
        )
        exists = cur.fetchone()[0]
        if exists:
            # Kiểm tra HNSW index
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'dapchatbot' AND indexdef LIKE '%%hnsw%%'"
            )
            need_index = cur.fetchone() is None
            cur.close()
            # Đóng transaction hiện tại trước khi set autocommit
            conn.commit()
            if need_index:
                conn.autocommit = True
                idx_cur = conn.cursor()
                try:
                    idx_cur.execute(
                        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                        "idx_dapchatbot_embedding_hnsw "
                        "ON dapchatbot USING hnsw (embedding vector_cosine_ops) "
                        "WITH (m = 16, ef_construction = 64)"
                    )
                except Exception:
                    pass  # Index fail không ảnh hưởng hoạt động
                finally:
                    idx_cur.close()
                    conn.autocommit = False
        else:
            cur.close()
        return exists
    except Exception:
        return False
    finally:
        if conn:
            put_pg_connection(conn)


class Document:
    """Lightweight document object compatible with LangChain conventions."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ---------------------------
# Embedding cache (LRU via OrderedDict)
# ---------------------------
_embedding_cache: OrderedDict[str, list] = OrderedDict()
_EMBEDDING_CACHE_MAX = 500


def _get_embedding(text: str) -> list:
    """Return embedding vector with proper LRU cache eviction."""
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _embedding_cache:
        _embedding_cache.move_to_end(key)  # mark as recently used
        return _embedding_cache[key]
    vec = embedding_model.encode([text])["dense_vecs"][0].tolist()
    _embedding_cache[key] = vec
    if len(_embedding_cache) > _EMBEDDING_CACHE_MAX:
        _embedding_cache.popitem(last=False)  # evict least-recently-used
    return vec


def pgvector_search(
    query: str,
    k: int = 15,
    similarity_threshold: float = 0.45,
) -> List[Document]:
    """
    Hybrid search: combines cosine-similarity (pgvector) with BM25-style
    full-text ranking (ts_rank) for better keyword + semantic coverage.

    Final score = 0.7 * cosine_similarity + 0.3 * ts_rank (normalized).
    """
    query_embedding = _get_embedding(query)
    vec_str = "[" + ",".join(map(str, query_embedding)) + "]"

    conn = get_pg_connection()
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
        put_pg_connection(conn)

    docs: List[Document] = []
    for row in rows:
        combined = 0.7 * float(row["vec_sim"]) + 0.3 * float(row["text_rank"])
        content = (
            f"Câu hỏi: {row['question']}\n"
            f"Trả lời: {row['answer']}"
        )
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


def _normalize_question_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"^\s*câu\s*hỏi\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _strip_vietnamese_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


# Pre-compiled regex patterns for canonicalization (performance)
_CANONICAL_PREFIX_PATTERNS = tuple(re.compile(p) for p in (
    r"^cho\s+(toi|minh|em|anh|chi|ad|admin)\s+(hoi|biet)\s+",
    r"^xin\s+(hoi|cho\s+biet|tu\s+van)\s+",
    r"^toi\s+muon\s+(hoi|biet)\s+",
    r"^minh\s+muon\s+(hoi|biet)\s+",
    r"^giai\s+thich\s+cho\s+(toi|minh|em|anh|chi)\s+",
    r"^tu\s+van\s+giu?p\s+(toi|minh|em|anh|chi)\s+",
    r"^cho\s+(toi|minh|em|anh|chi)\s+hoi\s+ve\s+",
    r"^quy\s+dinh\s+(phap\s+luat\s+)?ve\s+",
    r"^theo\s+quy\s+dinh\s+(cua\s+)?phap\s+luat\s+(viet\s+nam\s+)?ve\s+",
    r"^theo\s+luat\s+(viet\s+nam\s+)?ve\s+",
    r"^thong\s+tin\s+ve\s+",
    r"^noi\s+ve\s+",
))

_CANONICAL_SUFFIX_PATTERNS = tuple(re.compile(p) for p in (
    r"\s+la\s+gi\s*$",
    r"\s+la\s+sao\s*$",
    r"\s+nhu\s+the\s+nao\s*$",
    r"\s+ra\s+sao\s*$",
    r"\s+duoc\s+khong\s*$",
    r"\s+khong\s*$",
    r"\s+ntn\s*$",
    r"\s+nhe\s*$",
    r"\s+a\s*$",
    r"\s+ah\s*$",
))

_CANONICAL_NOISE_PATTERNS = tuple(re.compile(p) for p in (
    r"\bphap\s+luat\b",
    r"\bquy\s+dinh\b",
    r"\btai\s+viet\s+nam\b",
    r"\bo\s+viet\s+nam\b",
    r"\bve\s+viec\b",
    r"\blien\s+quan\s+den\b",
    r"\bdoi\s+voi\b",
    r"\btrong\s+truong\s+hop\b",
    r"\bcho\s+biet\b",
    r"\bxin\s+hoi\b",
    r"\btu\s+van\b",
    r"\bgiai\s+thich\b",
))

_CANONICAL_STOPWORDS = {
    "cho", "toi", "minh", "em", "anh", "chi", "biet", "hoi",
    "xin", "tu", "van", "giai", "thich", "giup", "dum", "nhe",
    "a", "ah", "la", "gi", "nhu", "the", "nao", "ra", "sao",
    "duoc", "khong", "ve", "tai", "o", "viet", "nam", "theo",
    "cua", "noi", "thong", "tin", "lien", "quan", "den", "doi",
    "voi", "trong", "truong", "hop", "viec",
}


def _canonicalize_question_text(text: str) -> str:
    text = _normalize_question_text(text)
    text = _strip_vietnamese_accents(text)
    text = text.replace("ntn", "nhu the nao")

    for pattern in _CANONICAL_PREFIX_PATTERNS:
        text = pattern.sub("", text)

    for pattern in _CANONICAL_SUFFIX_PATTERNS:
        text = pattern.sub("", text)

    for pattern in _CANONICAL_NOISE_PATTERNS:
        text = pattern.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = [token for token in text.split() if token not in _CANONICAL_STOPWORDS]
    return " ".join(tokens)


# ---------------------------
# DB Question Key Cache (lazy-loaded, avoids re-computing on every lookup)
# ---------------------------
_db_question_cache: Optional[List[Dict[str, str]]] = None
_db_question_cache_time: float = 0
_DB_CACHE_TTL = 300  # 5 minutes


def _get_db_question_cache() -> List[Dict[str, str]]:
    """Load and cache DB questions with their normalized/canonical keys.
    Cache is refreshed every 5 minutes to pick up DB changes."""
    global _db_question_cache, _db_question_cache_time
    import time
    now = time.time()
    
    if _db_question_cache is not None and (now - _db_question_cache_time) < _DB_CACHE_TTL:
        return _db_question_cache
    
    conn = get_pg_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT question, answer FROM dapchatbot")
        rows = cur.fetchall()
        cur.close()
    finally:
        put_pg_connection(conn)
    
    cache = []
    for row in rows:
        cache.append({
            "question": row["question"],
            "answer": row["answer"],
            "normalized": _normalize_question_text(row["question"]),
            "canonical": _canonicalize_question_text(row["question"]),
        })
    
    _db_question_cache = cache
    _db_question_cache_time = now
    logger.info("DB question cache refreshed: %d entries", len(cache))
    return cache


def exact_db_lookup(question: str) -> Optional[Dict[str, str]]:
    """Fast DB lookup using cached canonical keys."""
    try:
        normalized = _normalize_question_text(question)
        canonical = _canonicalize_question_text(question)
        
        for row in _get_db_question_cache():
            if row["normalized"] == normalized or row["canonical"] == canonical:
                return {"question": row["question"], "answer": row["answer"]}
        return None
    except Exception as e:
        logger.warning("exact_db_lookup failed: %s", e)
        return None


# ---------------------------
# State
# ---------------------------
class AgentState(TypedDict):
    question: str
    chat_history: Optional[List[dict]]  # conversation memory
    intent: Optional[str]
    entities: Optional[List[str]]
    docs: Optional[list]
    rewritten_query: Optional[str]
    relevance: Optional[str]
    topic: Optional[str]
    plan: Optional[List[str]]
    evidence: Optional[List[str]]
    answer: Optional[str]
    answer_source: Optional[str]
    verification: Optional[str]
    retry_count: Optional[int]  # verification retry counter
    direct_quality: Optional[str]  # "good" or "insufficient"


# ---------------------------
# Helpers
# ---------------------------
def safe_json_loads(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Try to extract a JSON object from LLM output (handles markdown fences)."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return default


def _docs_context(state: AgentState, max_chars: int = 6000) -> str:
    """Concatenate page_content of docs with a character budget to avoid
    overflowing the LLM context window."""
    docs = state.get("docs") or []
    parts: List[str] = []
    total = 0
    for d in docs:
        if total + len(d.page_content) > max_chars:
            break
        parts.append(d.page_content)
        total += len(d.page_content)
    return "\n\n".join(parts)


def _chat_history_text(state: AgentState, max_turns: int = 5) -> str:
    """Format recent chat history for prompt injection."""
    history = state.get("chat_history") or []
    recent = history[-max_turns:] if len(history) > max_turns else history
    if not recent:
        return ""
    lines = []
    for turn in recent:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        prefix = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _safe_llm_invoke(prompt: str, fallback: str = "") -> str:
    """Invoke LLM with error handling — returns fallback on failure."""
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return fallback


# ---------------------------
# Nodes
# ---------------------------
# --- Keyword patterns cho greeting / system_info ---
_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"xin\s*chào|chào\s*bạn|chào|hello|hi\b|hey\b"
    r"|cảm\s*ơn|cám\s*ơn|thanks|thank\s*you|tks"
    r"|tạm\s*biệt|bye|goodbye|see\s*you"
    r"|ok\b|okay|ừ|ờ|vâng|dạ"
    r"|tốt|hay\s*lắm|giỏi\s*lắm|great|good"
    r")\s*[!.?~]*\s*$",
    re.IGNORECASE,
)

_SYSTEM_INFO_PATTERNS = re.compile(
    r"("
    r"bạn\s*(là|tên)\s*(ai|gì)|tên\s*(của\s*)?bạn"
    r"|you\s*are\s*who|what.*your\s*name"
    r"|mấy\s*giờ|ngày\s*(bao\s*nhiêu|mấy)|hôm\s*nay\s*ngày"
    r"|what\s*time|what.*date|today"
    r"|bạn\s*có\s*thể\s*làm\s*(gì|được\s*gì)"
    r"|bạn\s*làm\s*được\s*gì"
    r"|chức\s*năng"
    r"|bạn\s*(có|biết|giúp)\s*(những?|được)?\s*(gì|gì\s*không)"
    r")",
    re.IGNORECASE,
)


# Keyword pattern to detect knowledge-heavy / legal queries
_KNOWLEDGE_KEYWORDS = re.compile(
    r"("
    r"luật|điều\s*\d|khoản|nghị\s*định|thông\s*tư|bộ\s*luật|quy\s*định"
    r"|pháp\s*luật|hình\s*sự|dân\s*sự|hành\s*chính|xử\s*phạt"
    r"|bồi\s*thường|hợp\s*đồng|quyền\s*sở\s*hữu|thừa\s*kế"
    r"|giải\s*thích.*điều|phân\s*tích|so\s*sánh|trình\s*bày"
    r"|tại\s*sao|vì\s*sao|nguyên\s*nhân|hậu\s*quả"
    r")",
    re.IGNORECASE,
)


def intent_router(state: AgentState) -> AgentState:
    """
    Routing đơn giản:
      1. Greeting / system_info → keyword
      2. Mọi câu hỏi khác → query (đi qua direct_answer trước,
         nếu không đủ info → fallback sang knowledge_query)
    """
    question = state["question"]
    q_stripped = unicodedata.normalize("NFC", question.strip())

    # --- Greeting / System info ---
    if _GREETING_PATTERNS.match(q_stripped):
        logger.info("Router → greeting")
        return {**state, "intent": "greeting", "entities": []}

    if _SYSTEM_INFO_PATTERNS.search(q_stripped):
        logger.info("Router → direct_system_info")
        return {**state, "intent": "direct_system_info", "entities": []}

    # --- Mọi câu hỏi khác → thử direct_answer trước ---
    logger.info("Router → query (unified path)")
    return {**state, "intent": "query", "entities": []}


def system_info_node(state: AgentState) -> AgentState:
    q = state["question"].strip().lower()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if re.search(r"tên.*bạn|bạn.*tên|bạn\s*là\s*ai|your\s*name|who\s*are", q):
        answer = (
            "Tôi là **ALO** — trợ lý AI chuyên hỗ trợ tra cứu kiến thức. "
            "Bạn có thể hỏi tôi bất cứ điều gì! 😊"
        )
    elif re.search(r"làm\s*(gì|được)|khả\s*năng|chức\s*năng|capabilit|bạn\s*(có|biết|giúp).*gì", q):
        answer = (
            "Tôi có thể giúp bạn:\n"
            "• Tra cứu thông tin từ cơ sở dữ liệu\n"
            "• Trả lời câu hỏi kiến thức\n"
            "• Phân tích và tổng hợp thông tin\n"
            "Hãy hỏi tôi bất cứ điều gì! 😊"
        )
    else:
        answer = f"Bây giờ là {now}. Tôi có thể giúp gì thêm cho bạn?"
    return {**state, "answer": answer}


def query_rewriter(state: AgentState) -> AgentState:
    logger.info("Entering knowledge pipeline via query_rewriter")
    history_text = _chat_history_text(state)
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Rewrite the following question into a concise Vietnamese search query "
        "suitable for semantic retrieval.  Return ONLY JSON.\n\n"
        "Example 1:\n"
        "Question: Theo quy định pháp luật hiện hành, người lao động có quyền gì khi bị sa thải trái phép?\n"
        '{"rewritten_query": "quyền người lao động bị sa thải trái pháp luật"}\n\n'
        "Example 2:\n"
        "Question: Hợp đồng lao động là gì?\n"
        '{"rewritten_query": "khái niệm hợp đồng lao động"}\n\n'
        f"{history_block}"
        f"Question: {state['question']}"
    )
    content = _safe_llm_invoke(prompt, fallback="")
    data = safe_json_loads(content, {"rewritten_query": state["question"]})
    logger.info("Rewritten query: %s", data.get("rewritten_query", "")[:80])
    return {**state, **data, "answer_source": "knowledge_pipeline"}


def rag_lookup(state: AgentState) -> AgentState:
    """Retrieve relevant docs via hybrid search (pgvector + BM25)."""
    query = state.get("rewritten_query") or state["question"]
    docs = pgvector_search(query, k=15, similarity_threshold=0.45)
    logger.info("RAG lookup returned %d docs", len(docs))
    return {**state, "docs": docs}


def topic_judge(state: AgentState) -> AgentState:
    """Deterministic topic classification — no LLM call needed.
    Questions routed here already matched _KNOWLEDGE_KEYWORDS (legal terms),
    so they are legal by definition."""
    topic = "legal" if _KNOWLEDGE_KEYWORDS.search(state["question"]) else "other"
    logger.info("Topic: %s (deterministic)", topic)
    return {**state, "topic": topic}


def answer_draft(state: AgentState) -> AgentState:
    context = _docs_context(state)
    history_text = _chat_history_text(state)
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Bạn là trợ lý pháp luật Việt Nam. Dựa vào ngữ cảnh được cung cấp, "
        "hãy trả lời câu hỏi một cách ngắn gọn, chính xác bằng tiếng Việt.\n"
        "Nếu ngữ cảnh không đủ, hãy nói rõ rằng bạn không có đủ thông tin."
        f"{history_block}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    content = _safe_llm_invoke(prompt, fallback="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi.")
    logger.info("Answer draft generated (%d chars)", len(content))
    return {**state, "answer": content}


def answer_verifier(state: AgentState) -> AgentState:
    prompt = (
        "Verify whether the answer below is strong, well-supported and accurate.\n"
        'Return ONLY JSON: {"verification": "good"} or {"verification": "weak"}.\n\n'
        "Example 1:\n"
        "Question: Thời hiệu khởi kiện vụ án dân sự?\n"
        "Answer: Theo Bộ luật Tố tụng dân sự 2015, thời hiệu khởi kiện vụ án dân sự là 03 năm kể từ ngày biết.\n"
        '{"verification": "good"}\n\n'
        "Example 2:\n"
        "Question: Mức phạt vi phạm giao thông?\n"
        "Answer: Tùy trường hợp.\n"
        '{"verification": "weak"}\n\n'
        f"Question: {state['question']}\nAnswer: {state.get('answer')}"
    )
    content = _safe_llm_invoke(prompt, fallback='{"verification": "good"}')
    data = safe_json_loads(content, {"verification": "good"})
    # Track retry count
    retry = (state.get("retry_count") or 0)
    if data.get("verification") == "weak":
        retry += 1
    logger.info("Verification: %s (retry=%d)", data.get("verification"), retry)
    return {**state, **data, "retry_count": retry}


def clarify_question(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": "Xin lỗi, tôi chưa chắc chắn về câu trả lời. "
                  "Bạn có thể làm rõ câu hỏi thêm được không?",
    }


def answer_node(state: AgentState) -> AgentState:
    if state.get("answer"):
        return state
    return answer_draft(state)


def greeting_node(state: AgentState) -> AgentState:
    """Phản hồi linh hoạt theo loại greeting."""
    q = state["question"].strip().lower()

    if re.search(r"cảm\s*ơn|cám\s*ơn|thanks|thank", q):
        answer = "Không có gì! 😊 Nếu bạn cần hỏi thêm điều gì, cứ hỏi nhé."
    elif re.search(r"tạm\s*biệt|bye|goodbye", q):
        answer = "Tạm biệt bạn! 👋 Hẹn gặp lại."
    elif re.search(r"ok\b|okay|ừ|ờ|vâng|dạ", q):
        answer = "Vâng! Bạn cần hỏi thêm gì không? 😊"
    elif re.search(r"tốt|hay\s*lắm|giỏi|great|good", q):
        answer = "Cảm ơn bạn! 😄 Tôi luôn sẵn sàng hỗ trợ."
    else:
        answer = "Xin chào! 👋 Tôi là trợ lý DAP — có thể giúp gì cho bạn?"

    return {**state, "answer": answer}


def direct_answer_node(state: AgentState) -> AgentState:
    """RAG-based answer: retrieve docs → check relevance → respond.
    Sets 'direct_quality' = 'good' nếu tìm được docs liên quan,
    'insufficient' nếu không đủ info → fallback sang knowledge pipeline."""
    query = state.get("rewritten_query") or state["question"]

    exact_row = exact_db_lookup(query)
    if exact_row:
        logger.info("Direct answer: exact DB match found")
        doc = Document(
            page_content=(
                f"Câu hỏi: {exact_row['question']}\n"
                f"Trả lời: {exact_row['answer']}"
            ),
            metadata={
                "question": exact_row["question"],
                "answer": exact_row["answer"],
                "similarity": 1.0,
                "vec_sim": 1.0,
                "text_rank": 1.0,
                "match_type": "exact_db",
            },
        )
        return {
            **state,
            "docs": [doc],
            "answer": exact_row["answer"],
            "answer_source": "direct_answer",
            "direct_quality": "good",
        }

    docs = state.get("docs") or pgvector_search(
        query, k=5, similarity_threshold=0.45,
    )

    # Check relevance: nếu không có docs hoặc similarity thấp → insufficient
    if not docs:
        logger.info("Direct answer: no docs found → insufficient")
        return {**state, "docs": [], "direct_quality": "insufficient"}

    max_vec = max(d.metadata.get("vec_sim", 0) for d in docs)
    max_hybrid = max(d.metadata.get("similarity", 0) for d in docs)
    logger.info("Direct answer: max_vec=%.4f, max_hybrid=%.4f, n_docs=%d", max_vec, max_hybrid, len(docs))
    if max_vec < 0.5:
        logger.info("Direct answer: max_vec=%.3f < 0.50 → insufficient", max_vec)
        return {**state, "docs": docs, "direct_quality": "insufficient"}

    # Đủ info → generate answer
    state_with_docs = {**state, "docs": docs}
    n_found = len(docs)
    context = _docs_context(state_with_docs)
    history_text = _chat_history_text(state)
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Bạn là trợ lý thông minh. Dưới đây là TẤT CẢ các thông tin liên quan "
        f"tìm được từ cơ sở dữ liệu ({n_found} kết quả).\n"
        "Hãy TỔNG HỢP tất cả thông tin từ các nguồn liên quan và trả lời "
        "một cách đầy đủ, có cấu trúc bằng tiếng Việt.\n"
        "Nếu có nhiều khía cạnh khác nhau, hãy trình bày từng khía cạnh rõ ràng.\n"
        "Nếu ngữ cảnh không đủ, hãy nói rõ rằng bạn không có đủ thông tin."
        f"{history_block}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    content = _safe_llm_invoke(prompt, fallback="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi.")
    logger.info("Direct answer: good (max_vec=%.3f, %d docs, %d chars)", max_vec, n_found, len(content))
    return {
        **state,
        "docs": docs,
        "answer": content,
        "answer_source": "direct_answer",
        "direct_quality": "good",
    }


# ---------------------------
# Evidence Subgraph
# ---------------------------
def search_planner(state: AgentState) -> AgentState:
    prompt = (
        "Plan a short retrieval strategy for answering this legal question.\n"
        'Return ONLY JSON: {"plan": ["step1", "step2", ...]}.\n\n'
        "Example:\n"
        "Question: So sánh Nghị định 15 và 117 về xử phạt vi phạm an toàn thực phẩm\n"
        '{"plan": ["quy định xử phạt Nghị định 15 an toàn thực phẩm", '
        '"quy định xử phạt Nghị định 117 an toàn thực phẩm", '
        '"so sánh mức phạt an toàn thực phẩm"]}\n\n'
        f"Question: {state['question']}"
    )
    content = _safe_llm_invoke(prompt, fallback='{"plan": []}')
    data = safe_json_loads(content, {"plan": []})
    logger.info("Search plan: %d steps", len(data.get("plan", [])))
    return {**state, **data}


def multi_source_retrieval(state: AgentState) -> AgentState:
    """Retrieve docs using plan steps as additional queries, merging unique results."""
    plan = state.get("plan") or []
    all_docs: List[Document] = list(state.get("docs") or [])
    seen = {d.metadata.get("question", "") for d in all_docs}

    # Build query list: original question + each plan step
    queries = [state["question"]] + [s for s in plan if isinstance(s, str)]

    for query in queries:
        new_docs = pgvector_search(query, k=10, similarity_threshold=0.40)
        for d in new_docs:
            q = d.metadata.get("question", "")
            if q not in seen:
                seen.add(q)
                all_docs.append(d)

    return {**state, "docs": all_docs}


def source_ranker(state: AgentState) -> AgentState:
    """Sort docs by similarity."""
    docs = state.get("docs") or []
    docs.sort(key=lambda d: d.metadata.get("similarity", 0), reverse=True)
    return {**state, "docs": docs}


def evidence_extractor(state: AgentState) -> AgentState:
    context = _docs_context(state)
    prompt = (
        "Extract the key evidence snippets from the context below.\n"
        'Return ONLY JSON: {"evidence": ["snippet1", "snippet2", ...]}.\n\n'
        "Example:\n"
        "Context: Câu hỏi: Thời hạn hợp đồng lao động?\n"
        "Trả lời: Hợp đồng lao động có 2 loại: xác định thời hạn (không quá 36 tháng) và không xác định thời hạn.\n"
        '{"evidence": ["Hợp đồng xác định thời hạn: không quá 36 tháng", "Hợp đồng không xác định thời hạn"]}\n\n'
        f"Context:\n{context}"
    )
    content = _safe_llm_invoke(prompt, fallback='{"evidence": []}')
    data = safe_json_loads(content, {"evidence": []})
    logger.info("Extracted %d evidence snippets", len(data.get("evidence", [])))
    return {**state, **data}


def conclusion_builder(state: AgentState) -> AgentState:
    """Tổng hợp evidence, đưa ra kết luận VÀ tự kiểm chứng (gộp claim_verifier)."""
    evidence = "\n".join(state.get("evidence") or [])
    history_text = _chat_history_text(state)
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Bạn là trợ lý pháp luật Việt Nam. Hãy thực hiện 2 bước:\n"
        "1. Kiểm chứng các bằng chứng: loại bỏ thông tin mâu thuẫn hoặc không đáng tin.\n"
        "2. Tổng hợp bằng chứng đáng tin và đưa ra câu trả lời chính xác bằng tiếng Việt."
        f"{history_block}\n\n"
        f"Bằng chứng:\n{evidence}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    content = _safe_llm_invoke(prompt, fallback="Xin lỗi, đã xảy ra lỗi khi tổng hợp kết luận.")
    logger.info("Conclusion built (%d chars)", len(content))
    return {**state, "answer": content}


# ---------------------------
# Routing functions
# ---------------------------
def route_intent(state: AgentState):
    return state.get("intent", "query")


def route_direct_quality(state: AgentState):
    """Sau direct_answer: nếu đủ info → end, nếu không → knowledge pipeline."""
    decision = state.get("direct_quality")
    logger.info("Route after direct_answer: direct_quality=%s", decision)
    if decision == "good":
        return "good"
    return "insufficient"


def route_topic(state: AgentState):
    return state.get("topic", "other")


def route_verification(state: AgentState):
    if state.get("verification") == "good":
        return "good"
    if (state.get("retry_count") or 0) > 1:
        return "weak"
    return "retry"


# ---------------------------
# Build LangGraph
# ---------------------------
graph = StateGraph(AgentState)

graph.add_node("router",           intent_router)
graph.add_node("system_info",      system_info_node)
graph.add_node("direct_answer",    direct_answer_node)
graph.add_node("query_rewriter",   query_rewriter)
graph.add_node("rag",              rag_lookup)
# relevance_judge removed (output was unused)
graph.add_node("topic_judge",      topic_judge)
graph.add_node("answer_draft",     answer_draft)
graph.add_node("answer_verifier",  answer_verifier)
graph.add_node("clarify_question", clarify_question)
graph.add_node("answer",           answer_node)
graph.add_node("greeting",         greeting_node)

graph.add_node("search_planner",          search_planner)
graph.add_node("multi_source_retrieval",  multi_source_retrieval)
graph.add_node("source_ranker",           source_ranker)
graph.add_node("evidence_extractor",      evidence_extractor)
graph.add_node("conclusion_builder",      conclusion_builder)

graph.set_entry_point("router")

# Router: greeting / system_info / query (unified)
graph.add_conditional_edges("router", route_intent, {
    "greeting":           "greeting",
    "direct_system_info": "system_info",
    "query":              "direct_answer",
})

# Direct answer → good → END, insufficient → knowledge pipeline
graph.add_conditional_edges("direct_answer", route_direct_quality, {
    "good":         END,
    "insufficient": "query_rewriter",
})

# Knowledge pipeline
graph.add_edge("system_info",    "answer")
graph.add_edge("query_rewriter", "rag")
graph.add_edge("rag",            "topic_judge")

graph.add_conditional_edges("topic_judge", route_topic, {
    "legal": "search_planner",
    "other": "answer_draft",
})

graph.add_edge("search_planner",         "multi_source_retrieval")
graph.add_edge("multi_source_retrieval",  "source_ranker")
graph.add_edge("source_ranker",           "evidence_extractor")
graph.add_edge("evidence_extractor",      "conclusion_builder")
graph.add_edge("conclusion_builder",      "answer_verifier")

graph.add_edge("answer_draft", "answer")

graph.add_conditional_edges("answer_verifier", route_verification, {
    "weak":  "clarify_question",
    "good":  "answer",
    "retry": "rag",
})

graph.add_edge("clarify_question", END)
graph.add_edge("greeting",        END)
graph.add_edge("answer",          END)

app = graph.compile()

# ---------------------------
# Interactive run loop
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  LangGraph Agent  –  OpenAI + pgvector RAG")
    print("  DB: PostgreSQL @ %s:%s/%s" % (PG_HOST, PG_PORT, PG_DB))
    print("  LLM: %s (OpenAI)" % OPENAI_MODEL)
    print("=" * 60)

    if not test_pg_connection():
        print(
            "\n⚠️  Could not verify the 'dapchatbot' table.\n"
            "    Make sure Docker is running (docker compose up -d)\n"
            "    and embeddings/embedd.py has been executed.\n"
        )

    print("\nType your question (or 'exit' to quit):\n")

    chat_history: List[dict] = []

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() in ("exit", "quit"):
            break

        try:
            out = app.invoke({"question": q, "chat_history": chat_history})
            answer = out.get("answer", "(no answer)")
            print("\nAgent:", answer, "\n")

            # Update conversation memory
            chat_history.append({"role": "user", "content": q})
            chat_history.append({"role": "assistant", "content": answer})
            # Keep last 10 turns to avoid unbounded growth
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            print(f"\n❌ Error: {exc}\n")
