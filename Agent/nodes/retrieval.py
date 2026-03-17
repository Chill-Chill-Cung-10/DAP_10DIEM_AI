"""Retrieval nodes: direct answer, query rewriting, RAG lookup."""

import logging
import re

from Agent.config import SIMILARITY_THRESHOLD, DIRECT_ANSWER_MIN_VEC
from Agent.bootstrap.search import pgvector_search, exact_db_lookup, Document
from Agent.graph.state import AgentState
from Agent.utils.llm import safe_llm_invoke, safe_json_loads, docs_context, chat_history_text

logger = logging.getLogger(__name__)

_KNOWLEDGE_KEYWORDS_RE = re.compile(
    r"("
    r"bệnh|nấm|vi khuẩn|virus|triệu chứng|lây lan|ký sinh"
    r"|fungus|bacteria|pathogen|infection|disease|rust|blight|rot|mildew|scab"
    r"|Venturia|Phytophthora|Alternaria|Cercospora|Puccinia|Guignardia"
    r"|thuốc|fungicide|phun|xử lý|phòng ngừa|kháng bệnh"
    r"|cây trồng|táo|nho|ngô|cà chua|khoai tây|đào|ớt|anh đào|cam|chanh|lá bệnh|bề mặt lá|lá cây| cây"
    r"|so sánh|phân tích|giải thích|nguyên nhân|vòng đời"
    r"|giải\s*thích.*điều|phân\s*tích|so\s*sánh|trình\s*bày"
    r"|tại\s*sao|vì\s*sao|nguyên\s*nhân|hậu\s*quả"
    r")",
    re.IGNORECASE,
)


def direct_answer_node(state: AgentState) -> AgentState:
    """
    Try to answer directly from DB (exact match or high-similarity vector).
    Sets direct_quality = 'good' | 'insufficient'.
    """
    query = state.get("rewritten_query") or state["question"]

    # 1. Exact match
    exact_row = exact_db_lookup(query)
    if exact_row:
        logger.info("Direct answer: exact DB match")
        doc = Document(
            page_content=f"Câu hỏi: {exact_row['question']}\nTrả lời: {exact_row['answer']}",
            metadata={
                "question":   exact_row["question"],
                "answer":     exact_row["answer"],
                "similarity": 1.0,
                "vec_sim":    1.0,
                "text_rank":  1.0,
                "match_type": "exact_db",
            },
        )
        return {
            **state,
            "docs":           [doc],
            "answer":         exact_row["answer"],
            "answer_source":  "direct_answer",
            "direct_quality": "good",
        }

    # 2. Vector search
    docs = state.get("docs") or pgvector_search(query, k=5, similarity_threshold=SIMILARITY_THRESHOLD)

    if not docs:
        logger.info("Direct answer: no docs → insufficient")
        return {**state, "docs": [], "direct_quality": "insufficient"}

    max_vec    = max(d.metadata.get("vec_sim", 0) for d in docs)
    max_hybrid = max(d.metadata.get("similarity", 0) for d in docs)
    logger.info("Direct answer: max_vec=%.4f, max_hybrid=%.4f, n=%d", max_vec, max_hybrid, len(docs))

    if max_vec < DIRECT_ANSWER_MIN_VEC:
        logger.info("Direct answer: max_vec=%.3f < %.2f → insufficient", max_vec, DIRECT_ANSWER_MIN_VEC)
        return {**state, "docs": docs, "direct_quality": "insufficient"}

    # 3. Generate answer
    context      = docs_context(docs)
    history_text = chat_history_text(state.get("chat_history") or [])
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        f"Bạn là trợ lý thông minh. Dưới đây là TẤT CẢ các thông tin liên quan "
        f"tìm được từ cơ sở dữ liệu ({len(docs)} kết quả).\n"
        "Hãy TỔNG HỢP tất cả thông tin từ các nguồn liên quan và trả lời "
        "một cách đầy đủ, có cấu trúc bằng tiếng Việt.\n"
        "Nếu có nhiều khía cạnh khác nhau, hãy trình bày từng khía cạnh rõ ràng.\n"
        "Nếu ngữ cảnh không đủ, hãy nói rõ rằng bạn không có đủ thông tin."
        f"{history_block}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    content = safe_llm_invoke(prompt, fallback="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi.")
    logger.info("Direct answer: good (%d docs, %d chars)", len(docs), len(content))
    return {
        **state,
        "docs":           docs,
        "answer":         content,
        "answer_source":  "direct_answer",
        "direct_quality": "good",
    }

def query_rewriter_node(state: AgentState) -> AgentState:
    history_text  = chat_history_text(state.get("chat_history") or [])
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Rewrite the following question into a concise English/Vietnamese search query "
        "suitable for semantic retrieval in a plant disease database. "
        "Focus on: disease name, pathogen, symptoms, crop type, or management method. "
        "Return ONLY JSON.\n\n"
        "Example 1:\n"
        "Question: Bệnh ghẻ táo có những triệu chứng gì trên lá và quả?\n"
        '{"rewritten_query": "apple scab Venturia inaequalis leaf fruit symptoms"}\n\n'
        "Example 2:\n"
        "Question: Làm thế nào để phòng ngừa bệnh mốc sương trên cà chua?\n"
        '{"rewritten_query": "late blight tomato Phytophthora infestans management prevention"}\n\n'
        "Example 3:\n"
        "Question: Nấm nào gây bệnh thối đen trên nho?\n"
        '{"rewritten_query": "black rot grapes Guignardia bidwelli fungus"}\n\n'
        f"{history_block}"
        f"Question: {state['question']}"
    )
    content = safe_llm_invoke(prompt, fallback="")
    data    = safe_json_loads(content, {"rewritten_query": state["question"]})
    logger.info("Rewritten query: %s", data.get("rewritten_query", "")[:80])
    return {**state, **data, "answer_source": "knowledge_pipeline"}


def rag_lookup_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["question"]
    docs  = pgvector_search(query, k=15, similarity_threshold=SIMILARITY_THRESHOLD)
    logger.info("RAG lookup: %d docs", len(docs))
    return {**state, "docs": docs}


def topic_judge_node(state: AgentState) -> AgentState:
    """Deterministic topic classification — no LLM needed."""
    topic = "legal" if _KNOWLEDGE_KEYWORDS_RE.search(state["question"]) else "other"
    logger.info("Topic: %s", topic)
    return {**state, "topic": topic}