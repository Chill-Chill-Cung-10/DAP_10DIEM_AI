"""Intent-aware retrieval and reasoning nodes."""

import logging
import re
from datetime import datetime, timedelta, timezone

from Agent.config import (
    SIMILARITY_THRESHOLD,
    K_HYBRID_DEFAULT,
    K_DIAGNOSIS,
    K_COMPARE_PER_SIDE,
    K_RECENT_CHECK,
)
from Agent.bootstrap.search import pgvector_search
from Agent.graph.state import AgentState
from Agent.utils.llm import safe_llm_invoke, docs_context, chat_history_text

logger = logging.getLogger(__name__)


def _history_block(state: AgentState) -> str:
    history_text = chat_history_text(state.get("chat_history") or [])
    return f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""


def _ensure_docs(state: AgentState, query: str, k: int = K_RECENT_CHECK) -> list:
    docs = state.get("docs") or []
    if docs:
        return docs
    return pgvector_search(query, k=k, similarity_threshold=SIMILARITY_THRESHOLD)


def _doc_created_at(metadata_value: str):
    if not metadata_value:
        return None
    try:
        # Handles values like "2026-03-19 10:22:33+00:00"
        text = metadata_value.replace(" ", "T", 1)
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def logical_reasoning_node(state: AgentState) -> AgentState:
    """Direct reasoning for logic questions without retrieval."""
    history_block = _history_block(state)
    prompt = (
        "Bạn là trợ lý tư duy logic. Trả lời trực tiếp dựa trên lập luận, "
        "không gọi tìm kiếm dữ liệu và không viện dẫn cơ sở dữ liệu nội bộ.\n"
        "Nếu đề bài thiếu giả định, hãy nêu rõ giả định hợp lý trước khi kết luận.\n"
        "Trả lời ngắn gọn, mạch lạc bằng tiếng Việt."
        f"{history_block}\n"
        f"Câu hỏi: {state['question']}"
    )
    answer = safe_llm_invoke(prompt, fallback="Xin lỗi, tôi chưa thể suy luận câu này lúc này.")
    return {**state, "answer": answer, "answer_source": "logical_reasoning", "docs": []}


def diagnosis_retrieval_node(state: AgentState) -> AgentState:
    """Symptom-focused retrieval for diagnosis questions."""
    query = state["question"]
    diagnosis_query = f"{query} triệu chứng chẩn đoán tác nhân quản lý"
    docs = pgvector_search(diagnosis_query, k=K_DIAGNOSIS, similarity_threshold=SIMILARITY_THRESHOLD)
    logger.info("Diagnosis retrieval: %d docs", len(docs))
    return {**state, "docs": docs, "rewritten_query": diagnosis_query}


def diagnosis_reasoning_node(state: AgentState) -> AgentState:
    context = docs_context(state.get("docs") or [])
    history_block = _history_block(state)
    prompt = (
        "Bạn là chuyên gia chẩn đoán bệnh cây trồng. "
        "Dựa vào triệu chứng và ngữ cảnh, hãy đưa ra chẩn đoán theo cấu trúc:\n"
        "1) Khả năng cao nhất\n"
        "2) Chẩn đoán phân biệt\n"
        "3) Lý do nhận định\n"
        "4) Bước xác nhận tại hiện trường\n"
        "5) Hướng xử lý ban đầu\n"
        "Nếu dữ liệu chưa đủ chắc chắn, phải ghi rõ mức độ chắc chắn thấp."
        f"{history_block}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    answer = safe_llm_invoke(prompt, fallback="Xin lỗi, tôi chưa đủ dữ liệu để chẩn đoán chắc chắn.")
    return {**state, "answer": answer, "answer_source": "diagnosis_reasoning"}


def compare_split_queries_node(state: AgentState) -> AgentState:
    """Split comparison query into two targets when possible."""
    targets = [s for s in (state.get("sub_queries") or []) if isinstance(s, str) and s.strip()]
    if len(targets) >= 2:
        return {**state, "sub_queries": targets[:2]}

    q = state["question"]
    m = re.search(r"giữa\s+(.+?)\s+và\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return {**state, "sub_queries": [m.group(1).strip(), m.group(2).strip()]}

    # Fallback: compare same query with two aspects
    return {**state, "sub_queries": [q, f"{q} management"]}


def compare_retrieval_node(state: AgentState) -> AgentState:
    """Dual retrieval for compare flow."""
    queries = state.get("sub_queries") or [state["question"]]
    merged = []
    seen = set()

    for idx, query in enumerate(queries[:2], start=1):
        docs = pgvector_search(query, k=K_COMPARE_PER_SIDE, similarity_threshold=SIMILARITY_THRESHOLD)
        for d in docs:
            key = d.metadata.get("pk_id")
            if key in seen:
                continue
            seen.add(key)
            d.metadata["compare_side"] = idx
            d.metadata["compare_query"] = query
            merged.append(d)

    logger.info("Compare retrieval: %d docs", len(merged))
    return {**state, "docs": merged}


def compare_synthesis_node(state: AgentState) -> AgentState:
    context = docs_context(state.get("docs") or [])
    queries = state.get("sub_queries") or []
    left = queries[0] if len(queries) > 0 else "Đối tượng A"
    right = queries[1] if len(queries) > 1 else "Đối tượng B"
    history_block = _history_block(state)

    prompt = (
        "Bạn là chuyên gia bệnh học thực vật. Hãy so sánh có cấu trúc theo bảng ý:\n"
        "1) Tác nhân gây bệnh\n"
        "2) Triệu chứng điển hình\n"
        "3) Điều kiện phát sinh\n"
        "4) Mức độ rủi ro\n"
        "5) Biện pháp quản lý\n"
        "6) Kết luận khác biệt chính\n"
        "Nêu rõ phần nào chưa đủ bằng chứng nếu dữ liệu thiếu."
        f"{history_block}\n\n"
        f"Đối tượng so sánh A: {left}\n"
        f"Đối tượng so sánh B: {right}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi người dùng: {state['question']}"
    )
    answer = safe_llm_invoke(prompt, fallback="Xin lỗi, tôi chưa đủ dữ liệu để so sánh rõ ràng.")
    return {**state, "answer": answer, "answer_source": "compare_synthesis"}


def recent_freshness_check_node(state: AgentState) -> AgentState:
    """Check if retrieved docs are fresh enough for recent-information queries."""
    docs = _ensure_docs(state, state["question"], k=K_RECENT_CHECK)
    now = datetime.now(timezone.utc)
    freshness_window = timedelta(days=540)  # ~18 months

    newest = None
    for d in docs:
        created = _doc_created_at(d.metadata.get("created_at", ""))
        if created is None:
            continue
        if newest is None or created > newest:
            newest = created

    if newest is None:
        status = "stale"
    else:
        if newest.tzinfo is None:
            newest = newest.replace(tzinfo=timezone.utc)
        status = "fresh" if (now - newest) <= freshness_window else "stale"

    logger.info("Freshness check: %s (docs=%d)", status, len(docs))
    return {**state, "docs": docs, "freshness_status": status}


def recent_fallback_node(state: AgentState) -> AgentState:
    """Fallback response when no fresh data is available."""
    context = docs_context(state.get("docs") or [])
    history_block = _history_block(state)
    prompt = (
        "Bạn đang xử lý câu hỏi cần thông tin mới. "
        "Dữ liệu hiện có có thể không còn cập nhật.\n"
        "Hãy trả lời theo format:\n"
        "1) Cảnh báo độ mới dữ liệu\n"
        "2) Thông tin tốt nhất hiện có từ ngữ cảnh\n"
        "3) Khuyến nghị người dùng cần kiểm chứng nguồn mới hơn\n"
        "Giữ câu trả lời trung thực, không bịa nguồn mới."
        f"{history_block}\n\n"
        f"Ngữ cảnh hiện có:\n{context}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    answer = safe_llm_invoke(prompt, fallback="Dữ liệu hiện có chưa đủ mới để trả lời chắc chắn.")
    return {**state, "answer": answer, "answer_source": "recent_fallback"}


def hybrid_search_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["question"]
    docs = pgvector_search(query, k=K_HYBRID_DEFAULT, similarity_threshold=SIMILARITY_THRESHOLD)
    logger.info("Hybrid search: %d docs", len(docs))
    return {**state, "docs": docs}


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (existing imports may still reference these)
# ---------------------------------------------------------------------------

def direct_answer_node(state: AgentState) -> AgentState:
    # Keep old graph compatibility by delegating to hybrid retrieval + synthesis path.
    return {**state, "direct_quality": "insufficient"}


def query_rewriter_node(state: AgentState) -> AgentState:
    return {**state, "rewritten_query": state["question"], "answer_source": "knowledge_pipeline"}


def rag_lookup_node(state: AgentState) -> AgentState:
    return hybrid_search_node(state)


def topic_judge_node(state: AgentState) -> AgentState:
    return {**state, "topic": "other"}
