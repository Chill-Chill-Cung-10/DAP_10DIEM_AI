"""Answer synthesis nodes: draft, verify, evidence pipeline."""

import logging

from Agent.graph.state import AgentState
from Agent.utils.llm import safe_llm_invoke, safe_json_loads, docs_context, chat_history_text
from Agent.bootstrap.search import pgvector_search
from Agent.config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple draft + verify
# ---------------------------------------------------------------------------

def answer_draft_node(state: AgentState) -> AgentState:
    context      = docs_context(state.get("docs") or [])
    history_text = chat_history_text(state.get("chat_history") or [])
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Bạn là trợ lý pháp luật Việt Nam. Dựa vào ngữ cảnh được cung cấp, "
        "hãy trả lời câu hỏi một cách ngắn gọn, chính xác bằng tiếng Việt.\n"
        "Nếu ngữ cảnh không đủ, hãy nói rõ rằng bạn không có đủ thông tin."
        f"{history_block}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    content = safe_llm_invoke(prompt, fallback="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi.")
    logger.info("Answer draft: %d chars", len(content))
    return {**state, "answer": content}


def answer_verifier_node(state: AgentState) -> AgentState:
    prompt = (
        "Verify whether the answer below is strong, well-supported and accurate.\n"
        'Return ONLY JSON: {"verification": "good"} or {"verification": "weak"}.\n\n'
        "Example 1:\n"
        "Question: Thời hiệu khởi kiện vụ án dân sự?\n"
        "Answer: Theo Bộ luật Tố tụng dân sự 2015, thời hiệu là 03 năm kể từ ngày biết.\n"
        '{"verification": "good"}\n\n'
        "Example 2:\n"
        "Question: Mức phạt vi phạm giao thông?\n"
        "Answer: Tùy trường hợp.\n"
        '{"verification": "weak"}\n\n'
        f"Question: {state['question']}\nAnswer: {state.get('answer')}"
    )
    content = safe_llm_invoke(prompt, fallback='{"verification": "good"}')
    data    = safe_json_loads(content, {"verification": "good"})

    retry = state.get("retry_count") or 0
    if data.get("verification") == "weak":
        retry += 1

    logger.info("Verification: %s (retry=%d)", data.get("verification"), retry)
    return {**state, **data, "retry_count": retry}


# ---------------------------------------------------------------------------
# Evidence sub-pipeline (legal queries)
# ---------------------------------------------------------------------------

def search_planner_node(state: AgentState) -> AgentState:
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
    content = safe_llm_invoke(prompt, fallback='{"plan": []}')
    data    = safe_json_loads(content, {"plan": []})
    logger.info("Search plan: %d steps", len(data.get("plan", [])))
    return {**state, **data}


def multi_source_retrieval_node(state: AgentState) -> AgentState:
    """Retrieve docs for each plan step, merging unique results."""
    plan     = state.get("plan") or []
    all_docs = list(state.get("docs") or [])
    seen     = {d.metadata.get("question", "") for d in all_docs}
    queries  = [state["question"]] + [s for s in plan if isinstance(s, str)]

    for query in queries:
        for d in pgvector_search(query, k=10, similarity_threshold=0.40):
            q = d.metadata.get("question", "")
            if q not in seen:
                seen.add(q)
                all_docs.append(d)

    return {**state, "docs": all_docs}


def source_ranker_node(state: AgentState) -> AgentState:
    docs = sorted(
        state.get("docs") or [],
        key=lambda d: d.metadata.get("similarity", 0),
        reverse=True,
    )
    return {**state, "docs": docs}


def evidence_extractor_node(state: AgentState) -> AgentState:
    context = docs_context(state.get("docs") or [])
    prompt = (
        "Extract the key evidence snippets from the context below.\n"
        'Return ONLY JSON: {"evidence": ["snippet1", "snippet2", ...]}.\n\n'
        "Example:\n"
        "Context: Câu hỏi: Thời hạn hợp đồng lao động?\n"
        "Trả lời: Hợp đồng lao động có 2 loại: xác định thời hạn (không quá 36 tháng) và không xác định thời hạn.\n"
        '{"evidence": ["Hợp đồng xác định thời hạn: không quá 36 tháng", "Hợp đồng không xác định thời hạn"]}\n\n'
        f"Context:\n{context}"
    )
    content = safe_llm_invoke(prompt, fallback='{"evidence": []}')
    data    = safe_json_loads(content, {"evidence": []})
    logger.info("Evidence: %d snippets", len(data.get("evidence", [])))
    return {**state, **data}


def conclusion_builder_node(state: AgentState) -> AgentState:
    """Synthesise evidence into a final verified answer."""
    evidence     = "\n".join(state.get("evidence") or [])
    history_text = chat_history_text(state.get("chat_history") or [])
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Bạn là trợ lý pháp luật Việt Nam. Hãy thực hiện 2 bước:\n"
        "1. Kiểm chứng các bằng chứng: loại bỏ thông tin mâu thuẫn hoặc không đáng tin.\n"
        "2. Tổng hợp bằng chứng đáng tin và đưa ra câu trả lời chính xác bằng tiếng Việt."
        f"{history_block}\n\n"
        f"Bằng chứng:\n{evidence}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    content = safe_llm_invoke(prompt, fallback="Xin lỗi, đã xảy ra lỗi khi tổng hợp kết luận.")
    logger.info("Conclusion: %d chars", len(content))
    return {**state, "answer": content}