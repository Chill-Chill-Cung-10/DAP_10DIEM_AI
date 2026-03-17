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
    context       = docs_context(state.get("docs") or [])
    history_text  = chat_history_text(state.get("chat_history") or [])
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""

    # Nếu có docs nhưng similarity thấp → có context nhưng không đủ tin cậy
    # Nếu không có docs → hoàn toàn ngoài phạm vi
    has_context = bool(context.strip())

    if has_context:
        context_block = f"Ngữ cảnh tham khảo (có thể không liên quan trực tiếp):\n{context}\n\n"
    else:
        context_block = ""

    prompt = (
        "Bạn là trợ lý chuyên về bệnh học thực vật và cây trồng. "
        "Phạm vi kiến thức của bạn tập trung vào các bệnh trên cây trồng, "
        "triệu chứng, tác nhân gây bệnh và biện pháp quản lý.\n\n"

        "Hãy xử lý câu hỏi theo các trường hợp sau:\n\n"

        "• Nếu câu hỏi LIÊN QUAN đến bệnh cây trồng nhưng không có trong cơ sở dữ liệu:\n"
        "  Trả lời dựa trên kiến thức chung, ghi rõ "
        "  'Thông tin này dựa trên kiến thức chung, không có trong cơ sở dữ liệu.'\n\n"

        "• Nếu câu hỏi NGOÀI PHẠM VI (không liên quan đến bệnh cây trồng):\n"
        "  Lịch sự từ chối và hướng dẫn người dùng hỏi đúng chủ đề.\n"
        "  Ví dụ: 'Xin lỗi, tôi chỉ hỗ trợ các câu hỏi về bệnh học thực vật và cây trồng. "
        "  Bạn có thể hỏi tôi về triệu chứng, tác nhân gây bệnh hoặc biện pháp xử lý bệnh cây.'\n\n"

        "• Nếu câu hỏi là CHÀO HỎI hoặc CHUNG CHUNG:\n"
        "  Trả lời thân thiện và gợi ý chủ đề có thể hỗ trợ.\n\n"

        "Giữ nguyên tên khoa học, tên thuốc và thuật ngữ kỹ thuật bằng tiếng Anh."
        f"{history_block}\n\n"
        f"{context_block}"
        f"Câu hỏi: {state['question']}"
    )

    content = safe_llm_invoke(prompt, fallback="Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi.")
    logger.info("Answer draft (fallback): %d chars", len(content))
    return {**state, "answer": content}

def answer_verifier_node(state: AgentState) -> AgentState:
    prompt = (
        "Bạn là chuyên gia kiểm duyệt thông tin bệnh học thực vật. "
        "Hãy đánh giá câu trả lời dưới đây có đầy đủ, chính xác và hữu ích không.\n\n"
        "Tiêu chí đánh giá 'good':\n"
        "- Có nêu tên bệnh hoặc tác nhân gây bệnh cụ thể\n"
        "- Có mô tả triệu chứng hoặc điều kiện phát sinh\n"
        "- Có ít nhất một biện pháp quản lý hoặc xử lý\n\n"
        "Tiêu chí đánh giá 'weak':\n"
        "- Trả lời chung chung, không có thông tin cụ thể\n"
        "- Thiếu tên bệnh hoặc tác nhân gây bệnh\n"
        "- Không có biện pháp xử lý nào\n\n"
        'Trả về ONLY JSON: {"verification": "good"} hoặc {"verification": "weak"}\n\n'
        "Ví dụ 1:\n"
        "Question: Apple scab gây ra triệu chứng gì trên lá?\n"
        "Answer: Apple scab do Venturia inaequalis gây ra, tạo đốm tròn màu xanh ô liu "
        "trên lá, sau chuyển nâu đen. Quản lý bằng captan hoặc lime-sulfur.\n"
        '{"verification": "good"}\n\n'
        "Ví dụ 2:\n"
        "Question: Bệnh thối đen trên nho điều trị thế nào?\n"
        "Answer: Cần xem xét thêm.\n"
        '{"verification": "weak"}\n\n'
        f"Question: {state['question']}\n"
        f"Answer: {state.get('answer')}"
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
        "Plan a short retrieval strategy for answering this plant disease question. "
        "Break it down into specific search queries targeting: "
        "pathogen name, symptoms, affected crops, lifecycle, or management methods. "
        "Return ONLY JSON: {\"plan\": [\"query1\", \"query2\", ...]}.\n\n"
        "Example 1:\n"
        "Question: So sánh bệnh early blight và late blight trên cà chua\n"
        '{"plan": [\n'
        '  "early blight tomato Alternaria solani symptoms management",\n'
        '  "late blight tomato Phytophthora infestans symptoms management",\n'
        '  "early blight late blight tomato difference"\n'
        "]}\n\n"
        "Example 2:\n"
        "Question: Các bệnh nấm phổ biến trên cây táo là gì?\n"
        '{"plan": [\n'
        '  "apple scab Venturia inaequalis fungal disease",\n'
        '  "black rot apple Diplodia seriata fungal disease",\n'
        '  "cedar apple rust Gymnosporangium fungal disease"\n'
        "]}\n\n"
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
        "Extract the key evidence snippets relevant to plant disease from the context below.\n"
        "Focus on extracting: pathogen names, symptoms, infection conditions, "
        "lifecycle details, and management/treatment methods.\n"
        'Return ONLY JSON: {"evidence": ["snippet1", "snippet2", ...]}.\n\n'
        "Example:\n"
        "Context: Câu hỏi: Apple scab là gì?\n"
        "Trả lời: Apple scab do Venturia inaequalis gây ra. Đốm lá tròn màu xanh ô liu, "
        "đường kính tới 1/2 inch. Nấm qua đông trên lá rụng. "
        "Quản lý bằng captan, lime-sulfur.\n"
        '{"evidence": [\n'
        '  "Tác nhân: Venturia inaequalis",\n'
        '  "Triệu chứng: đốm lá tròn màu xanh ô liu, đường kính tới 1/2 inch",\n'
        '  "Vòng đời: nấm qua đông trên lá rụng",\n'
        '  "Quản lý: captan, lime-sulfur"\n'
        "]}\n\n"
        f"Context:\n{context}"
    )
    content = safe_llm_invoke(prompt, fallback='{"evidence": []}')
    data    = safe_json_loads(content, {"evidence": []})
    logger.info("Evidence: %d snippets", len(data.get("evidence", [])))
    return {**state, **data}

def conclusion_builder_node(state: AgentState) -> AgentState:
    evidence      = "\n".join(state.get("evidence") or [])
    history_text  = chat_history_text(state.get("chat_history") or [])
    history_block = f"\nLịch sử hội thoại:\n{history_text}\n" if history_text else ""
    prompt = (
        "Bạn là chuyên gia bệnh học thực vật. Hãy thực hiện 2 bước:\n\n"
        "Bước 1 - Kiểm chứng bằng chứng:\n"
        "  - Loại bỏ thông tin mâu thuẫn hoặc không liên quan\n"
        "  - Ưu tiên thông tin có tên khoa học và số liệu cụ thể\n\n"
        "Bước 2 - Tổng hợp và trả lời theo cấu trúc:\n"
        "  1. Tên bệnh & tác nhân gây bệnh\n"
        "  2. Triệu chứng nhận dạng\n"
        "  3. Điều kiện phát sinh & vòng đời\n"
        "  4. Biện pháp quản lý & thuốc xử lý\n\n"
        "Giữ nguyên tên khoa học, tên thuốc và thuật ngữ kỹ thuật bằng tiếng Anh. "
        "Trả lời bằng tiếng Việt."
        f"{history_block}\n\n"
        f"Bằng chứng:\n{evidence}\n\n"
        f"Câu hỏi: {state['question']}"
    )
    content = safe_llm_invoke(
        prompt,
        fallback="Xin lỗi, đã xảy ra lỗi khi tổng hợp kết luận."
    )
    logger.info("Conclusion: %d chars", len(content))
    return {**state, "answer": content}