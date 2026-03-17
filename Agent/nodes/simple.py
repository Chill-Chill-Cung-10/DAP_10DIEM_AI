"""Simple response nodes that need no retrieval or heavy LLM reasoning."""

import logging
import re
from datetime import datetime

from Agent.graph.state import AgentState

logger = logging.getLogger(__name__)


def greeting_node(state: AgentState) -> AgentState:
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


def system_info_node(state: AgentState) -> AgentState:
    q   = state["question"].strip().lower()
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


def clarify_question_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": (
            "Xin lỗi, tôi chưa chắc chắn về câu trả lời. "
            "Bạn có thể làm rõ câu hỏi thêm được không?"
        ),
    }


def answer_node(state: AgentState) -> AgentState:
    """Pass-through: answer should already be set by an upstream node."""
    return state