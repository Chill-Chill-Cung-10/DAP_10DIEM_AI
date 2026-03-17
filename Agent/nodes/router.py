"""Intent-routing node — keyword-based, no LLM call."""

import logging
import re
import unicodedata

from Agent.graph.state import AgentState

logger = logging.getLogger(__name__)

_GREETING_RE = re.compile(
    r"^\s*("
    r"xin\s*chào|chào\s*bạn|chào|hello|hi\b|hey\b"
    r"|cảm\s*ơn|cám\s*ơn|thanks|thank\s*you|tks"
    r"|tạm\s*biệt|bye|goodbye|see\s*you"
    r"|ok\b|okay|ừ|ờ|vâng|dạ"
    r"|tốt|hay\s*lắm|giỏi\s*lắm|great|good"
    r")\s*[!.?~]*\s*$",
    re.IGNORECASE,
)

_SYSTEM_INFO_RE = re.compile(
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


def intent_router(state: AgentState) -> AgentState:
    q = unicodedata.normalize("NFC", state["question"].strip())

    if _GREETING_RE.match(q):
        logger.info("Router → greeting")
        return {**state, "intent": "greeting", "entities": []}

    if _SYSTEM_INFO_RE.search(q):
        logger.info("Router → direct_system_info")
        return {**state, "intent": "direct_system_info", "entities": []}

    logger.info("Router → query")
    return {**state, "intent": "query", "entities": []}