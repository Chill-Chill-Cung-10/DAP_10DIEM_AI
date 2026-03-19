"""Intent routing + lightweight structure/info extraction."""

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

_COMPARE_RE = re.compile(
    r"(so\s*sánh|khác\s*nhau|điểm\s*giống|vs\b|versus|giữa.+và)",
    re.IGNORECASE,
)

_DIAGNOSIS_RE = re.compile(
    r"(chẩn\s*đoán|chuẩn\s*đoán|triệu\s*chứng|dấu\s*hiệu|bị\s*bệnh\s*gì|lá\s*bị|quả\s*bị)",
    re.IGNORECASE,
)

_RECENT_RE = re.compile(
    r"(mới\s*nhất|gần\s*đây|cập\s*nhật|hiện\s*nay|latest|recent|new\s*research|new\s*study)",
    re.IGNORECASE,
)

_LOGICAL_RE = re.compile(
    r"(suy\s*luận|logic|nếu.+thì|why|tại\s*sao|vì\s*sao|nguyên\s*lý)",
    re.IGNORECASE,
)

_KNOWN_ENTITIES = [
    "apple scab",
    "black rot",
    "cedar-apple rust",
    "powdery mildew",
    "gray leaf spot",
    "common rust",
    "cà chua",
    "nho",
    "táo",
    "ngô",
    "khoai tây",
    "Venturia inaequalis",
    "Phytophthora infestans",
    "Alternaria solani",
    "Cercospora zeae-maydis",
    "Puccinia sorghi",
]


def _extract_entities(question: str) -> list[str]:
    lowered = question.lower()
    entities: list[str] = []
    for ent in _KNOWN_ENTITIES:
        if ent.lower() in lowered:
            entities.append(ent)
    return entities


def _split_compare_targets(question: str) -> list[str]:
    # Handle common Vietnamese/English compare separators.
    separators = [" so sánh ", " vs ", " versus ", " giữa "]
    q = f" {question.strip()} "
    for sep in separators:
        if sep in q.lower():
            parts = [p.strip(" .,?!") for p in re.split(sep, q, flags=re.IGNORECASE) if p.strip()]
            if len(parts) >= 2:
                return parts[:2]

    # Fallback for pattern "giữa A và B"
    m = re.search(r"giữa\s+(.+?)\s+và\s+(.+)", question, flags=re.IGNORECASE)
    if m:
        return [m.group(1).strip(" .,?!"), m.group(2).strip(" .,?!")]
    return []


def _classify_intent(question: str) -> str:
    if _COMPARE_RE.search(question):
        return "compare"
    if _DIAGNOSIS_RE.search(question):
        return "diagnosis"
    if _RECENT_RE.search(question):
        return "recent_information"
    if _LOGICAL_RE.search(question):
        return "logical_reasoning"
    return "explanation_retrieve"


def intent_router(state: AgentState) -> AgentState:
    q = unicodedata.normalize("NFC", state["question"].strip())

    if _GREETING_RE.match(q):
        logger.info("Router → greeting")
        return {**state, "intent": "greeting", "entities": []}

    if _SYSTEM_INFO_RE.search(q):
        logger.info("Router → direct_system_info")
        return {**state, "intent": "direct_system_info", "entities": []}

    intent = _classify_intent(q)
    entities = _extract_entities(q)
    compare_targets = _split_compare_targets(q) if intent == "compare" else []

    logger.info("Router → %s (entities=%d, compare_targets=%d)", intent, len(entities), len(compare_targets))
    return {
        **state,
        "intent": intent,
        "entities": entities,
        "sub_queries": compare_targets,
        "freshness_status": "unknown" if intent == "recent_information" else state.get("freshness_status"),
    }