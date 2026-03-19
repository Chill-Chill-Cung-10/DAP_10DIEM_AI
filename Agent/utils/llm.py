"""LLM client and shared prompt helpers."""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from Agent.config import OPENAI_API_KEY, OPENAI_MODEL, DOC_CONTEXT_MAX_CHARS, CHAT_HISTORY_MAX_TURNS

logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0.2,
    timeout=60,
)


def safe_llm_invoke(prompt: str, fallback: str = "") -> str:
    """Invoke LLM and return content string; returns *fallback* on any error."""
    try:
        return llm.invoke(prompt).content
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return fallback


def safe_json_loads(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code fences."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return default


def docs_context(docs: list, max_chars: int = DOC_CONTEXT_MAX_CHARS) -> str:
    """Concatenate doc page_content within a character budget."""
    parts: List[str] = []
    total = 0
    for d in docs:
        if total + len(d.page_content) > max_chars:
            break
        parts.append(d.page_content)
        total += len(d.page_content)
    return "\n\n".join(parts)


def chat_history_text(history: list, max_turns: int = CHAT_HISTORY_MAX_TURNS) -> str:
    """Format recent chat history for prompt injection."""
    recent = history[-max_turns:] if len(history) > max_turns else history
    lines = []
    for turn in recent:
        prefix = "Người dùng" if turn.get("role") == "user" else "Trợ lý"
        lines.append(f"{prefix}: {turn.get('content', '')}")
    return "\n".join(lines)


def append_chat_history(
    history: List[dict],
    question: str,
    answer: str,
    max_items: int = 20,
) -> List[dict]:
    """Append a Q/A turn and keep history bounded."""
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history[-max_items:] if len(history) > max_items else history