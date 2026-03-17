"""LangGraph shared state definition."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    question:       str
    chat_history:   Optional[List[dict]]
    intent:         Optional[str]
    entities:       Optional[List[str]]
    docs:           Optional[list]
    rewritten_query: Optional[str]
    relevance:      Optional[str]
    topic:          Optional[str]
    plan:           Optional[List[str]]
    evidence:       Optional[List[str]]
    answer:         Optional[str]
    answer_source:  Optional[str]
    verification:   Optional[str]
    retry_count:    Optional[int]
    direct_quality: Optional[str]   # "good" | "insufficient"