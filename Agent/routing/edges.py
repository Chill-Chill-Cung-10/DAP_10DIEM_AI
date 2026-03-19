"""Conditional edge routing functions for LangGraph."""

import logging

from Agent.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_intent(state: AgentState) -> str:
    return state.get("intent", "query")


def route_direct_quality(state: AgentState) -> str:
    decision = state.get("direct_quality")
    logger.info("Route direct_quality=%s", decision)
    return "good" if decision == "good" else "insufficient"


def route_topic(state: AgentState) -> str:
    return state.get("topic", "other")


def route_verification(state: AgentState) -> str:
    if state.get("verification") == "good":
        return "good"
    if (state.get("retry_count") or 0) > 1:
        return "weak"
    return "retry"