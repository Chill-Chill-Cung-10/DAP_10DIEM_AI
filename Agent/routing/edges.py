"""Conditional edge routing functions for LangGraph."""

import logging

from Agent.config import VERIFY_INTENTS
from Agent.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_intent(state: AgentState) -> str:
    return state.get("intent", "explanation_retrieve")


def route_direct_quality(state: AgentState) -> str:
    decision = state.get("direct_quality")
    logger.info("Route direct_quality=%s", decision)
    return "good" if decision == "good" else "insufficient"


def route_topic(state: AgentState) -> str:
    return state.get("topic", "other")


def route_verification(state: AgentState) -> str:
    return "good" if state.get("verification") == "good" else "weak"


def route_freshness(state: AgentState) -> str:
    return "fresh" if state.get("freshness_status") == "fresh" else "stale"


def route_quality_gate(state: AgentState) -> str:
    intent = state.get("intent") or "explanation_retrieve"
    return "verify" if intent in VERIFY_INTENTS else "passthrough"