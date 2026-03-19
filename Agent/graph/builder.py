"""Assemble and compile the LangGraph application."""

from langgraph.graph import StateGraph, END

from Agent.graph.state import AgentState
from Agent.nodes.router import intent_router
from Agent.nodes.simple import greeting_node, system_info_node, clarify_question_node, answer_node
from Agent.nodes.retrieval import (
    logical_reasoning_node,
    diagnosis_retrieval_node,
    diagnosis_reasoning_node,
    compare_split_queries_node,
    compare_retrieval_node,
    compare_synthesis_node,
    recent_freshness_check_node,
    recent_fallback_node,
    hybrid_search_node,
)
from Agent.nodes.synthesis import answer_draft_node, answer_verifier_node
from Agent.routing.edges import route_intent, route_verification, route_freshness


def build_app():
    """Build and compile the LangGraph agent with intent-aware branches."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router", intent_router)
    graph.add_node("greeting", greeting_node)
    graph.add_node("system_info", system_info_node)

    graph.add_node("logical_reasoning", logical_reasoning_node)

    graph.add_node("diagnosis_retrieval", diagnosis_retrieval_node)
    graph.add_node("diagnosis_reasoning", diagnosis_reasoning_node)

    graph.add_node("compare_split", compare_split_queries_node)
    graph.add_node("compare_retrieval", compare_retrieval_node)
    graph.add_node("compare_synthesis", compare_synthesis_node)

    graph.add_node("recent_freshness_check", recent_freshness_check_node)
    graph.add_node("recent_fallback", recent_fallback_node)

    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("answer_draft", answer_draft_node)

    graph.add_node("intent_quality_check", answer_verifier_node)
    graph.add_node("clarify_question", clarify_question_node)
    graph.add_node("answer_passthrough", answer_node)

    graph.set_entry_point("router")

    # Intent dispatch
    graph.add_conditional_edges(
        "router",
        route_intent,
        {
            "greeting": "greeting",
            "direct_system_info": "system_info",
            "logical_reasoning": "logical_reasoning",
            "diagnosis": "diagnosis_retrieval",
            "compare": "compare_split",
            "recent_information": "recent_freshness_check",
            "explanation_retrieve": "hybrid_search",
        },
    )

    # logical_reasoning -> quality check
    graph.add_edge("logical_reasoning", "intent_quality_check")

    # diagnosis -> retrieval -> reasoning -> quality check
    graph.add_edge("diagnosis_retrieval", "diagnosis_reasoning")
    graph.add_edge("diagnosis_reasoning", "intent_quality_check")

    # compare -> split -> dual retrieval -> structured compare -> quality check
    graph.add_edge("compare_split", "compare_retrieval")
    graph.add_edge("compare_retrieval", "compare_synthesis")
    graph.add_edge("compare_synthesis", "intent_quality_check")

    # recent information -> freshness gate
    graph.add_conditional_edges(
        "recent_freshness_check",
        route_freshness,
        {
            "fresh": "hybrid_search",
            "stale": "recent_fallback",
        },
    )
    graph.add_edge("recent_fallback", "intent_quality_check")

    # default explanation/retrieve path
    graph.add_edge("hybrid_search", "answer_draft")
    graph.add_edge("answer_draft", "intent_quality_check")

    # intent-aware quality check
    graph.add_conditional_edges(
        "intent_quality_check",
        route_verification,
        {
            "good": "answer_passthrough",
            "weak": "clarify_question",
        },
    )

    graph.add_edge("greeting", END)
    graph.add_edge("system_info", "answer_passthrough")
    graph.add_edge("clarify_question", END)
    graph.add_edge("answer_passthrough", END)

    return graph.compile()
