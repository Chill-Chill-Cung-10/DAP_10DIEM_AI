"""Assemble and compile the LangGraph application."""

from langgraph.graph import StateGraph, END

from Agent.graph.state import AgentState
from Agent.nodes.router   import intent_router
from Agent.nodes.simple   import greeting_node, system_info_node, clarify_question_node, answer_node
from Agent.nodes.retrieval import (
    direct_answer_node,
    query_rewriter_node,
    rag_lookup_node,
    topic_judge_node,
)
from Agent.nodes.synthesis import (
    answer_draft_node,
    answer_verifier_node,
    search_planner_node,
    multi_source_retrieval_node,
    source_ranker_node,
    evidence_extractor_node,
    conclusion_builder_node,
)
from Agent.routing.edges import (
    route_intent,
    route_direct_quality,
    route_topic,
    route_verification,
)


def build_app():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # --- Register nodes ---
    graph.add_node("router",               intent_router)
    graph.add_node("greeting",             greeting_node)
    graph.add_node("system_info",          system_info_node)
    graph.add_node("direct_answer",        direct_answer_node)
    graph.add_node("query_rewriter",       query_rewriter_node)
    graph.add_node("rag",                  rag_lookup_node)
    graph.add_node("topic_judge",          topic_judge_node)
    graph.add_node("answer_draft",         answer_draft_node)
    graph.add_node("answer_verifier",      answer_verifier_node)
    graph.add_node("clarify_question",     clarify_question_node)
    graph.add_node("answer",               answer_node)
    graph.add_node("search_planner",       search_planner_node)
    graph.add_node("multi_source_retrieval", multi_source_retrieval_node)
    graph.add_node("source_ranker",        source_ranker_node)
    graph.add_node("evidence_extractor",   evidence_extractor_node)
    graph.add_node("conclusion_builder",   conclusion_builder_node)

    # --- Entry point ---
    graph.set_entry_point("router")

    # --- Edges ---
    graph.add_conditional_edges("router", route_intent, {
        "greeting":          "greeting",
        "direct_system_info": "system_info",
        "query":             "direct_answer",
    })

    graph.add_conditional_edges("direct_answer", route_direct_quality, {
        "good":        END,
        "insufficient": "query_rewriter",
    })

    graph.add_edge("system_info",    "answer")
    graph.add_edge("query_rewriter", "rag")
    graph.add_edge("rag",            "topic_judge")

    graph.add_conditional_edges("topic_judge", route_topic, {
        "legal": "search_planner",
        "other": "answer_draft",
    })

    graph.add_edge("search_planner",         "multi_source_retrieval")
    graph.add_edge("multi_source_retrieval", "source_ranker")
    graph.add_edge("source_ranker",          "evidence_extractor")
    graph.add_edge("evidence_extractor",     "conclusion_builder")
    graph.add_edge("conclusion_builder",     "answer_verifier")

    graph.add_edge("answer_draft", "answer")

    graph.add_conditional_edges("answer_verifier", route_verification, {
        "good":  "answer",
        "weak":  "clarify_question",
        "retry": "rag",
    })

    graph.add_edge("clarify_question", END)
    graph.add_edge("greeting",         END)
    graph.add_edge("answer",           END)

    return graph.compile()