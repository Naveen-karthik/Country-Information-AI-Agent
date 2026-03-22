from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    intent_parser_node,
    tool_invocation_node,
    answer_synthesis_node,
)


def _should_fetch_or_skip(state: AgentState) -> str:
    """
    Conditional edge after intent parsing.
    If an error occurred (e.g. LLM failed, no country found),
    skip tool invocation and go straight to synthesis.
    """
    if state.get("error"):
        return "synthesize"
    return "invoke_tool"


def build_agent_graph():
    """
    Builds and compiles the Country Agent LangGraph.

    Flow:
        parse_intent
            ├─ (error) ──────────────→ synthesize → END
            └─ (ok) → invoke_tool → synthesize → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("parse_intent", intent_parser_node)
    graph.add_node("invoke_tool", tool_invocation_node)
    graph.add_node("synthesize", answer_synthesis_node)

    # Entry point
    graph.set_entry_point("parse_intent")

    # Conditional edge: error shortcircuits tool call
    graph.add_conditional_edges(
        "parse_intent",
        _should_fetch_or_skip,
        {
            "invoke_tool": "invoke_tool",
            "synthesize": "synthesize",
        },
    )

    # Happy path: tool → synthesize → done
    graph.add_edge("invoke_tool", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Compile once at import time — reused across all requests
agent = build_agent_graph()