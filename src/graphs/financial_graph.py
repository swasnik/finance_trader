"""
Core graph definition - orchestrates the financial analysis workflow.

Patterns established:
- StateGraph with typed state (AgentState)
- Linear + conditional flow
- Entry and exit patterns
- Node composition from src/nodes/
- Checkpointing via MemorySaver for multi-turn persistence
- Streaming via astream for real-time node updates
"""

import logging
from typing import Any, AsyncIterator, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    analyzer,
    data_loader,
    error_handler,
    input_validator,
    router_has_data,
    router_needs_refinement,
    synthesizer,
)
from src.state import AgentState

logger = logging.getLogger(__name__)


def build_financial_graph(checkpointer=None):
    """
    Build and compile the financial analysis graph with checkpointing.

    Graph structure:
    START -> input_validator -> [data_loader or error_handler]
             -> analyzer -> synthesizer -> [refine or output] -> END

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.
                     Defaults to MemorySaver() if None.

    Returns:
        CompiledGraph: compiled graph ready for .invoke() / .astream()
    """

    graph = StateGraph(AgentState)

    graph.add_node("input_validator", input_validator)
    graph.add_node("data_loader", data_loader)
    graph.add_node("analyzer", analyzer)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("error_handler", error_handler)

    graph.add_edge(START, "input_validator")

    graph.add_conditional_edges(
        "input_validator",
        router_has_data,
        {
            "data_loader": "data_loader",
            "analyzer": "analyzer",
            "error_handler": "error_handler",
        },
    )

    graph.add_edge("data_loader", "analyzer")
    graph.add_edge("analyzer", "synthesizer")

    graph.add_conditional_edges(
        "synthesizer",
        router_needs_refinement,
        {
            "refine": "analyzer",
            "output": END,
        },
    )

    graph.add_edge("error_handler", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


def build_simple_graph(checkpointer=None):
    """
    Simplified 4-node graph for quick testing.

    Linear flow: input_validator -> data_loader -> analyzer -> synthesizer -> END

    Args:
        checkpointer: Optional LangGraph checkpointer. Defaults to MemorySaver().

    Returns:
        CompiledGraph: compiled graph ready for .invoke() / .astream()
    """

    graph = StateGraph(AgentState)

    graph.add_node("input_validator", input_validator)
    graph.add_node("data_loader", data_loader)
    graph.add_node("analyzer", analyzer)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "input_validator")
    graph.add_edge("input_validator", "data_loader")
    graph.add_edge("data_loader", "analyzer")
    graph.add_edge("analyzer", "synthesizer")
    graph.add_edge("synthesizer", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# Graph Execution Utilities
# ============================================================================


async def run_financial_graph(
    user_query: str,
    data: Optional[dict] = None,
    thread_id: str = "default",
    checkpointer=None,
) -> dict:
    """
    Execute the financial analysis graph with checkpointing.

    Args:
        user_query: User's financial analysis request
        data: Optional pre-loaded data dict
        thread_id: Conversation thread ID for multi-turn persistence
        checkpointer: Optional checkpointer (defaults to MemorySaver)

    Returns:
        dict: Final state after graph execution
    """
    compiled = build_financial_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"user_query": user_query, "source_data": data or {}}

    logger.info(f"Starting graph execution: {user_query}")
    result = await compiled.ainvoke(initial_state, config=config)
    logger.info(f"Graph completed. Agent state: {result.get('agent_state')}")

    return result


async def stream_financial_graph(
    user_query: str,
    data: Optional[dict] = None,
    thread_id: str = "default",
    checkpointer=None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream the financial analysis graph, yielding node-level updates.

    Yields dicts like: {"node": "analyzer", "update": {...state changes...}}

    Example:
        async for event in stream_financial_graph("Analyze AAPL"):
            print(f"[{event['node']}] {list(event['update'].keys())}")
    """
    compiled = build_financial_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"user_query": user_query, "source_data": data or {}}

    async for event in compiled.astream(initial_state, config=config, stream_mode="updates"):
        for node_name, state_update in event.items():
            yield {"node": node_name, "update": state_update}


def run_simple_graph_sync(user_query: str) -> dict:
    """
    Execute the simple graph synchronously (for testing/scripts).

    Args:
        user_query: User's financial analysis request

    Returns:
        dict: Final state after graph execution
    """
    compiled = build_simple_graph()
    config = {"configurable": {"thread_id": "sync-default"}}
    initial_state = {"user_query": user_query}

    logger.info(f"Starting simple graph: {user_query}")
    result = compiled.invoke(initial_state, config=config)
    logger.info(f"Graph completed. Agent state: {result.get('agent_state')}")

    return result
