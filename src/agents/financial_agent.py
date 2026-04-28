"""
Financial agent with tool-calling capability.

Uses LangGraph's ToolNode to execute financial metric calculations
as LLM-directed tool calls.
"""

import logging
import operator
from datetime import date
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.tools import get_all_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are a professional financial analyst assistant. Today is {date.today().isoformat()}.

Your role is to analyze financial data and compute metrics using available tools.
Think step-by-step: identify what data is needed, call tools to compute metrics, then synthesize findings.

When price data is available in the conversation, use the calculation tools to provide precise metrics.
If data is insufficient or unavailable, say so clearly rather than speculating.

Important: This analysis is for informational purposes only and does not constitute investment advice."""


class AgentGraphState(TypedDict, total=False):
    """State for the ToolNode-based financial agent.

    Uses add_messages reducer so LangChain BaseMessage objects are
    properly accumulated — required for tools_condition to inspect
    tool_calls on the last AI message.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    source_data: dict[str, Any]
    agent_state: str
    error: Optional[str]
    analysis_steps: Annotated[list[str], operator.add]


def build_agent_graph(checkpointer=None):
    """
    Build a ReAct-style financial agent with tool calling.

    Uses LangGraph's ToolNode for automatic tool execution.
    Falls back gracefully if LLM is not configured.

    Returns:
        CompiledGraph ready for invoke/astream
    """
    from langgraph.checkpoint.memory import MemorySaver

    from src.llm.provider import get_llm, is_llm_configured

    tools = get_all_tools()

    def agent_node(state: dict) -> dict[str, Any]:
        """LLM node that decides which tools to call."""
        if not is_llm_configured():
            return {
                "messages": [
                    AIMessage(
                        content="LLM not configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
                    )
                ],
                "agent_state": "complete",
            }

        try:
            llm = get_llm(temperature=0)
            llm_with_tools = llm.bind_tools(tools)

            existing_messages: list = state.get("messages", [])
            query: str = state.get("user_query", "")
            source_data: dict = state.get("source_data", {})

            lc_messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

            has_human = any(isinstance(m, HumanMessage) for m in existing_messages)
            if query and not has_human:
                context = f"Query: {query}"
                if source_data:
                    tickers = source_data.get("tickers", [])
                    context += f"\nAvailable tickers: {tickers}"
                lc_messages.append(HumanMessage(content=context))

            lc_messages.extend(existing_messages)

            response = llm_with_tools.invoke(lc_messages)

            tool_calls = getattr(response, "tool_calls", None)
            new_state = "tool_calling" if tool_calls else "analyzed"
            return {"messages": [response], "agent_state": new_state}

        except Exception as e:
            logger.error(f"Agent node error: {e}")
            return {"error": str(e), "agent_state": "error"}

    graph = StateGraph(AgentGraphState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    cp = checkpointer or MemorySaver()
    return graph.compile(checkpointer=cp)


def run_agent_sync(
    user_query: str,
    source_data: Optional[dict] = None,
    thread_id: str = "default",
) -> dict:
    """Run the financial agent synchronously."""
    compiled = build_agent_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: dict = {"user_query": user_query, "source_data": source_data or {}}
    return compiled.invoke(initial, config=config)
