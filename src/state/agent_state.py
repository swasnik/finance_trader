"""
Agent state definitions and management.

Centralized state schema for the financial analysis agent graph.
All nodes operate on a shared AgentState that flows through the graph.
"""

import operator
from typing import Annotated, Any, Optional, TypedDict


class Message(TypedDict, total=False):
    """Represents a message in the conversation history."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str


class AgentState(TypedDict, total=False):
    """
    Centralized state for the financial analysis agent.

    This TypedDict is used as the LangGraph StateGraph schema. Each node reads
    from and writes to fields it cares about; LangGraph merges partial updates
    automatically. List fields with Annotated reducers are accumulated across
    node updates rather than replaced.

    Patterns:
    - Input: populated by entry nodes (user query, data sources)
    - Processing: intermediate steps append results to lists/dicts
    - Output: final results assembled from state fields
    """

    # Input
    user_query: str
    """The user's financial analysis request."""

    source_data: dict[str, Any]
    """Raw financial data from external sources (APIs, files, etc)."""

    # Processing — Annotated with operator.add so LangGraph accumulates entries
    messages: Annotated[list[dict], operator.add]
    """Conversation history for multi-turn reasoning."""

    analysis_steps: Annotated[list[str], operator.add]
    """Log of analysis steps executed so far."""

    intermediate_results: dict[str, Any]
    """Results from intermediate processing nodes."""

    # Agent work
    agent_state: str
    """Current state of the agent ('idle', 'analyzing', 'error')."""

    current_tool: Optional[str]
    """The tool the agent is currently using, if any."""

    # Output
    final_analysis: str
    """The final synthesized analysis provided to the user."""

    recommendations: Annotated[list[str], operator.add]
    """Actionable recommendations from the analysis."""

    # Metadata
    error: Optional[str]
    """Error message if processing failed."""

    metadata: dict[str, Any]
    """Additional metadata (timestamps, metrics, etc)."""

    refinement_count: int
    """Tracks how many refinement iterations have occurred (default 0)."""

    period: str
    """The fetch period parsed from the user query (default '1mo')."""

    ticker_info: dict
    """Fundamental data keyed by ticker symbol (sector, marketCap, P/E, etc.)."""


def get_default_state() -> AgentState:
    """Return an AgentState dict populated with all default values."""
    return AgentState(
        user_query="",
        source_data={},
        messages=[],
        analysis_steps=[],
        intermediate_results={},
        agent_state="",
        current_tool=None,
        final_analysis="",
        recommendations=[],
        error=None,
        metadata={},
        refinement_count=0,
        period="1mo",
        ticker_info={},
    )


# Type aliases for type hints
StateDict = dict[str, Any]
"""Type alias: state can be passed as dict during graph execution."""
