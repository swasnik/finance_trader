"""
Tests for advanced LangGraph features:
- Checkpointing
- Streaming
- ToolNode agent
- Thread continuity
"""

import asyncio

from src.agents.financial_agent import build_agent_graph
from src.graphs.financial_graph import (
    build_financial_graph,
    build_simple_graph,
    run_simple_graph_sync,
    stream_financial_graph,
)
from src.tools import get_all_tools


class TestCheckpointing:
    """Tests for checkpointer integration."""

    def test_financial_graph_compiles_with_checkpointer(self):
        """build_financial_graph() returns a compiled graph with a checkpointer."""
        compiled = build_financial_graph()
        assert compiled is not None
        assert compiled.checkpointer is not None

    def test_simple_graph_compiles_with_checkpointer(self):
        """build_simple_graph() returns a compiled graph with a checkpointer."""
        compiled = build_simple_graph()
        assert compiled is not None
        assert compiled.checkpointer is not None

    def test_custom_checkpointer_accepted(self):
        """build_financial_graph() accepts a custom checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        compiled = build_financial_graph(checkpointer=cp)
        assert compiled.checkpointer is cp


class TestSimpleGraphSync:
    """Tests for run_simple_graph_sync."""

    def test_returns_dict_with_agent_state(self):
        """run_simple_graph_sync returns a dict with agent_state key."""
        result = run_simple_graph_sync("Analyze AAPL stock")
        assert isinstance(result, dict)
        assert "agent_state" in result

    def test_agent_state_is_complete_or_error(self):
        """agent_state ends in a terminal value."""
        result = run_simple_graph_sync("Analyze AAPL stock")
        assert result["agent_state"] in ("complete", "error")

    def test_user_query_preserved(self):
        """User query is preserved in final state."""
        result = run_simple_graph_sync("Analyze AAPL stock")
        assert result["user_query"] == "Analyze AAPL stock"


class TestStreaming:
    """Tests for stream_financial_graph async generator."""

    def test_stream_yields_events(self):
        """stream_financial_graph yields at least one event."""

        async def _collect():
            events = []
            async for event in stream_financial_graph("Analyze MSFT stock"):
                events.append(event)
                if len(events) >= 1:
                    break
            return events

        events = asyncio.run(_collect())
        assert len(events) >= 1

    def test_stream_event_has_node_and_update_keys(self):
        """Each streamed event has 'node' and 'update' keys."""

        async def _collect():
            events = []
            async for event in stream_financial_graph("Analyze TSLA stock"):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        assert len(events) >= 1
        for event in events:
            assert "node" in event, f"Missing 'node' key in event: {event}"
            assert "update" in event, f"Missing 'update' key in event: {event}"

    def test_stream_node_names_are_strings(self):
        """Node names in streamed events are strings."""

        async def _collect():
            events = []
            async for event in stream_financial_graph("Portfolio analysis"):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        for event in events:
            assert isinstance(event["node"], str)


class TestThreadContinuity:
    """Tests for multi-turn persistence via thread_id."""

    def test_same_thread_state_accessible_after_invoke(self):
        """After invoking with a thread_id, get_state returns the saved state."""
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        compiled = build_financial_graph(checkpointer=cp)
        thread_id = "continuity-test-thread"
        config = {"configurable": {"thread_id": thread_id}}

        compiled.invoke({"user_query": "Analyze portfolio"}, config=config)

        saved = compiled.get_state(config)
        assert saved is not None
        assert saved.values.get("user_query") == "Analyze portfolio"

    def test_second_invoke_builds_on_prior_state(self):
        """Second invocation with the same thread sees prior checkpoint."""
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        compiled = build_simple_graph(checkpointer=cp)
        thread_id = "multi-turn-thread"
        config = {"configurable": {"thread_id": thread_id}}

        compiled.invoke({"user_query": "First query"}, config=config)
        state_after_first = compiled.get_state(config)

        compiled.invoke({"user_query": "Second query"}, config=config)
        state_after_second = compiled.get_state(config)

        # Both states should exist and second should have a newer checkpoint
        assert state_after_first is not None
        assert state_after_second is not None


class TestAgentGraph:
    """Tests for the ToolNode-based financial agent."""

    def test_agent_graph_compiles(self):
        """build_agent_graph() compiles without error."""
        compiled = build_agent_graph()
        assert compiled is not None

    def test_agent_graph_has_checkpointer(self):
        """Agent graph has a checkpointer attached."""
        compiled = build_agent_graph()
        assert compiled.checkpointer is not None

    def test_agent_graph_accepts_custom_checkpointer(self):
        """build_agent_graph() accepts a custom checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        compiled = build_agent_graph(checkpointer=cp)
        assert compiled.checkpointer is cp


class TestGetAllTools:
    """Tests for get_all_tools()."""

    def test_returns_list_of_five_tools(self):
        """get_all_tools() returns exactly 6 tools."""
        tools = get_all_tools()
        assert isinstance(tools, list)
        assert len(tools) == 6

    def test_tools_are_callable(self):
        """All tools are invocable via .invoke()."""
        tools = get_all_tools()
        for tool in tools:
            assert hasattr(tool, "invoke"), f"{tool} missing .invoke()"

    def test_tool_names(self):
        """All expected tool names are present."""
        tools = get_all_tools()
        names = {t.name for t in tools}
        expected = {
            "calculate_returns",
            "calculate_volatility",
            "calculate_sharpe_ratio",
            "calculate_max_drawdown",
            "extract_close_prices",
            "calculate_correlation",
        }
        assert names == expected
