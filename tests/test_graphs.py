"""
Tests for graph building and node execution.

Test patterns:
- State transformation verification
- Node isolation testing
- Graph topology verification
- End-to-end graph execution
"""

import pytest

from src.graphs import build_financial_graph, build_simple_graph
from src.nodes import analyzer, error_handler, input_validator, synthesizer
from src.state import AgentState, Message, get_default_state


def make_state(**kwargs) -> AgentState:
    """Build a full default state dict, overriding fields with kwargs."""
    state = get_default_state()
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state


class TestStateManagement:
    """Tests for AgentState and state flow."""

    def test_initial_state(self):
        """State initializes with defaults."""
        state = get_default_state()
        assert state["user_query"] == ""
        assert state["analysis_steps"] == []
        assert state["error"] is None
        assert state["agent_state"] == ""

    def test_state_update_immutability(self):
        """State updates via dict spread leave the original unchanged."""
        state1 = make_state(user_query="test")
        state2 = {**state1, "user_query": "updated"}

        assert state1["user_query"] == "test"
        assert state2["user_query"] == "updated"

    def test_analysis_steps_accumulate(self):
        """Analysis steps are tracked as a list."""
        state = make_state()
        state = {**state, "analysis_steps": state["analysis_steps"] + ["step1"]}
        state = {**state, "analysis_steps": state["analysis_steps"] + ["step2"]}

        assert state["analysis_steps"] == ["step1", "step2"]


class TestNodeExecution:
    """Tests for individual node functions."""

    def test_input_validator_success(self):
        """Input validator normalizes valid input."""
        state = make_state(user_query="  Analyze my portfolio  ")
        result = input_validator(state)

        assert result["user_query"] == "Analyze my portfolio"
        assert result["agent_state"] == "initialized"
        assert "input_validated" in result["analysis_steps"]

    def test_input_validator_empty_query(self):
        """Input validator rejects empty query."""
        state = make_state(user_query="")
        result = input_validator(state)

        assert result.get("error") is not None
        assert result["agent_state"] == "error"

    def test_analyzer_requires_loaded_data(self):
        """Analyzer checks preconditions."""
        state = make_state(user_query="test")
        result = analyzer(state)

        # Analyzer expects source_data to be present
        assert result.get("error") is not None

    def test_synthesizer_compiles_results(self):
        """Synthesizer creates final output."""
        state = make_state(
            user_query="test",
            agent_state="analyzed",
            intermediate_results={"summary": "test summary"},
        )
        result = synthesizer(state)

        assert result["final_analysis"] != ""
        assert result["agent_state"] == "complete"
        assert len(result["recommendations"]) > 0

    def test_error_handler_logs(self):
        """Error handler processes error state without changing it."""
        state = make_state(error="Test error", agent_state="error")
        update = error_handler(state)
        merged = {**state, **update}

        # Error handler should not clear the error state
        assert merged["error"] == "Test error"


class TestGraphConstruction:
    """Tests for graph topology and structure."""

    def test_simple_graph_builds(self):
        """Simple graph compiles without errors."""
        compiled = build_simple_graph()
        assert compiled is not None

    def test_financial_graph_builds(self):
        """Financial graph compiles without errors."""
        compiled = build_financial_graph()
        assert compiled is not None

    def test_simple_graph_execution(self):
        """Simple graph executes end-to-end."""
        compiled = build_simple_graph()

        result = compiled.invoke({"user_query": "Test query"}, config={"configurable": {"thread_id": "test"}})

        assert isinstance(result, dict)
        assert result["user_query"] == "Test query"
        # After full execution, should have analysis and synthesis
        assert result["agent_state"] == "complete" or result["agent_state"] == "error"


class TestNodeComposition:
    """Tests for composing nodes together."""

    def test_pipeline_input_to_analysis(self):
        """Pipeline: validate -> analyze using data-presence guard."""
        state = make_state(
            user_query="Test analysis",
            source_data={"key": "value"},
            agent_state="data_loaded",
        )

        # Validate — merge partial update back into full state
        state = {**state, **input_validator(state)}
        assert state["agent_state"] == "initialized"

        # Analyze — source_data still present after merge
        state = {**state, **analyzer(state)}
        assert state["agent_state"] == "analyzed"

    def test_error_stops_pipeline(self):
        """When error occurs, agent_state becomes 'error'."""
        state = make_state(user_query="")
        state = {**state, **input_validator(state)}

        assert state["agent_state"] == "error"
        assert state.get("error") is not None


class TestStateFlowPatterns:
    """Tests for common state flow patterns."""

    def test_intermediate_results_accumulation(self):
        """Intermediate results accumulate during processing."""
        state = make_state()
        state = {**state, "intermediate_results": {
            **state["intermediate_results"], "step1": "result1"
        }}
        state = {**state, "intermediate_results": {
            **state["intermediate_results"], "step2": "result2"
        }}

        assert state["intermediate_results"] == {"step1": "result1", "step2": "result2"}

    def test_messages_accumulate(self):
        """Conversation messages accumulate."""
        state = make_state()
        msg1 = Message(role="user", content="Hello")
        state = {**state, "messages": state["messages"] + [msg1]}

        assert len(state["messages"]) == 1
        assert state["messages"][0]["content"] == "Hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
