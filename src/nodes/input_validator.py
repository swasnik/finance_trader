"""Input validation node — entry point of the financial analysis graph."""

from typing import Any

from src.nodes.base import node_with_logging
from src.state import AgentState


def _parse_period(query: str) -> str:
    """Parse temporal intent from query to yfinance period string."""
    q = query.lower()
    if any(x in q for x in ["5 year", "5-year", "five year", "5yr"]):
        return "5y"
    if any(x in q for x in ["3 year", "3-year", "three year", "3yr"]):
        return "3y"
    if any(x in q for x in ["2 year", "2-year", "two year", "2yr"]):
        return "2y"
    if any(x in q for x in ["1 year", "1-year", "one year", "annual", "yearly", "ytd"]):
        return "1y"
    if any(x in q for x in ["6 month", "6-month", "six month", "half year"]):
        return "6mo"
    if any(x in q for x in ["3 month", "3-month", "three month", "quarter"]):
        return "3mo"
    if any(x in q for x in ["1 month", "1-month", "one month", "monthly"]):
        return "1mo"
    if any(x in q for x in ["1 week", "1-week", "one week", "weekly"]):
        return "5d"
    if any(x in q for x in ["today", "daily", "1 day", "1-day"]):
        return "1d"
    return "1mo"  # default


@node_with_logging("input_validator")
def input_validator(state: AgentState) -> dict[str, Any]:
    """
    Validates user input and initializes the state.

    Pattern: Entry node — validates and normalizes input.
    Returns: Partial state dict with normalized query and initial metadata.
    """
    query = state.get("user_query", "")
    if not query or not query.strip():
        return {"error": "Empty query provided", "agent_state": "error"}

    return {
        "user_query": query.strip(),
        "agent_state": "initialized",
        "analysis_steps": ["input_validated"],
        "period": _parse_period(query),
    }
