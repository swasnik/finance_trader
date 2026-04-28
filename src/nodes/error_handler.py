"""Error handler node — graceful failure recovery."""

import logging
from typing import Any

from src.nodes.base import node_with_logging
from src.state import AgentState

logger = logging.getLogger(__name__)


@node_with_logging("error_handler")
def error_handler(state: AgentState) -> dict[str, Any]:
    """
    Handles errors during processing.

    Pattern: Error recovery node — logs and gracefully degrades.
    Returns: Empty dict (state is unchanged; error field already set).
    """
    logger.error(f"Pipeline error: {state.get('error')}")
    return {}
