"""
Base utilities for LangGraph nodes.

Provides type conversion helpers and decorator factories used by all node modules.
"""

import logging
from typing import Any, Callable

from src.state import AgentState

logger = logging.getLogger(__name__)


def _to_python_native(obj: Any) -> Any:
    """Recursively convert numpy/non-serializable types to Python native types."""
    import numpy as np

    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_native(v) for v in obj]
    return obj


def create_node(name: str) -> Callable:
    """
    Decorator to create a typed LangGraph node (async).

    Usage:
        @create_node("process_data")
        async def process_data_node(state: AgentState) -> dict:
            return {"agent_state": "processed"}
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(state: AgentState) -> dict[str, Any]:
            try:
                logger.info(f"Node {name}: starting")
                result = await func(state)
                logger.info(f"Node {name}: completed")
                return result
            except Exception as e:
                logger.error(f"Node {name}: error - {str(e)}")
                return {"error": str(e), "agent_state": "error"}

        wrapper.__name__ = name
        return wrapper

    return decorator


def node_with_logging(name: str) -> Callable:
    """
    Lightweight node wrapper with logging only (non-async).

    For simple synchronous nodes that don't need async/await.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(state: AgentState) -> dict[str, Any]:
            try:
                logger.info(f"Node {name}: starting")
                result = func(state)
                logger.info(f"Node {name}: completed")
                return result
            except Exception as e:
                logger.error(f"Node {name}: error - {str(e)}")
                return {"error": str(e), "agent_state": "error"}

        wrapper.__name__ = name
        return wrapper

    return decorator
