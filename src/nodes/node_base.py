"""
Backward-compatibility shim. All implementations have been split into:
  src/nodes/base.py, input_validator.py, data_loader.py, analyzer.py,
  synthesizer.py, error_handler.py, routers.py

Import from src.nodes directly instead of this module.
"""
from src.nodes.base import _to_python_native, create_node, node_with_logging
from src.nodes.analyzer import analyzer
from src.nodes.data_loader import data_loader
from src.nodes.error_handler import error_handler
from src.nodes.input_validator import input_validator
from src.nodes.routers import router_has_data, router_needs_refinement
from src.nodes.synthesizer import synthesizer

__all__ = [
    "_to_python_native",
    "create_node",
    "node_with_logging",
    "input_validator",
    "data_loader",
    "analyzer",
    "synthesizer",
    "router_has_data",
    "router_needs_refinement",
    "error_handler",
]
