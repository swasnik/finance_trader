"""Router (conditional edge) functions for the financial analysis graph."""

from src.state import AgentState


def router_has_data(state: AgentState) -> str:
    """
    Router node — determines next path based on available data.

    Pattern: Conditional branching at graph entry.
    Returns: Name of next node to execute.
    """
    if state.get("error"):
        return "error_handler"
    elif state.get("source_data"):
        return "analyzer"
    else:
        return "data_loader"


def router_needs_refinement(state: AgentState) -> str:
    """
    Determines if analysis needs additional refinement.

    Includes a max-iteration guard: if refinement_count >= 3 we route to
    output regardless of whether final_analysis is populated, preventing
    infinite loops in the graph.

    Pattern: Loop-back decision in graphs.
    Returns: "refine" or "output".
    """
    if state.get("refinement_count", 0) >= 3:
        return "output"
    if not state.get("final_analysis") or state.get("error"):
        return "refine"
    return "output"
