from src.tools.data_extractor import extract_close_prices
from src.tools.metrics import (
    calculate_correlation,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_volatility,
)


def get_all_tools():
    """Return the list of all financial calculation tools for LLM binding."""
    return [
        calculate_returns,
        calculate_volatility,
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        extract_close_prices,
        calculate_correlation,
    ]


__all__ = [
    "calculate_returns",
    "calculate_volatility",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_correlation",
    "extract_close_prices",
    "get_all_tools",
]
