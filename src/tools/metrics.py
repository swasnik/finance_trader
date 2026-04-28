
import numpy as np
from langchain_core.tools import tool


@tool
def calculate_returns(prices: list[float], period: str = "daily") -> dict:
    """Calculate returns from a price series.
    
    Args:
        prices: List of prices in chronological order
        period: 'daily', 'monthly', or 'annual'
    
    Returns:
        dict with total_return, average_return, period_returns
    """
    if len(prices) < 2:
        return {"error": "Need at least 2 prices"}

    arr = np.array(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"error": "No valid (finite) price data after filtering NaN/Inf values"}
    if len(arr) < 2:
        return {"error": "Need at least 2 prices"}

    period_returns = ((arr[1:] - arr[:-1]) / arr[:-1]).tolist()
    total_return = (arr[-1] - arr[0]) / arr[0]

    multiplier = {"daily": 252, "monthly": 12, "annual": 1}.get(period, 252)

    return {
        "total_return": round(total_return * 100, 4),
        "average_return": round(float(np.mean(period_returns)) * 100, 4),
        "annualized_return": round(float(np.mean(period_returns) * multiplier) * 100, 4),
        "period_returns": [round(r * 100, 4) for r in period_returns[-10:]],
    }


@tool
def calculate_volatility(prices: list[float], annualize: bool = True) -> dict:
    """Calculate price volatility (standard deviation of returns).
    
    Args:
        prices: List of prices in chronological order
        annualize: Whether to annualize (multiply by sqrt(252))
    
    Returns:
        dict with volatility percentage and risk_level
    """
    if len(prices) < 2:
        return {"error": "Need at least 2 prices"}

    arr = np.array(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"error": "No valid (finite) price data after filtering NaN/Inf values"}
    if len(arr) < 2:
        return {"error": "Need at least 2 prices"}

    returns = (arr[1:] - arr[:-1]) / arr[:-1]
    vol = float(np.std(returns))
    if annualize:
        vol *= np.sqrt(252)

    risk_level = "low" if vol < 0.15 else "medium" if vol < 0.30 else "high"

    return {
        "volatility": round(vol * 100, 4),
        "annualized": annualize,
        "risk_level": risk_level,
    }


@tool
def calculate_sharpe_ratio(prices: list[float], risk_free_rate: float = 0.05) -> dict:
    """Calculate Sharpe ratio (risk-adjusted return).
    
    Args:
        prices: List of prices in chronological order
        risk_free_rate: Annual risk-free rate (default 5% = 0.05)
    
    Returns:
        dict with sharpe_ratio and interpretation
    """
    if len(prices) < 2:
        return {"error": "Need at least 2 prices"}

    arr = np.array(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"error": "No valid (finite) price data after filtering NaN/Inf values"}
    if len(arr) < 2:
        return {"error": "Need at least 2 prices"}

    returns = (arr[1:] - arr[:-1]) / arr[:-1]
    mean_return = float(np.mean(returns)) * 252
    volatility = float(np.std(returns)) * np.sqrt(252)

    if volatility == 0:
        return {"error": "Zero volatility — cannot compute Sharpe"}

    sharpe = (mean_return - risk_free_rate) / volatility
    interpretation = "excellent" if sharpe > 2 else "good" if sharpe > 1 else "acceptable" if sharpe > 0 else "poor"

    return {"sharpe_ratio": round(sharpe, 4), "interpretation": interpretation}


@tool
def calculate_max_drawdown(prices: list[float]) -> dict:
    """Calculate maximum drawdown (worst peak-to-trough loss).
    
    Args:
        prices: List of prices in chronological order
    
    Returns:
        dict with max_drawdown percentage, peak and trough indices
    """
    if len(prices) < 2:
        return {"error": "Need at least 2 prices"}

    arr = np.array(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"error": "No valid (finite) price data after filtering NaN/Inf values"}
    if len(arr) < 2:
        return {"error": "Need at least 2 prices"}

    peak = arr[0]
    max_dd = 0.0
    peak_idx, trough_idx = 0, 0
    temp_peak_idx = 0

    for i, price in enumerate(arr):
        if price > peak:
            peak = price
            temp_peak_idx = i
        dd = (price - peak) / peak
        if dd < max_dd:
            max_dd = dd
            peak_idx = temp_peak_idx
            trough_idx = i

    return {
        "max_drawdown": round(max_dd * 100, 4),
        "peak_index": peak_idx,
        "trough_index": trough_idx,
        "severity": "severe" if max_dd < -0.30 else "moderate" if max_dd < -0.10 else "mild",
    }


@tool
def calculate_correlation(prices_a: list[float], prices_b: list[float]) -> dict:
    """Calculate correlation between two asset price series.
    
    Args:
        prices_a: List of prices for first asset in chronological order
        prices_b: List of prices for second asset in chronological order
    
    Returns:
        dict with correlation coefficient and interpretation
    """
    if len(prices_a) < 2 or len(prices_b) < 2:
        return {"error": "Need at least 2 prices for each asset"}

    min_len = min(len(prices_a), len(prices_b))
    arr_a = np.array(prices_a[:min_len], dtype=float)
    arr_b = np.array(prices_b[:min_len], dtype=float)

    finite_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[finite_mask]
    arr_b = arr_b[finite_mask]
    if len(arr_a) == 0:
        return {"error": "No valid (finite) price data after filtering NaN/Inf values"}
    if len(arr_a) < 2:
        return {"error": "Need at least 2 prices for each asset"}

    returns_a = (arr_a[1:] - arr_a[:-1]) / arr_a[:-1]
    returns_b = (arr_b[1:] - arr_b[:-1]) / arr_b[:-1]

    correlation = float(np.corrcoef(returns_a, returns_b)[0, 1])

    if abs(correlation) > 0.8:
        interpretation = "strongly correlated" if correlation > 0 else "strongly inversely correlated"
    elif abs(correlation) > 0.5:
        interpretation = "moderately correlated" if correlation > 0 else "moderately inversely correlated"
    else:
        interpretation = "weakly correlated"

    return {
        "correlation": round(correlation, 4),
        "interpretation": interpretation,
        "periods_used": min_len - 1,
    }
