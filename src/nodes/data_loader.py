"""Data loader node — fetches stock prices and economic indicators."""

from typing import Any

from src.nodes.base import node_with_logging
from src.state import AgentState


@node_with_logging("data_loader")
def data_loader(state: AgentState) -> dict[str, Any]:
    """
    Loads financial data based on the user query.

    Extracts ticker symbols, fetches OHLCV prices from yfinance, and
    optionally fetches economic indicators from FRED (requires FRED_API_KEY).
    Returns: Partial state dict with source_data, agent_state, and metadata.
    """
    from src.data.cache import DataCache, get_cache
    from src.data.fetchers import FredFetcher, YFinanceFetcher
    from src.data.schemas import FinancialDataBundle, StockPrice

    from src.data.schemas import StockPrice

    query = state.get("user_query", "")
    period = state.get("period", "1mo")
    cache = get_cache()

    yf_fetcher = YFinanceFetcher()
    fred_fetcher = FredFetcher()

    tickers = yf_fetcher.extract_tickers_from_query(query)

    validated_prices: dict[str, list[StockPrice]] = {}
    ticker_info: dict[str, dict] = {}
    errors: list[str] = []

    for ticker in tickers[:5]:
        cache_key = f"stock_prices:{ticker}:{period}"
        prices = cache.get_or_fetch(
            cache_key,
            lambda t=ticker, p=period: yf_fetcher.fetch_stock_prices(t, p),
            ttl=DataCache.DEFAULT_TTLS["stock_prices"],
        )
        if prices:
            validated_prices[ticker] = prices
        else:
            errors.append(f"No price data available for {ticker}")

        info = cache.get_or_fetch(
            f"ticker_info:{ticker}",
            lambda t=ticker: yf_fetcher.get_ticker_info(t),
            ttl=DataCache.DEFAULT_TTLS["ticker_info"],
        )
        ticker_info[ticker] = info

    indicators = cache.get_or_fetch(
        "fred:key_indicators",
        fred_fetcher.fetch_key_indicators,
        ttl=DataCache.DEFAULT_TTLS["economic"],
    )

    bundle = FinancialDataBundle(
        query=query,
        tickers=tickers,
        stock_prices=validated_prices,
        economic_indicators=indicators or [],
        errors=errors,
    )
    bundle_dict = bundle.to_dict()

    fetched_at = bundle.fetched_at
    fetched_at_str = fetched_at.isoformat() if hasattr(fetched_at, "isoformat") else str(fetched_at)

    return {
        "source_data": bundle_dict,
        "ticker_info": ticker_info,
        "agent_state": "data_loaded",
        "analysis_steps": ["data_loaded"],
        "metadata": {
            "tickers_found": tickers,
            "data_fetched_at": fetched_at_str,
        },
    }
