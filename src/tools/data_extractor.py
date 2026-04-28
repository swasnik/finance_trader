from langchain_core.tools import tool


@tool
def extract_close_prices(source_data: dict, ticker: str) -> dict:
    """Extract close prices for a ticker from the source_data dict structure.
    
    Args:
        source_data: The FinancialDataBundle.to_dict() output
        ticker: The ticker symbol to extract prices for
    
    Returns:
        dict with ticker, prices (list of floats), and count
    """
    stock_prices = source_data.get("stock_prices", {})

    if ticker not in stock_prices:
        available = list(stock_prices.keys())
        return {
            "error": f"Ticker '{ticker}' not found",
            "available_tickers": available,
        }

    price_records = stock_prices[ticker]
    closes = [
        record.get("close", 0.0)
        for record in price_records
        if record.get("close") is not None
    ]

    return {
        "ticker": ticker,
        "prices": closes,
        "count": len(closes),
    }
