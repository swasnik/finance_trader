"""Real data fetchers for yfinance and FRED."""

import logging
import os
import re
from typing import Optional

from src.data.schemas import EconomicIndicator, StockPrice

logger = logging.getLogger(__name__)


class YFinanceFetcher:
    def fetch_stock_prices(self, ticker: str, period: str = "1mo") -> list[StockPrice]:
        """Fetch historical OHLCV prices. period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            prices = []
            for ts, row in hist.iterrows():
                try:
                    prices.append(
                        StockPrice(
                            ticker=ticker,
                            timestamp=ts.to_pydatetime(),
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=int(row["Volume"]),
                        )
                    )
                except Exception as row_err:
                    logger.debug(f"Skipping row for {ticker}: {row_err}")
            return prices
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            return []

    def get_ticker_info(self, ticker: str) -> dict:
        """Get company info, sector, market cap etc."""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info
            result = {}
            for k, v in info.items():
                if v is None:
                    continue
                # Ensure JSON-serializable (convert numpy scalar types)
                if hasattr(v, "item"):
                    v = v.item()
                result[k] = v
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch ticker info for {ticker}: {e}")
            return {"symbol": ticker, "error": "info unavailable"}

    def extract_tickers_from_query(self, query: str) -> list[str]:
        """Extract ticker symbols from natural language query."""
        tickers = re.findall(r"\b[A-Z]{1,5}\b", query)
        stopwords = {
            "I", "A", "AN", "THE", "IN", "ON", "AT", "TO", "FOR", "OF",
            "AND", "OR", "IS", "IT", "BE", "AS", "MY", "WHAT", "HOW",
            "WHY", "WHEN", "ARE", "NOT", "WITH", "FROM", "BY", "US",
            "WE", "DO", "IF", "ALL",
            # Finance / business acronyms
            "ROI", "CEO", "CFO", "CTO", "COO", "IPO", "YTD", "QOQ", "YOY",
            "EPS", "PE", "PB", "AM", "PM", "ETF", "NAV", "AUM", "GDP",
            "CPI", "PPI", "PMI", "FED", "SEC", "FDA", "DOJ", "IRS",
            "LLC", "INC", "LTD", "PLC", "ESP",
            # Tech / data acronyms
            "API", "URL", "SQL", "CSV", "JSON", "XML", "HTML", "CSS",
            # Currencies
            "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "HKD", "CNY",
            # Crypto
            "BTC", "ETH",
        }
        return [t for t in tickers if t not in stopwords]


class FredFetcher:
    DEFAULT_SERIES = [
        "GDP", "FEDFUNDS", "CPIAUCSL", "UNRATE",
        "DGS10",          # 10-Year Treasury yield
        "DGS2",           # 2-Year Treasury yield
        "T10Y2Y",         # 10Y-2Y yield curve spread
        "VIXCLS",         # VIX - CBOE Volatility Index
        "BAMLH0A0HYM2",   # ICE BofA High Yield spread
    ]

    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from fredapi import Fred

                self._client = Fred(api_key=self.api_key)
            except ImportError:
                logger.warning("fredapi not installed")
        return self._client

    def fetch_indicator(self, series_id: str) -> Optional[EconomicIndicator]:
        client = self._get_client()
        if not client:
            return None
        try:
            series = client.get_series(series_id)
            latest_date = series.index[-1]
            latest_value = float(series.iloc[-1])
            return EconomicIndicator(
                name=series_id,
                series_id=series_id,
                value=latest_value,
                date=latest_date.to_pydatetime(),
            )
        except Exception as e:
            logger.warning(f"FRED fetch failed for {series_id}: {e}")
            return None

    def fetch_key_indicators(self) -> list[EconomicIndicator]:
        return [
            ind
            for sid in self.DEFAULT_SERIES
            if (ind := self.fetch_indicator(sid)) is not None
        ]
