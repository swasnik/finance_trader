"""Pydantic v2 models for financial data."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class StockPrice(BaseModel):
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str = "yfinance"

    @field_validator("close", "open", "high", "low")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price must be positive, got {v}")
        return v


class EconomicIndicator(BaseModel):
    name: str
    series_id: str
    value: float
    date: datetime
    source: str = "fred"
    unit: str = ""


class NewsItem(BaseModel):
    title: str
    summary: str
    source: str
    published_at: datetime
    url: str = ""
    sentiment: Optional[float] = None  # -1 to 1


class FinancialDataBundle(BaseModel):
    """Complete bundle of data for one analysis session."""

    query: str
    tickers: list[str] = Field(default_factory=list)
    stock_prices: dict[str, list[StockPrice]] = Field(default_factory=dict)
    economic_indicators: list[EconomicIndicator] = Field(default_factory=list)
    news_items: list[NewsItem] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
