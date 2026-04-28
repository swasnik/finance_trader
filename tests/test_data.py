"""Tests for the data layer: schemas, cache, fetchers, and data_loader node."""

import time
from datetime import datetime

import pytest

from src.data.cache import DataCache
from src.data.fetchers import YFinanceFetcher
from src.data.schemas import FinancialDataBundle, StockPrice
from src.nodes.node_base import data_loader
from src.state import get_default_state

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestStockPriceValidation:
    def _make_price(self, **overrides) -> dict:
        base = dict(
            ticker="AAPL",
            timestamp=datetime(2024, 1, 2),
            open=180.0,
            high=185.0,
            low=179.0,
            close=182.0,
            volume=1_000_000,
        )
        base.update(overrides)
        return base

    def test_valid_price(self):
        p = StockPrice(**self._make_price())
        assert p.ticker == "AAPL"
        assert p.close == 182.0
        assert p.source == "yfinance"

    def test_negative_close_raises(self):
        with pytest.raises(Exception):
            StockPrice(**self._make_price(close=-1.0))

    def test_zero_open_raises(self):
        with pytest.raises(Exception):
            StockPrice(**self._make_price(open=0.0))

    def test_negative_high_raises(self):
        with pytest.raises(Exception):
            StockPrice(**self._make_price(high=-5.0))


class TestFinancialDataBundleSerialization:
    def test_to_dict_is_json_safe(self):
        bundle = FinancialDataBundle(query="test", tickers=["AAPL"])
        d = bundle.to_dict()
        assert isinstance(d, dict)
        assert d["query"] == "test"
        assert d["tickers"] == ["AAPL"]
        # fetched_at should be a string (JSON-safe), not a datetime object
        assert isinstance(d["fetched_at"], str)

    def test_empty_bundle(self):
        bundle = FinancialDataBundle(query="")
        d = bundle.to_dict()
        assert d["stock_prices"] == {}
        assert d["economic_indicators"] == []
        assert d["errors"] == []


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestDataCacheTTL:
    def test_set_and_get(self):
        cache = DataCache()
        cache.set("k", "v", ttl=60)
        assert cache.get("k") == "v"

    def test_expired_entry_returns_none(self):
        cache = DataCache()
        cache.set("k", "v", ttl=0)  # expires immediately
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_get_or_fetch_calls_fetcher_once(self):
        cache = DataCache()
        call_count = {"n": 0}

        def fetcher():
            call_count["n"] += 1
            return "result"

        result1 = cache.get_or_fetch("key", fetcher, ttl=60)
        result2 = cache.get_or_fetch("key", fetcher, ttl=60)

        assert result1 == result2 == "result"
        assert call_count["n"] == 1  # fetcher only called once

    def test_get_or_fetch_refetches_after_expiry(self):
        cache = DataCache()
        call_count = {"n": 0}

        def fetcher():
            call_count["n"] += 1
            return "result"

        cache.get_or_fetch("key", fetcher, ttl=0)
        time.sleep(0.01)
        cache.get_or_fetch("key", fetcher, ttl=60)

        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# YFinanceFetcher tests
# ---------------------------------------------------------------------------


class TestTickerExtraction:
    def setup_method(self):
        self.fetcher = YFinanceFetcher()

    def test_extracts_single_ticker(self):
        tickers = self.fetcher.extract_tickers_from_query("What is AAPL doing today?")
        assert "AAPL" in tickers

    def test_extracts_multiple_tickers(self):
        tickers = self.fetcher.extract_tickers_from_query("Compare MSFT and GOOGL performance")
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_filters_english_stopwords(self):
        tickers = self.fetcher.extract_tickers_from_query("What IS the best ETF FOR me?")
        assert "IS" not in tickers
        assert "THE" not in tickers
        assert "FOR" not in tickers
        assert "ETF" not in tickers

    def test_empty_query(self):
        tickers = self.fetcher.extract_tickers_from_query("")
        assert tickers == []

    def test_no_tickers_in_lowercase_query(self):
        tickers = self.fetcher.extract_tickers_from_query("what is the market doing")
        assert tickers == []


# ---------------------------------------------------------------------------
# data_loader node integration test
# ---------------------------------------------------------------------------


class TestDataLoaderNode:
    def _make_state(self, query: str) -> dict:
        state = get_default_state()
        state.update({"user_query": query})
        return state

    def test_returns_data_loaded_status(self):
        state = self._make_state("Analyze AAPL stock performance")
        result = data_loader(state)
        assert result["agent_state"] == "data_loaded"
        assert "data_loaded" in result["analysis_steps"]

    def test_source_data_populated(self):
        state = self._make_state("Analyze AAPL stock performance")
        result = data_loader(state)
        assert "source_data" in result
        assert isinstance(result["source_data"], dict)

    def test_metadata_has_tickers(self):
        state = self._make_state("Analyze AAPL stock performance")
        result = data_loader(state)
        assert "tickers_found" in result["metadata"]
        assert "AAPL" in result["metadata"]["tickers_found"]

    def test_no_tickers_does_not_crash(self):
        state = self._make_state("what is the market doing today")
        result = data_loader(state)
        assert result["agent_state"] == "data_loaded"
        assert result["metadata"]["tickers_found"] == []

    def test_source_data_has_stock_prices_key(self):
        state = self._make_state("Analyze AAPL stock performance")
        result = data_loader(state)
        assert "stock_prices" in result["source_data"]
        # AAPL prices may be empty if network unavailable, but key must exist
        assert isinstance(result["source_data"]["stock_prices"], dict)
