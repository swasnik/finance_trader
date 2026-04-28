"""
Tests for LLM integration — provider configuration, financial tools, and node fallbacks.

All tests run without API keys (no LLM calls made).
"""

import os

import pytest

# Ensure no API keys are set for unit tests
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)


class TestLLMProvider:
    """Tests for LLM provider configuration."""

    def test_is_llm_configured_returns_false_when_no_keys(self):
        """is_llm_configured returns False when no API keys are set."""
        from src.llm.provider import is_llm_configured
        assert is_llm_configured() is False

    def test_is_llm_configured_returns_true_with_anthropic_key(self, monkeypatch):
        """is_llm_configured returns True when ANTHROPIC_API_KEY is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from src.llm.provider import is_llm_configured
        assert is_llm_configured() is True

    def test_is_llm_configured_returns_true_with_openai_key(self, monkeypatch):
        """is_llm_configured returns True when OPENAI_API_KEY is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from src.llm.provider import is_llm_configured
        assert is_llm_configured() is True

    def test_get_llm_raises_when_anthropic_key_missing(self):
        """get_llm raises ValueError when ANTHROPIC_API_KEY is not set."""
        from src.llm.provider import get_llm
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_llm(provider="anthropic")

    def test_get_llm_raises_when_openai_key_missing(self):
        """get_llm raises ValueError when OPENAI_API_KEY is not set."""
        from src.llm.provider import get_llm
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm(provider="openai")

    def test_get_llm_raises_for_unknown_provider(self):
        """get_llm raises ValueError for unknown provider."""
        from src.llm.provider import get_llm
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm(provider="unknown_provider")


class TestCalculateReturns:
    """Tests for calculate_returns tool."""

    def test_basic_returns(self):
        """Returns are computed correctly."""
        from src.tools.metrics import calculate_returns
        result = calculate_returns.invoke({"prices": [100.0, 110.0, 121.0]})
        assert "total_return" in result
        assert abs(result["total_return"] - 21.0) < 0.01

    def test_insufficient_prices(self):
        """Returns error dict for fewer than 2 prices."""
        from src.tools.metrics import calculate_returns
        result = calculate_returns.invoke({"prices": [100.0]})
        assert "error" in result

    def test_returns_structure(self):
        """Result contains expected keys."""
        from src.tools.metrics import calculate_returns
        result = calculate_returns.invoke({"prices": [100.0, 105.0, 102.0, 108.0]})
        assert "total_return" in result
        assert "average_return" in result
        assert "annualized_return" in result
        assert "period_returns" in result

    def test_declining_prices(self):
        """Negative returns for declining prices."""
        from src.tools.metrics import calculate_returns
        result = calculate_returns.invoke({"prices": [100.0, 90.0, 80.0]})
        assert result["total_return"] < 0


class TestCalculateVolatility:
    """Tests for calculate_volatility tool."""

    def test_basic_volatility(self):
        """Volatility is computed and returned."""
        from src.tools.metrics import calculate_volatility
        result = calculate_volatility.invoke({"prices": [100.0, 102.0, 99.0, 101.0, 98.0, 103.0]})
        assert "volatility" in result
        assert result["volatility"] >= 0

    def test_risk_level_classification(self):
        """Risk level is one of low/medium/high."""
        from src.tools.metrics import calculate_volatility
        result = calculate_volatility.invoke({"prices": [100.0, 101.0, 100.5, 101.5, 100.0]})
        assert result["risk_level"] in ("low", "medium", "high")

    def test_insufficient_prices(self):
        """Returns error for fewer than 2 prices."""
        from src.tools.metrics import calculate_volatility
        result = calculate_volatility.invoke({"prices": [100.0]})
        assert "error" in result

    def test_annualize_flag(self):
        """Annualized volatility differs from non-annualized."""
        from src.tools.metrics import calculate_volatility
        prices = [100.0, 102.0, 98.0, 104.0, 96.0, 105.0]
        annualized = calculate_volatility.invoke({"prices": prices, "annualize": True})
        not_annualized = calculate_volatility.invoke({"prices": prices, "annualize": False})
        # Annualized should be larger than non-annualized (multiplied by sqrt(252))
        assert annualized["volatility"] > not_annualized["volatility"]


class TestCalculateSharpeRatio:
    """Tests for calculate_sharpe_ratio tool."""

    def test_basic_sharpe(self):
        """Sharpe ratio is computed."""
        from src.tools.metrics import calculate_sharpe_ratio
        prices = [100.0 + i * 0.5 for i in range(20)]
        result = calculate_sharpe_ratio.invoke({"prices": prices})
        assert "sharpe_ratio" in result
        assert "interpretation" in result

    def test_interpretation_values(self):
        """Interpretation is one of expected values."""
        from src.tools.metrics import calculate_sharpe_ratio
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        result = calculate_sharpe_ratio.invoke({"prices": prices})
        assert result["interpretation"] in ("excellent", "good", "acceptable", "poor")

    def test_insufficient_prices(self):
        """Returns error for fewer than 2 prices."""
        from src.tools.metrics import calculate_sharpe_ratio
        result = calculate_sharpe_ratio.invoke({"prices": [100.0]})
        assert "error" in result


class TestCalculateMaxDrawdown:
    """Tests for calculate_max_drawdown tool."""

    def test_basic_drawdown(self):
        """Drawdown is computed correctly."""
        from src.tools.metrics import calculate_max_drawdown
        # Goes up to 120 then falls to 80 — 33% drawdown
        prices = [100.0, 110.0, 120.0, 100.0, 80.0]
        result = calculate_max_drawdown.invoke({"prices": prices})
        assert "max_drawdown" in result
        assert result["max_drawdown"] < 0  # drawdown is negative

    def test_no_drawdown(self):
        """Monotonically increasing prices have zero drawdown."""
        from src.tools.metrics import calculate_max_drawdown
        prices = [100.0, 105.0, 110.0, 115.0, 120.0]
        result = calculate_max_drawdown.invoke({"prices": prices})
        assert result["max_drawdown"] == 0.0

    def test_severity_classification(self):
        """Severity is one of mild/moderate/severe."""
        from src.tools.metrics import calculate_max_drawdown
        prices = [100.0, 110.0, 120.0, 100.0, 80.0]
        result = calculate_max_drawdown.invoke({"prices": prices})
        assert result["severity"] in ("mild", "moderate", "severe")

    def test_insufficient_prices(self):
        """Returns error for fewer than 2 prices."""
        from src.tools.metrics import calculate_max_drawdown
        result = calculate_max_drawdown.invoke({"prices": [100.0]})
        assert "error" in result

    def test_severe_drawdown(self):
        """Severe drawdown (>30%) is classified correctly."""
        from src.tools.metrics import calculate_max_drawdown
        prices = [100.0, 110.0, 120.0, 50.0, 40.0]  # 66% drawdown
        result = calculate_max_drawdown.invoke({"prices": prices})
        assert result["severity"] == "severe"


class TestAnalyzerFallback:
    """Tests for analyzer node fallback path (no LLM configured)."""

    def test_analyzer_returns_error_without_source_data(self):
        """Analyzer returns error when no source_data."""
        from src.nodes.node_base import analyzer
        from src.state import get_default_state
        state = get_default_state()
        result = analyzer(state)
        assert result.get("error") is not None
        assert result["agent_state"] == "error"

    def test_analyzer_fallback_with_source_data(self):
        """Analyzer returns structured fallback when source_data present and no LLM."""
        from src.nodes.node_base import analyzer
        from src.state import get_default_state
        state = get_default_state()
        state["user_query"] = "Analyze AAPL performance"
        state["source_data"] = {
            "tickers": ["AAPL"],
            "stock_prices": {
                "AAPL": [
                    {"close": 150.0, "date": "2024-01-01"},
                    {"close": 155.0, "date": "2024-01-02"},
                    {"close": 160.0, "date": "2024-01-03"},
                ]
            },
            "economic_indicators": [],
        }
        result = analyzer(state)
        assert result["agent_state"] == "analyzed"
        assert "intermediate_results" in result
        assert "summary" in result["intermediate_results"]
        assert "key_findings" in result["intermediate_results"]

    def test_analyzer_calculates_metrics_for_price_data(self):
        """Analyzer calculates financial metrics when prices are available."""
        from src.nodes.node_base import analyzer
        from src.state import get_default_state
        state = get_default_state()
        state["user_query"] = "How did AAPL perform?"
        state["source_data"] = {
            "tickers": ["AAPL"],
            "stock_prices": {
                "AAPL": [{"close": 100.0 + i} for i in range(10)]
            },
            "economic_indicators": [],
        }
        result = analyzer(state)
        assert result["agent_state"] == "analyzed"
        metrics = result["intermediate_results"].get("metrics", {})
        assert "AAPL" in metrics


class TestSynthesizerFallback:
    """Tests for synthesizer node fallback path (no LLM configured)."""

    def test_synthesizer_returns_error_without_data(self):
        """Synthesizer returns error when neither intermediate_results nor source_data."""
        from src.nodes.node_base import synthesizer
        from src.state import get_default_state
        state = get_default_state()
        result = synthesizer(state)
        assert result.get("error") is not None
        assert result["agent_state"] == "error"

    def test_synthesizer_fallback_with_intermediate_results(self):
        """Synthesizer produces final_analysis from intermediate_results."""
        from src.nodes.node_base import synthesizer
        from src.state import get_default_state
        state = get_default_state()
        state["user_query"] = "Analyze AAPL"
        state["intermediate_results"] = {
            "summary": "AAPL showed strong performance.",
            "key_findings": ["Up 10% YTD", "Low volatility"],
            "risk_assessment": "low",
            "metrics": {},
        }
        result = synthesizer(state)
        assert result["agent_state"] == "complete"
        assert result["final_analysis"] != ""
        assert len(result["recommendations"]) > 0

    def test_synthesizer_includes_synthesis_step(self):
        """Synthesizer adds synthesis_complete to analysis_steps."""
        from src.nodes.node_base import synthesizer
        from src.state import get_default_state
        state = get_default_state()
        state["intermediate_results"] = {"summary": "Test", "key_findings": [], "risk_assessment": "low", "metrics": {}}
        result = synthesizer(state)
        assert "synthesis_complete" in result["analysis_steps"]
