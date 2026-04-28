"""Analyzer node — core LLM-powered financial analysis."""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.nodes.base import _to_python_native, node_with_logging
from src.state import AgentState


class AnalysisResult(BaseModel):
    summary: str = Field(description="Brief summary of the financial analysis")
    key_findings: list[str] = Field(description="List of key findings from the data")
    risk_assessment: str = Field(description="Assessment of risks identified")
    market_context: str = Field(description="Broader market context")
    confidence: str = Field(description="Confidence level: high, medium, or low")

logger = logging.getLogger(__name__)


@node_with_logging("analyzer")
def analyzer(state: AgentState) -> dict[str, Any]:
    """
    Core analysis node — uses LLM to analyze financial data.
    Falls back to structured summary if no LLM is configured.
    Increments refinement_count on each run to guard against infinite loops.
    """
    from src.llm.provider import get_llm, is_llm_configured

    if not state.get("source_data"):
        return {"error": "No source data available", "agent_state": "error"}

    refinement_count = state.get("refinement_count", 0) + 1

    source_data = state["source_data"]
    query = state.get("user_query", "")

    tickers = source_data.get("tickers", [])
    stock_prices = source_data.get("stock_prices", {})
    economic_indicators = source_data.get("economic_indicators", [])

    data_summary = f"Tickers analyzed: {', '.join(tickers) if tickers else 'none identified'}\n"

    for ticker, prices in stock_prices.items():
        if prices:
            first_close = prices[0].get("close", 0)
            last_close = prices[-1].get("close", 0)
            pct_change = ((last_close - first_close) / first_close * 100) if first_close else 0
            data_summary += (
                f"{ticker}: {len(prices)} days, close range "
                f"${first_close:.2f}→${last_close:.2f} ({pct_change:+.1f}%)\n"
            )

    if economic_indicators:
        data_summary += "\nKey Economic Indicators:\n"
        for ind in economic_indicators[:4]:
            data_summary += f"  {ind.get('series_id', ind.get('name', '?'))}: {ind.get('value', '?')}\n"

    if ticker_info := state.get("ticker_info", {}):
        data_summary += "\n\nFundamental Data:\n"
        for ticker, info in ticker_info.items():
            if "error" not in info:
                data_summary += f"  {ticker}: sector={info.get('sector', 'N/A')}, "
                data_summary += f"marketCap={info.get('marketCap', 'N/A')}, "
                data_summary += f"trailingPE={info.get('trailingPE', 'N/A')}, "
                data_summary += f"forwardEPS={info.get('forwardEps', 'N/A')}\n"

    metrics = {}
    for ticker, prices in stock_prices.items():
        closes = [p.get("close", 0) for p in prices if p.get("close")]
        if len(closes) >= 2:
            try:
                from src.tools.metrics import (
                    calculate_max_drawdown,
                    calculate_returns,
                    calculate_sharpe_ratio,
                    calculate_volatility,
                )

                metrics[ticker] = _to_python_native({
                    "returns": calculate_returns.invoke({"prices": closes}),
                    "volatility": calculate_volatility.invoke({"prices": closes}),
                    "sharpe": calculate_sharpe_ratio.invoke({"prices": closes}),
                    "drawdown": calculate_max_drawdown.invoke({"prices": closes}),
                })
            except Exception as e:
                logger.warning(f"Metrics calculation failed for {ticker}: {e}")

    if is_llm_configured():
        try:
            llm = get_llm(temperature=0)

            prompt = f"""You are a professional financial analyst. Analyze the following financial data and provide a structured assessment.

USER QUERY: {query}

FINANCIAL DATA:
{data_summary}

CALCULATED METRICS:
{json.dumps(metrics, indent=2) if metrics else "No metrics available"}"""

            from langchain_core.messages import HumanMessage

            structured_llm = llm.with_structured_output(AnalysisResult)
            analysis_result = structured_llm.invoke([HumanMessage(content=prompt)])
            result_dict = analysis_result.model_dump()

            return {
                "intermediate_results": {**result_dict, "metrics": metrics, "data_summary": data_summary},
                "analysis_steps": ["llm_analysis_complete"],
                "agent_state": "analyzed",
                "refinement_count": refinement_count,
                "messages": [{"role": "assistant", "content": result_dict["summary"]}],
            }
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}. Falling back to structured summary.")

    fallback_result = {
        "summary": f"Analysis of query: '{query}'. Found {len(tickers)} tickers: {', '.join(tickers) if tickers else 'none'}.",
        "key_findings": [
            f"Data available for: {', '.join(stock_prices.keys())}" if stock_prices else "No price data fetched"
        ],
        "risk_assessment": "unknown — LLM not configured",
        "market_context": (
            f"{len(economic_indicators)} economic indicators fetched"
            if economic_indicators
            else "No economic data"
        ),
        "confidence": "low",
    }

    return {
        "intermediate_results": {**fallback_result, "metrics": metrics, "data_summary": data_summary},
        "analysis_steps": ["structured_analysis_complete"],
        "agent_state": "analyzed",
        "refinement_count": refinement_count,
    }
