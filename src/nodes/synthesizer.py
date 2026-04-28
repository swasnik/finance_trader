"""Synthesizer node — generates the final user-facing financial report."""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.nodes.base import node_with_logging
from src.state import AgentState


class SynthesisResult(BaseModel):
    report: str = Field(description="Professional financial analysis report (3-4 paragraphs)")
    recommendations: list[str] = Field(description="List of specific, actionable recommendations")

logger = logging.getLogger(__name__)


@node_with_logging("synthesizer")
def synthesizer(state: AgentState) -> dict[str, Any]:
    """
    Synthesis node — generates final user-facing report using LLM.
    Falls back to structured text if no LLM is configured.
    """
    from src.llm.provider import get_llm, is_llm_configured

    if not state.get("intermediate_results") and not state.get("source_data"):
        return {"error": "Analysis incomplete", "agent_state": "error"}

    results = state.get("intermediate_results", {})
    query = state.get("user_query", "")

    summary = results.get("summary", "")
    key_findings = results.get("key_findings", [])
    risk = results.get("risk_assessment", "unknown")
    metrics = results.get("metrics", {})

    if is_llm_configured():
        try:
            llm = get_llm(temperature=0.3)

            prompt = f"""You are a senior financial analyst writing a client report.

USER QUERY: {query}

ANALYSIS RESULTS:
- Summary: {summary}
- Key Findings: {key_findings}
- Risk Assessment: {risk}
- Metrics: {json.dumps(metrics, indent=2) if metrics else "No metrics available"}

Write a professional, concise financial report (3-4 paragraphs) that:
1. Directly answers the user's question
2. Highlights the most important findings
3. States key risks clearly
4. Provides 2-3 specific, actionable recommendations

Be direct and professional. No fluff."""

            from langchain_core.messages import HumanMessage

            structured_llm = llm.with_structured_output(SynthesisResult)
            result = structured_llm.invoke([HumanMessage(content=prompt)])
            final_analysis = result.report

            return {
                "final_analysis": final_analysis,
                "recommendations": result.recommendations,
                "analysis_steps": ["synthesis_complete"],
                "agent_state": "complete",
                "messages": [{"role": "assistant", "content": final_analysis}],
            }
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}. Using fallback.")

    lines = [f"Financial Analysis: {query}", ""]
    if summary:
        lines.append(summary)
    if key_findings:
        lines.append("\nKey Findings:")
        for f in key_findings:
            lines.append(f"• {f}")
    lines.append(f"\nRisk Level: {risk}")

    return {
        "final_analysis": "\n".join(lines),
        "recommendations": key_findings[:3] if key_findings else ["Review data and retry with a more specific query"],
        "analysis_steps": ["synthesis_complete"],
        "agent_state": "complete",
    }
