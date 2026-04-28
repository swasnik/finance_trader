"""FastAPI entrypoint for the Financial Agent."""

import json
import uuid
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


app = FastAPI(
    title="Financial Agent API",
    description="Multi-step financial analysis powered by LangGraph",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    stream: bool = False


class AnalyzeResponse(BaseModel):
    query: str
    final_analysis: Optional[str] = None
    recommendations: list[str] = []
    error: Optional[str] = None
    thread_id: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "financial-agent"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    # Import here to avoid circular imports at module load time
    from src.graphs.financial_graph import run_financial_graph, stream_financial_graph

    thread_id = request.thread_id or str(uuid.uuid4())

    if request.stream:
        # Return a streaming response of newline-delimited JSON events
        async def event_stream() -> AsyncIterator[str]:
            async for event in stream_financial_graph(
                user_query=request.query,
                thread_id=thread_id,
            ):
                yield json.dumps(event) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    try:
        result = await run_financial_graph(
            user_query=request.query,
            thread_id=thread_id,
        )

        return AnalyzeResponse(
            query=request.query,
            final_analysis=result.get("final_analysis") or None,
            recommendations=result.get("recommendations") or [],
            error=result.get("error"),
            thread_id=thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static files mount MUST come after all API routes so /health and /analyze take precedence.
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
