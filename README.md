# Financial Agent

A multi-step financial analysis agent built with LangGraph for orchestrating complex agentic reasoning workflows.

## Project Structure

```
financial_agent/
├── src/
│   ├── agents/           # Agent definitions and interfaces
│   ├── graphs/           # Graph definitions and orchestration
│   ├── nodes/            # Node implementations (tools, logic)
│   ├── state/            # State definitions and schemas
│   └── __init__.py
├── tests/                # Test suite
├── docs/                 # Documentation
├── pyproject.toml        # Project metadata & dependencies
└── README.md
```

## Getting Started

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with API keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Core Concepts

**State Management**: Centralized `AgentState` in `src/state/agent_state.py`  
**Graph Definition**: Main orchestration logic in `src/graphs/financial_graph.py`  
**Nodes**: Individual processing steps in `src/nodes/`  
**Agents**: Tool-using agents in `src/agents/`  

## API

### Start the server

```bash
# Option 1 — module entry point
python -m src.api.main

# Option 2 — uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

**Health check**
```bash
curl http://localhost:8000/health
```

**Analyze** (blocking)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze AAPL performance"}'
```

**Analyze** (streaming — newline-delimited JSON)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze AAPL performance", "stream": true}'
```

Interactive docs available at `http://localhost:8000/docs` when the server is running.


### Langsmith Trace

<img width="1452" height="749" alt="langsmit_fta" src="https://github.com/user-attachments/assets/182a882b-e863-45a6-a5bd-012c119880be" />


### Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
ruff check src/ tests/
```

Type checking:
```bash
mypy src/
```

