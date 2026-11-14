# Universal Knowledge Agent

An adaptive, evidence-grounded AI agent built with Python, Groq, and Gradio.

## Features

- Autonomous tool selection and execution
- Five integrated tools: web search, document search, code search, memory search, and calculator
- Adaptive multi-step reasoning with tool-result observations
- Evidence aggregation with source references
- Grounded answer generation and citation validation
- Lightweight evidence-quality warnings
- Gradio dashboard for answers, sources, execution traces, and metrics
- Automated evaluation framework

## Architecture

User Question → Groq Decision Controller → Tool Registry → Tool Execution → Observation → Adaptive Decision → Evidence Aggregation → Grounded Answer → Evidence Guard → Gradio Dashboard

The system uses up to five tool calls per request.

## Tech Stack

Python, Groq, LangChain, Gradio, Pydantic, pytest, and DuckDuckGo search.

## Running Locally

1. Create and activate a Python virtual environment.
2. Install the project's dependencies.
3. Configure the required Groq API key in a local `.env` file.
4. From the `agentic-rag` directory, run:

```powershell
python dashboard_app.py
```

The dashboard will start on a local Gradio URL.

## Tests

```powershell
python -m pytest tests/ -q
```

## Evaluation

```powershell
python evaluation/run_eval.py
```

The initial live baseline passed 4 of 4 evaluation scenarios: calculator, code search, web search, and combined web/code search.

These results demonstrate functionality on a small benchmark, not general accuracy or guaranteed factual correctness.

## Known Limitations

- Web searches can fail intermittently due to external provider behavior.
- Code search may retrieve references from tests or prompt examples rather than actual implementation.
- Citation validation confirms evidence IDs exist, not that every claim is factually supported.
- Free-tier Groq limits and network availability can affect response times.
- Local file and memory tools should not be exposed through a public dashboard without access controls.

## Security

Keep `.env`, API keys, virtual environments, and local private data out of version control. The dashboard is intended for local use.

## Project Status

Functional prototype with automated tests, live evaluation, and a Gradio user interface.