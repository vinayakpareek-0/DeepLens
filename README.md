# DeepLens — Autonomous Research Agent

An autonomous multi-tool research agent built with **LangGraph** that decomposes queries, researches in parallel, and produces structured reports — with human-in-the-loop review.

## Tech Stack

- **Framework:** LangGraph
- **LLM:** Gemini 2.0 Flash (via OpenRouter)
- **Tools:** Tavily, Wikipedia, ArXiv
- **Validation:** Pydantic v2
- **Tracing:** LangSmith

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # Add your API keys
python agent.py
```

## LangGraph Studio

```bash
langgraph dev
```

## License

MIT
