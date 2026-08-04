# Multi-Agent Research Assistant

> An enterprise-grade AI research platform powered by 4 specialized CrewAI agents.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.80-purple.svg)](https://crewai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

Instead of asking a single LLM for an answer, this system deploys **4 specialized AI agents** that collaborate like a real research team:

```
User Query
    ↓
🧭 Planner Agent     → Breaks query into research sub-tasks
    ↓
🔍 Researcher Agent  → Searches Google + scrapes web sources
    ↓
✅ Verifier Agent    → Fact-checks, detects contradictions
    ↓
✍️  Writer Agent     → Generates structured analytical report
    ↓
📄 Final Report (Markdown with full source attribution)
```

---

## Features

| Feature | Details |
|---|---|
| **Search Engine** | Google Search via Serper API (primary) → DuckDuckGo (fallback) |
| **LLM** | Groq `llama-3.3-70b-versatile` (free) · Gemini · OpenAI |
| **Output** | Structured Markdown: Summary → Findings → Analysis → Recommendations → References |
| **API** | FastAPI REST + Server-Sent Events (SSE) for real-time streaming |
| **Frontend** | Premium dark-mode SPA with live agent pipeline visualization |
| **Storage** | Reports saved as Markdown + JSON to `data/Reports/` |
| **PDF Support** | Read local PDFs from `data/PDFs/` |
| **Docker** | Full Docker + Docker Compose support |

---

## Quick Start

### 1. Clone & Install

```bash
cd "e:\multi agent"
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure API Keys

Edit `.env`:

```env
# LLM (Groq is FREE — https://console.groq.com)
GROQ_API_KEY=your_groq_key_here

# Search — PRIMARY: Serper (Google, FREE 2500/month — https://serper.dev)
SERPER_API_KEY=your_serper_key_here

# Fallback: DuckDuckGo (automatic if SERPER_API_KEY is blank)
```

> 💡 **Get your free Serper key** at [serper.dev](https://serper.dev) — 2,500 Google searches/month, no credit card needed.

### 3. Run

```bash
# Development (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or directly
python -m app.main
```

### 4. Open UI

Navigate to **http://localhost:8000** — the premium dark-mode UI loads automatically.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health + config info |
| `POST` | `/api/research` | Start new research job |
| `GET` | `/api/research/{id}` | Get job status + result |
| `GET` | `/api/research/{id}/stream` | SSE live progress stream |
| `GET` | `/api/reports` | List all saved reports |
| `GET` | `/api/reports/{id}` | Get specific report |
| `DELETE` | `/api/reports/{id}` | Delete report |

Interactive docs: **http://localhost:8000/docs**

### Example: Start Research

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare CrewAI and LangGraph for production AI agent systems"}'
```

Response:
```json
{
  "job_id": "abc123-...",
  "status": "pending",
  "message": "Research job started. Use job_id to track progress.",
  "query": "Compare CrewAI and LangGraph..."
}
```

---

## Project Structure

```
multi-agent/
├── app/
│   ├── agents/
│   │   ├── planner.py        # Research Strategist agent
│   │   ├── researcher.py     # Senior Research Analyst agent
│   │   ├── verifier.py       # Fact-Checking Specialist agent
│   │   └── writer.py         # Technical Report Writer agent
│   ├── tasks/
│   │   └── research_tasks.py # Task definitions + context chaining
│   ├── crew/
│   │   └── research_crew.py  # Crew orchestration + job management
│   ├── tools/
│   │   ├── web_search.py     # Serper (Google) → DuckDuckGo fallback
│   │   ├── web_scraper.py    # httpx + BeautifulSoup scraper
│   │   └── file_reader.py    # PDF + text file reader
│   ├── api/
│   │   └── routes.py         # FastAPI routes + SSE streaming
│   ├── schemas/
│   │   └── models.py         # Pydantic request/response models
│   ├── config.py             # LLM factory (Groq/Gemini/OpenAI)
│   └── main.py               # FastAPI app entry point
├── frontend/
│   ├── index.html            # Premium dark-mode SPA
│   ├── style.css             # Dark UI with glassmorphism
│   └── app.js                # SSE streaming + pipeline visualization
├── data/
│   ├── PDFs/                 # Place PDFs here for analysis
│   ├── Reports/              # Generated Markdown reports
│   └── Outputs/              # Raw JSON outputs
├── tests/
│   └── test_crew.py          # Pytest test suite
├── .env                      # API keys + configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Configuration

All settings in `.env`:

```env
# ── LLM Provider ─────────────────────────────────────────
MODEL_PROVIDER=groq                     # groq | gemini | openai
MODEL_NAME=llama-3.3-70b-versatile      # Model name
MAX_TOKENS=4096

# ── API Keys ──────────────────────────────────────────────
GROQ_API_KEY=gsk_...                    # FREE at console.groq.com
GEMINI_API_KEY=...                      # Optional
OPENAI_API_KEY=...                      # Optional

# ── Search (Priority: Serper → DuckDuckGo) ───────────────
SERPER_API_KEY=...                      # FREE at serper.dev

# ── App Settings ──────────────────────────────────────────
APP_PORT=8000
CREW_VERBOSE=true
MAX_SEARCH_RESULTS=5
```

---

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Background
docker-compose up -d --build

# Stop
docker-compose down
```

---

## Running Tests

```bash
# Unit tests (no LLM calls)
pytest tests/ -v

# Full integration test (calls LLMs — takes 2-5 min)
pytest tests/ -v -m slow
```

---

## Example Output

**Query**: *"Compare CrewAI and LangGraph for production AI agent systems"*

The system generates a report with:

- **Executive Summary** — 3-5 paragraph strategic overview
- **Key Findings** — 8-10 specific, evidence-based findings
- **Detailed Analysis** — In-depth coverage of both frameworks
- **Comparison Table** — Feature-by-feature matrix
- **Recommendations** — Prioritised, actionable guidance
- **References** — 10+ cited sources with URLs

---

## Roadmap

| Phase | Status | Features |
|---|---|---|
| **Phase 1** | ✅ Complete | 4-Agent Pipeline, FastAPI, Serper Search, Playwright Scraper, Premium UI |
| **Phase 2** | 🔜 Planned | GitHub MCP, Memory MCP, Redis job store |
| **Phase 3** | 🔜 Planned | RAG + Vector DB (Qdrant/ChromaDB), PDF ingestion |
| **Phase 4** | 🔜 Planned | LangGraph orchestration, multi-agent memory, production deployment |

---

## Technology Stack

- **AI Orchestration**: CrewAI 0.80+
- **LLM**: Groq `llama-3.3-70b-versatile` (free) / Gemini / OpenAI
- **Search**: Serper API (Google) + DuckDuckGo fallback
- **Web Scraping**: Playwright + BeautifulSoup4
- **Backend**: FastAPI + uvicorn
- **Streaming**: Server-Sent Events (SSE)
- **Frontend**: Vanilla HTML/CSS/JS (zero framework)
- **PDF Processing**: pypdf
- **Deployment**: Docker + Docker Compose

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Built to explore modern AI Engineering concepts:
- Agentic Workflows & Multi-Agent Systems
- LLM Application Development
- Model Context Protocol (MCP)
- Context Engineering
- Retrieval-Augmented Generation (RAG)
