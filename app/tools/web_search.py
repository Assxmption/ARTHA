"""
Web Search Tool
===============
Priority: Serper API (Google Search) → DuckDuckGo fallback

Serper (https://serper.dev):
  - Real Google Search results
  - FREE: 2,500 queries/month (no credit card)
  - Set SERPER_API_KEY in .env to activate

DuckDuckGo:
  - Automatic fallback when SERPER_API_KEY is not set
  - No API key required
"""

import os
import json
import httpx
from typing import Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on Google/DuckDuckGo")
    max_results: int = Field(default=5, description="Number of results to return (1-10)")


class WebSearchTool(BaseTool):
    """
    Google-powered web search tool.
    Uses Serper API (Google) as primary engine, DuckDuckGo as fallback.
    """

    name: str = "Web Search"
    description: str = (
        "Search the web for up-to-date information using Google Search (via Serper API) "
        "or DuckDuckGo as fallback. Provide a specific search query to get relevant "
        "results including titles, URLs, and snippets. Use this to gather current "
        "information, documentation, news, and research data."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute search with Serper (Google) priority, DuckDuckGo fallback."""
        serper_key = os.getenv("SERPER_API_KEY", "").strip()

        if serper_key:
            result = self._search_serper(query, max_results, serper_key)
            if result:
                return result

        # Fallback to DuckDuckGo
        return self._search_duckduckgo(query, max_results)

    def _search_serper(self, query: str, max_results: int, api_key: str) -> str | None:
        """
        Search using Serper API — real Google Search results.
        Docs: https://serper.dev/api-reference
        """
        try:
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "q": query,
                "num": max_results,
                "gl": "us",
                "hl": "en",
            }

            with httpx.Client(timeout=30) as client:
                response = client.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return self._format_serper_results(query, data)

        except Exception as e:
            print(f"[WebSearchTool] Serper API error: {e} — falling back to DuckDuckGo")
            return None

    def _format_serper_results(self, query: str, data: dict) -> str:
        """Format Serper API response into readable text."""
        results = []
        results.append(f"🔍 Google Search Results for: '{query}'\n")
        results.append("=" * 60)

        # Knowledge Graph (if available)
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            results.append(f"\n📌 Knowledge Graph:")
            results.append(f"  Title: {kg.get('title', '')}")
            results.append(f"  Type: {kg.get('type', '')}")
            results.append(f"  Description: {kg.get('description', '')}")
            if "attributes" in kg:
                for k, v in list(kg["attributes"].items())[:5]:
                    results.append(f"  {k}: {v}")
            results.append("")

        # Answer Box
        if "answerBox" in data:
            ab = data["answerBox"]
            results.append(f"\n✅ Answer Box:")
            results.append(f"  {ab.get('answer', ab.get('snippet', ''))}")
            results.append("")

        # Organic Results
        organic = data.get("organic", [])
        if organic:
            results.append(f"\n📄 Top Results:")
            for i, item in enumerate(organic[:8], 1):
                results.append(f"\n[{i}] {item.get('title', 'No title')}")
                results.append(f"    URL: {item.get('link', '')}")
                results.append(f"    {item.get('snippet', 'No description available')}")

        # People Also Ask
        paa = data.get("peopleAlsoAsk", [])
        if paa:
            results.append(f"\n\n❓ People Also Ask:")
            for item in paa[:3]:
                results.append(f"  • {item.get('question', '')}")
                results.append(f"    {item.get('snippet', '')}")

        results.append("\n" + "=" * 60)
        results.append("Source: Google Search via Serper API")
        return "\n".join(results)

    def _search_duckduckgo(self, query: str, max_results: int) -> str:
        """Fallback search using DuckDuckGo."""
        if not DDGS_AVAILABLE:
            return (
                f"Search unavailable: Neither Serper API key nor duckduckgo-search "
                f"package is configured. Query was: '{query}'"
            )

        try:
            results = []
            results.append(f"🔍 DuckDuckGo Search Results for: '{query}'\n")
            results.append("=" * 60)

            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))

            if not search_results:
                return f"No results found for query: '{query}'"

            results.append(f"\n📄 Top Results:")
            for i, item in enumerate(search_results, 1):
                results.append(f"\n[{i}] {item.get('title', 'No title')}")
                results.append(f"    URL: {item.get('href', '')}")
                results.append(f"    {item.get('body', 'No description available')}")

            results.append("\n" + "=" * 60)
            results.append("Source: DuckDuckGo Search (fallback)")
            return "\n".join(results)

        except Exception as e:
            return f"Search failed: {str(e)}. Query was: '{query}'"
