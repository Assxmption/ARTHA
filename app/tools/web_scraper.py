"""
Web Scraper Tool
================
Fetches and extracts clean text content from URLs using
httpx (lightweight) with BeautifulSoup for HTML parsing.

Falls back gracefully if a URL is unreachable.
"""

import httpx
from typing import ClassVar
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup


class WebScraperInput(BaseModel):
    url: str = Field(..., description="The full URL to scrape and extract content from")
    max_chars: int = Field(
        default=2500,
        description="Maximum number of characters to return from the page content"
    )


class WebScraperTool(BaseTool):
    """
    Web scraping tool that extracts clean readable text from any URL.
    Uses httpx for HTTP requests and BeautifulSoup for HTML parsing.
    """

    name: str = "Web Scraper"
    description: str = (
        "Fetch and extract the full text content from a specific URL/webpage. "
        "Use this after finding relevant URLs from web search to get detailed "
        "information, documentation, articles, or technical content. "
        "Provide the complete URL including https://."
    )
    args_schema: type[BaseModel] = WebScraperInput

    # ClassVar so Pydantic v2 does not treat it as a model field
    HEADERS: ClassVar[dict] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def _run(self, url: str, max_chars: int = 2500) -> str:
        """Fetch URL and extract clean text content."""
        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            with httpx.Client(
                timeout=30,
                headers=self.HEADERS,
                follow_redirects=True,
            ) as client:
                response = client.get(url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Cannot scrape non-HTML content. Content-Type: {content_type}"

            return self._extract_text(url, response.text, max_chars)

        except httpx.TimeoutException:
            return f"Timeout: Could not reach {url} within 30 seconds."
        except httpx.HTTPStatusError as e:
            return f"HTTP Error {e.response.status_code}: Could not fetch {url}"
        except Exception as e:
            return f"Scraping failed for {url}: {str(e)}"

    def _extract_text(self, url: str, html: str, max_chars: int) -> str:
        """Parse HTML and extract meaningful text content."""
        soup = BeautifulSoup(html, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "advertisement", "iframe", "noscript"]):
            tag.decompose()

        # Try to find main content area
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.find(id="main") or
            soup.body
        )

        if not main_content:
            main_content = soup

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta_desc = meta_tag.get("content", "")

        # Get clean text
        text = main_content.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + "\n\n[... content truncated ...]"

        output = []
        output.append(f"📄 Web Content from: {url}")
        output.append("=" * 60)
        if title:
            output.append(f"Title: {title}")
        if meta_desc:
            output.append(f"Description: {meta_desc}")
        output.append("-" * 60)
        output.append(clean_text)
        output.append("=" * 60)

        return "\n".join(output)
