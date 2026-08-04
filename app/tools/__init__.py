"""
Tools Package — Web Search, Web Scraper, File Reader
Priority: Serper (Google) → DuckDuckGo fallback
"""
from app.tools.web_search import WebSearchTool
from app.tools.web_scraper import WebScraperTool
from app.tools.file_reader import FileReaderTool

__all__ = ["WebSearchTool", "WebScraperTool", "FileReaderTool"]
