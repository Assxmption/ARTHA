"""
File Reader Tool
================
Reads PDF files and plain text documents from the local filesystem.
Supports files in the data/PDFs/ directory (Filesystem MCP pattern).
"""

import os
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class FileReaderInput(BaseModel):
    file_path: str = Field(
        ...,
        description=(
            "Path to the file to read. Can be a filename in data/PDFs/ "
            "or a full absolute path. Supports PDF and text files."
        )
    )
    max_chars: int = Field(
        default=10000,
        description="Maximum characters to extract from the document"
    )


class FileReaderTool(BaseTool):
    """
    Read PDF and text files from the local filesystem.
    Primarily reads from data/PDFs/ directory.
    """

    name: str = "File Reader"
    description: str = (
        "Read content from PDF documents or text files stored locally. "
        "Provide a filename (e.g., 'architecture.pdf') to read from the data/PDFs/ "
        "directory, or provide a full file path. Supports PDF, TXT, MD files."
    )
    args_schema: type[BaseModel] = FileReaderInput

    def _run(self, file_path: str, max_chars: int = 10000) -> str:
        """Read and extract content from a local file."""
        # Resolve path
        resolved_path = self._resolve_path(file_path)

        if not resolved_path:
            return (
                f"File not found: '{file_path}'. "
                f"Place PDF files in data/PDFs/ directory."
            )

        suffix = resolved_path.suffix.lower()

        if suffix == ".pdf":
            return self._read_pdf(resolved_path, max_chars)
        elif suffix in [".txt", ".md", ".rst", ".csv"]:
            return self._read_text(resolved_path, max_chars)
        else:
            return f"Unsupported file type: {suffix}. Supported: PDF, TXT, MD, RST, CSV"

    def _resolve_path(self, file_path: str) -> Path | None:
        """Try to resolve the file path."""
        # Try as-is
        p = Path(file_path)
        if p.exists():
            return p

        # Try relative to data/PDFs/
        base_dirs = [
            Path("data/PDFs"),
            Path("data"),
            Path("."),
        ]
        for base in base_dirs:
            candidate = base / file_path
            if candidate.exists():
                return candidate

        return None

    def _read_pdf(self, path: Path, max_chars: int) -> str:
        """Extract text from PDF file."""
        if not PYPDF_AVAILABLE:
            return "PDF reading requires pypdf. Install with: pip install pypdf"

        try:
            reader = PdfReader(str(path))
            pages_text = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"--- Page {i + 1} ---\n{text}")

            full_text = "\n\n".join(pages_text)

            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "\n\n[... PDF truncated ...]"

            output = []
            output.append(f"📄 PDF Document: {path.name}")
            output.append(f"   Pages: {len(reader.pages)}")
            output.append("=" * 60)
            output.append(full_text)
            output.append("=" * 60)

            return "\n".join(output)

        except Exception as e:
            return f"Failed to read PDF '{path.name}': {str(e)}"

    def _read_text(self, path: Path, max_chars: int) -> str:
        """Read plain text file."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")

            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... file truncated ...]"

            output = []
            output.append(f"📄 File: {path.name}")
            output.append("=" * 60)
            output.append(content)
            output.append("=" * 60)

            return "\n".join(output)

        except Exception as e:
            return f"Failed to read file '{path.name}': {str(e)}"
