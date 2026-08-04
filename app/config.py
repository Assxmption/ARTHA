"""
Application Configuration
==========================
Environment loading and platform compatibility fixes.

LLM routing has moved to app/llm/router.py (tiered multi-provider).
This file no longer contains LLM construction or rate-limit workarounds.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Fix Windows charmap crash (emoji in stdout) ────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ── Fix CrewAI cache_breakpoint issue for non-Anthropic models ─────────────────
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass


# ── Legacy compatibility ───────────────────────────────────────────────────────
# The old get_llm() interface — redirects to the tiered router (cheap tier).
# Existing code that imports from config will still work.

def get_llm():
    """
    Legacy compatibility wrapper.

    New code should use app.llm.router directly::

        from app.llm.router import get_cheap_llm, get_mid_llm, get_deep_llm

    This function returns a cheap-tier LLM for backward compatibility
    with any old agents that haven't been migrated yet.
    """
    from app.llm.router import get_cheap_llm
    return get_cheap_llm()
