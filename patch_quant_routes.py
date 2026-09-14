import sys
import re

file_path = "/Users/goral/Documents/multi_agent_stock_analyzer/app/api/quant_routes.py"

with open(file_path, "r") as f:
    content = f.read()

new_route = """
@quant_router.get("/fundamentals/{symbol}/narrative_stream")
async def stream_fundamentals_narrative_route(symbol: str):
    \"\"\"Stream LLM narrative for a specific symbol's fundamentals.\"\"\"
    from app.data.fundamentals import get_full_fundamentals
    from app.factstore.store import FactStore
    from app.agents.fundamentals_agent import stream_fundamentals_narrative
    from fastapi.responses import StreamingResponse
    import time
    
    try:
        rows = get_full_fundamentals(symbol.upper())
        if not rows:
            raise HTTPException(status_code=404, detail=f"No fundamental data found for {symbol}")
            
        store = FactStore()
        job_id = f"ui_fund_{symbol.upper()}_{int(time.time())}"
        for row in rows:
            row.job_id = job_id
        store.put_facts(rows)
        
        generator = stream_fundamentals_narrative(symbol.upper(), job_id, store, rows)
        
        # We can just stream the text directly since we'll read it as raw bytes or text in the frontend
        return StreamingResponse(generator, media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming fundamentals narrative for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""

# Insert right after get_fundamentals_narrative
content = content.replace(
    "def get_fundamentals_narrative(symbol: str):",
    new_route.strip() + "\n\ndef get_fundamentals_narrative(symbol: str):"
)

with open(file_path, "w") as f:
    f.write(content)
