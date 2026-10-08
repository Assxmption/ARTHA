import sys
import re

file_path = "/Users/goral/Documents/multi_agent_stock_analyzer/app/agents/fundamentals_agent.py"

with open(file_path, "r") as f:
    content = f.read()

# I will add the streaming function right after narrate_fundamentals
new_function = """

def stream_fundamentals_narrative(
    symbol: str,
    job_id: str,
    store: FactStore,
    facts: Optional[list[FundamentalRow]] = None,
):
    \"\"\"
    Streaming version of narrate_fundamentals. Yields text chunks, and
    upon completion, validates citations and writes to FactStore.
    \"\"\"
    from app.factstore.schemas import RiskFlag, Severity, NarrativeFact
    from litellm import completion
    import logging

    if facts is None:
        all_facts = store.get_facts_for_symbol(job_id, symbol)
        facts = [f for f in all_facts if isinstance(f, FundamentalRow)]

    if not facts:
        logger.warning(f"No facts found for {symbol}. Cannot stream narrative.")
        yield "## Fundamental Analysis\\n\\n*No data available to narrate.*"
        return

    categorized = _categorize_facts(facts)
    facts_section = _format_categorized_facts(categorized)

    prompt = _NARRATION_PROMPT.format(
        symbol=symbol,
        facts_section=facts_section,
    )

    try:
        llm = get_cheap_llm()
        response = completion(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=llm.api_key,
            base_url=llm.base_url if hasattr(llm, "base_url") else None,
            stream=True
        )

        full_text = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield delta
                
    except Exception as e:
        logger.error(f"Streaming LLM failed for {symbol}: {e}")
        yield "\\n\\n*Error generating narrative.*"
        return

    # Post-processing: Validate and save to FactStore
    valid_ids, invalid_ids = _validate_citations(full_text, facts, job_id, store)
    
    if invalid_ids:
        risk_flag = RiskFlag(
            job_id=job_id,
            source=f"fundamentals_agent_narration_{symbol}",
            symbol_or_pair=symbol,
            flag_type="uncitable_narrative_claim",
            detail=f"Narrative contains {len(invalid_ids)} invalid citations: {invalid_ids[:5]}.",
            severity=Severity.WARNING,
        )
        store.put_fact(risk_flag)

    narrative_fact = NarrativeFact(
        job_id=job_id,
        source=f"fundamentals_agent_narration_{symbol}",
        agent_name="fundamentals_agent",
        section="fundamentals_analysis",
        content=full_text,
        referenced_fact_ids=valid_ids,
    )
    store.put_fact(narrative_fact)
    logger.info(f"Wrote Streaming NarrativeFact for {symbol}")

"""

content += new_function

with open(file_path, "w") as f:
    f.write(content)
