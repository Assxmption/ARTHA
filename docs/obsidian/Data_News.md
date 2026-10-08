# Data: News and Sentiment

Located in `app/data/news.py`.

This module aggregates financial news and scores its sentiment locally.

## RSS Feed Aggregation
Fetches and deduplicates (via title similarity) RSS feeds from:
1. Google News
2. Yahoo Finance
3. Economic Times (Markets)
4. MoneyControl

## Sentiment Analysis (VADER)
Rather than wasting expensive LLM tokens on basic sentiment scoring, ARTHA uses a local VADER (`SentimentIntensityAnalyzer`) pass.
- **Custom Finance Lexicon**: The default VADER lexicon is updated with finance-specific weights (e.g., "bullish" = +2.5, "upgrade" = +2.0, "crash" = -3.0, "dividend" = +1.0).
- **Fallback**: If VADER is not installed, it falls back to a simple keyword frequency counter.
- **Scoring**: A compound score $\ge 0.15$ is labeled `bullish`; $\le -0.15$ is labeled `bearish`.

The output is bundled into a summary containing the overall distribution, which the Agent Crew can then reason over.
