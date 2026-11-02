# Data: News & Sentiment Ingestion (FinBERT + VADER Ensemble)

**Status**: Implemented & Tested  
**Path**: `app/data/news.py`  
**Consuming Agent**: [[Sentiment_Agent]]  

## Overview
The **News and Sentiment Module** aggregates ticker-scoped financial news across Indian equities and MCX commodities and computes deterministic sentiment scores locally.

> [!IMPORTANT]
> **AGENTS.md Rule 1 Compliance**: All sentiment *scoring* happens deterministically in local Python code using a pre-trained transformer model and finance-augmented lexicon. **Zero LLM tokens are consumed to score sentiment.** LLM API quota is touched solely during narration downstream via the [[Sentiment_Agent]].

---

## Technical Architecture

### 1. Multi-Feed RSS Ingestion
Fetches and deduplicates (via string similarity matching) RSS news feeds across:
1. **Google News (India)**: `https://news.google.com/rss/search?q={query}+stock&hl=en-IN`
2. **Yahoo Finance**: `https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN`
3. **Economic Times (Markets)**: `https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms`
4. **MoneyControl**: `https://www.moneycontrol.com/rss/latestnews.xml`

A local memory cache with a **10-minute TTL** (`_CACHE_TTL = 600s`) prevents duplicate network requests across agent runs.

---

### 2. Dual Ensemble Sentiment Scorer (`EnsembleSentimentScorer`)
ARTHA combines two complementary sentiment architectures:
- **FinBERT (`ProsusAI/finbert`) — 65% Weight**:
  - Domain-specific BERT transformer fine-tuned on financial corpora.
  - Exceptionally accurate on earnings surprises, M&A announcements, executive turnover, and regulatory disclosures.
  - Runs locally on CPU ($\sim 15\text{ ms}$ per headline, $\sim 438\text{ MB}$ footprint).
  - Outputs 3-class softmax probabilities: $p_{\text{pos}}, p_{\text{neg}}, p_{\text{neu}}$.
  - Raw score: $s_{\text{FinBERT}} = p_{\text{pos}} - p_{\text{neg}} \in [-1.0, +1.0]$.
- **VADER (Finance-Augmented Lexicon) — 35% Weight**:
  - Rule-based, zero-latency lexical analyzer.
  - Standard VADER lexicon augmented with an Indian and global financial vocabulary (e.g., `"bullish": +2.5`, `"upgrade": +2.0`, `"crash": -3.0`, `"dividend": +1.0`, `"headwinds": -1.5`).
  - Raw score: $s_{\text{VADER}} \in [-1.0, +1.0]$.

### 3. Ensemble Synthesis & Directional Confidence
$$\text{Compound Score} = 0.65 \times s_{\text{FinBERT}} + 0.35 \times s_{\text{VADER}}$$
- **Bullish**: $\text{Compound} \ge +0.15$
- **Bearish**: $\text{Compound} \le -0.15$
- **Neutral**: $-0.15 < \text{Compound} < +0.15$

#### Disagreement & Confidence Metric:
The ensemble explicitly tracks model agreement:
$$\Delta = |s_{\text{FinBERT}} - s_{\text{VADER}}|$$
If the two models diverge in direction or if $\Delta > 0.30$, the ensemble reduces `confidence` and flags the output for conservative downstream handling.

---

## Output Data Model (`SentimentResult`)
```python
@dataclass
class SentimentResult:
    compound: float          # [-1.0, +1.0] final ensemble score
    label: str               # "bullish" | "bearish" | "neutral"
    finbert_score: float     # [-1.0, +1.0] FinBERT contribution
    vader_score: float       # [-1.0, +1.0] VADER contribution
    confidence: float        # [0.0, 1.0] agreement-based confidence
    positive: float          # probability of positive sentiment
    negative: float          # probability of negative sentiment
    neutral_prob: float      # probability of neutral sentiment
    method: str              # "ensemble" | "vader_only" | "finbert_only"
```

---

## System Connections
- **Data Layer Hierarchy**: Sits under [[Data_Layer]] alongside [[Data_NSE]], [[Data_NSE_FNO]], and [[Data_Fundamentals]].
- **Downstream Agent**: Ingested by `run_sentiment_analysis()` in [[Sentiment_Agent]], which writes typed `NewsSignal` records to the [[Fact_Store]].
- **Crew Integration**: Executed in Stage 1b of `AnalysisCrew` (`app/crew/analysis_crew.py`).
