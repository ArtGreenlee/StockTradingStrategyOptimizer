"""News-sentiment analysis (Alpaca News API).

Unlike options/UOA data, Alpaca's News API returns **historical** articles with
publication timestamps, so news sentiment can be used in the **backtest** as
well as live -- with one hard rule: **no look-ahead**. An article published at
time T may only influence decisions at bars at/after ``T + publish_lag``.

Sentiment is scored with a compact **finance lexicon** (a curated subset of the
Loughran-McDonald style positive/negative word lists, plus negation handling).
This is intentional, not a shortcut:

  * deterministic + reproducible (essential for backtests and optimization),
  * fast (thousands of bars x hundreds of headlines, no per-article LLM calls),
  * network-free, so it runs inside the spawned backtest/search subprocess,
  * identical behavior live and in backtest.

(The LLM Orchestrator tab remains for richer, ad-hoc manual analysis.)

Per-bar aggregation applies an exponential **time decay** (half-life) so a fresh
headline matters more than a stale one, and only articles already published (by
``as_of``) are ever included.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- Finance sentiment lexicon (curated; lowercase, word-boundary matched) ---
_BULLISH = {
    "beat", "beats", "beating", "surge", "surges", "surged", "soar", "soars",
    "soared", "jump", "jumps", "jumped", "rally", "rallies", "rallied",
    "upgrade", "upgraded", "upgrades", "outperform", "outperforms", "record",
    "records", "raise", "raised", "raises", "boost", "boosted", "strong",
    "strength", "growth", "grow", "grows", "profit", "profits", "profitable",
    "gain", "gains", "gained", "rise", "rises", "rose", "buyback", "buybacks",
    "dividend", "expansion", "expand", "expands", "bullish", "optimistic",
    "approval", "approved", "wins", "win", "won", "breakthrough", "milestone",
    "exceeds", "exceeded", "topped", "tops", "accelerate", "accelerating",
    "robust", "rebound", "rebounds", "rebounded", "momentum", "high", "higher",
    "positive", "upside", "rally", "soaring", "climb", "climbs", "climbed",
}
_BEARISH = {
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "slump",
    "slumps", "slumped", "tumble", "tumbles", "tumbled", "downgrade",
    "downgraded", "downgrades", "underperform", "cut", "cuts", "cutting",
    "lawsuit", "lawsuits", "investigation", "investigated", "probe", "recall",
    "recalls", "layoff", "layoffs", "bankruptcy", "bankrupt", "fall", "falls",
    "fell", "drop", "drops", "dropped", "decline", "declines", "declined",
    "weak", "weakness", "loss", "losses", "warning", "warn", "warns", "warned",
    "halt", "halts", "halted", "fraud", "fraudulent", "default", "defaults",
    "slash", "slashed", "slashes", "bearish", "pessimistic", "concern",
    "concerns", "risk", "risks", "lower", "lowered", "negative", "downside",
    "sink", "sinks", "sank", "crash", "crashes", "crashed", "selloff",
    "sell-off", "disappoint", "disappoints", "disappointing", "struggle",
    "struggles", "struggling", "scandal", "delays", "delayed", "shortfall",
}
# Words that flip the polarity of the next sentiment word.
_NEGATIONS = {"not", "no", "never", "without", "fails", "fail", "failed",
              "lacks", "lack", "unable", "isn't", "isnt", "wasn't", "wasnt"}

_TOKEN_RE = re.compile(r"[a-zA-Z'\-]+")


def score_text(text: str) -> float:
    """Score finance sentiment of ``text`` in [-1, 1] (0 = neutral/none).

    Counts bullish vs bearish lexicon hits with simple negation flipping, then
    normalizes by the total hits so longer texts aren't inherently stronger.
    """
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    bull = 0.0
    bear = 0.0
    negate = False
    for tok in tokens:
        if tok in _NEGATIONS:
            negate = True
            continue
        polarity = 0
        if tok in _BULLISH:
            polarity = 1
        elif tok in _BEARISH:
            polarity = -1
        if polarity != 0:
            if negate:
                polarity = -polarity
            if polarity > 0:
                bull += 1
            else:
                bear += 1
        # Negation only affects the immediately following sentiment word.
        if tok not in _NEGATIONS:
            negate = False
    total = bull + bear
    if total == 0:
        return 0.0
    return float((bull - bear) / total)


@dataclass
class NewsItem:
    """A single scored news article."""

    timestamp: datetime  # publication time (tz-aware UTC)
    headline: str
    source: str
    sentiment: float  # [-1, 1]


@dataclass
class NewsSentiment:
    """Aggregated, time-decayed news sentiment as of a point in time."""

    available: bool
    as_of: Optional[datetime] = None
    sentiment: float = 0.0  # [-1, 1], decay-weighted net
    article_count: int = 0  # articles inside the lookback window
    latest_headline: str = ""
    latest_score: float = 0.0
    latest_time: Optional[datetime] = None
    note: str = ""


def empty_news(note: str) -> NewsSentiment:
    return NewsSentiment(available=False, note=note)


def _as_utc(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        return pd.Timestamp(ts).to_pydatetime()
    except Exception:
        return None


def aggregate_news(
    items: Sequence[NewsItem],
    as_of: datetime,
    lookback_minutes: float,
    half_life_minutes: float,
) -> NewsSentiment:
    """Combine recent, already-published articles into one decayed sentiment.

    Only items with ``timestamp <= as_of`` and within ``lookback_minutes`` are
    used. Each is weighted by ``0.5 ** (age / half_life)`` so fresh news
    dominates. Returns a neutral, ``available=False`` result if nothing applies.
    """
    if not items:
        return empty_news("No news in window.")

    horizon = as_of.timestamp() - lookback_minutes * 60.0
    hl = max(1.0, float(half_life_minutes)) * 60.0

    num = 0.0
    den = 0.0
    count = 0
    latest: Optional[NewsItem] = None
    for it in items:
        t = it.timestamp.timestamp()
        if t > as_of.timestamp() or t < horizon:
            continue
        age = as_of.timestamp() - t
        w = 0.5 ** (age / hl)
        num += w * it.sentiment
        den += w
        count += 1
        if latest is None or it.timestamp > latest.timestamp:
            latest = it

    if count == 0 or den <= 0:
        return empty_news("No news in window.")

    net = float(np.clip(num / den, -1.0, 1.0))
    return NewsSentiment(
        available=True,
        as_of=as_of,
        sentiment=net,
        article_count=count,
        latest_headline=latest.headline if latest else "",
        latest_score=latest.sentiment if latest else 0.0,
        latest_time=latest.timestamp if latest else None,
        note="",
    )


def items_from_frame(df: Optional[pd.DataFrame]) -> List[NewsItem]:
    """Build scored :class:`NewsItem`s from a news DataFrame.

    Expected columns: ``timestamp`` (tz-aware), ``headline``, optional
    ``summary`` and ``source``. A precomputed ``sentiment`` column is used as-is
    when present (so scoring happens once); otherwise headline+summary is scored.
    """
    if df is None or len(df) == 0:
        return []
    items: List[NewsItem] = []
    has_sent = "sentiment" in df.columns
    has_summary = "summary" in df.columns
    has_source = "source" in df.columns
    for row in df.itertuples(index=False):
        ts = _as_utc(getattr(row, "timestamp", None))
        if ts is None:
            continue
        headline = str(getattr(row, "headline", "") or "")
        if has_sent:
            sent = float(getattr(row, "sentiment", 0.0) or 0.0)
        else:
            summary = str(getattr(row, "summary", "") or "") if has_summary else ""
            sent = score_text(f"{headline}. {summary}")
        source = str(getattr(row, "source", "") or "") if has_source else ""
        items.append(NewsItem(ts, headline, source, sent))
    items.sort(key=lambda x: x.timestamp)
    return items


def score_news_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a ``sentiment`` column scored from headline+summary."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    summ = out["summary"] if "summary" in out.columns else ""
    if isinstance(summ, str):
        out["sentiment"] = out["headline"].fillna("").map(score_text)
    else:
        out["sentiment"] = [
            score_text(f"{h}. {s}")
            for h, s in zip(out["headline"].fillna(""), summ.fillna(""))
        ]
    return out


def build_news_series(
    news_df: Optional[pd.DataFrame],
    bar_times: pd.DatetimeIndex,
    lookback_minutes: float,
    half_life_minutes: float,
    publish_lag_seconds: float,
) -> List[NewsSentiment]:
    """Build a strictly-causal per-bar :class:`NewsSentiment` list for a backtest.

    For each bar time ``t`` the result aggregates only articles whose
    *effective* time (``published + publish_lag``) is ``<= t`` and within the
    lookback window. ``publish_lag`` models the delay between publication and
    when the strategy could realistically act, and guards against look-ahead.
    """
    n = len(bar_times)
    items = items_from_frame(news_df)
    if not items:
        return [empty_news("No news available.") for _ in range(n)]

    lag = float(publish_lag_seconds)
    # Effective (actionable) time for each article, sorted ascending.
    eff = sorted(
        (NewsItem(
            timestamp=datetime.fromtimestamp(
                it.timestamp.timestamp() + lag, tz=timezone.utc
            ),
            headline=it.headline, source=it.source, sentiment=it.sentiment,
        ) for it in items),
        key=lambda x: x.timestamp,
    )

    out: List[NewsSentiment] = []
    j = 0  # pointer into eff: items with eff.timestamp <= current bar time
    active: List[NewsItem] = []
    for i in range(n):
        t = _as_utc(bar_times[i])
        # Admit newly-actionable articles.
        while j < len(eff) and eff[j].timestamp <= t:
            active.append(eff[j])
            j += 1
        out.append(
            aggregate_news(active, t, lookback_minutes, half_life_minutes)
        )
    return out
