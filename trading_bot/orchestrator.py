"""Sentiment-analysis orchestration over the LLM/Copilot endpoint.

Builds a structured sentiment-analysis prompt for arbitrary user text (e.g. a
news headline, earnings blurb, or social post), sends it to the configured LLM
via :class:`~trading_bot.llm_client.LLMClient`, and parses the reply into a
:class:`SentimentResult`. The model is instructed to return strict JSON; the
parser is defensive (handles code fences / stray prose) and falls back to a
keyword scan so the UI always gets something usable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm_client import LLMClient, LLMError

SYSTEM_PROMPT = (
    "You are a financial sentiment-analysis assistant. Classify the market "
    "sentiment of the user's text toward the relevant asset/company. "
    "Respond with STRICT JSON only (no markdown, no commentary) using exactly "
    "these keys:\n"
    '  "label": one of "BULLISH", "BEARISH", or "NEUTRAL"\n'
    '  "score": a number from -1.0 (very bearish) to 1.0 (very bullish)\n'
    '  "confidence": a number from 0.0 to 1.0\n'
    '  "rationale": a one-sentence explanation (max 200 chars)\n'
    "Base the judgment only on the provided text. If it is not market-relevant, "
    'use "NEUTRAL" with low confidence.'
)


@dataclass
class SentimentResult:
    """Parsed sentiment output."""

    label: str  # "BULLISH" | "BEARISH" | "NEUTRAL"
    score: float  # -1.0 .. 1.0
    confidence: float  # 0.0 .. 1.0
    rationale: str
    raw: str  # the raw model reply (for transparency / debugging)


def build_messages(text: str, instruction: Optional[str] = None) -> List[Dict[str, str]]:
    """Build the chat messages for a sentiment request."""
    return [
        {"role": "system", "content": instruction or SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extract_json(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of ``raw`` (tolerates code fences/prose)."""
    if not raw:
        return None
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # Otherwise grab the first balanced-looking {...} block.
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


def _keyword_fallback(raw: str) -> SentimentResult:
    """Last-resort sentiment from keywords when JSON parsing fails."""
    low = (raw or "").lower()
    if "bullish" in low:
        return SentimentResult("BULLISH", 0.4, 0.3, "Keyword fallback: 'bullish'.", raw)
    if "bearish" in low:
        return SentimentResult("BEARISH", -0.4, 0.3, "Keyword fallback: 'bearish'.", raw)
    return SentimentResult(
        "NEUTRAL", 0.0, 0.2,
        "Could not parse a structured sentiment from the model reply.", raw,
    )


def parse_sentiment(raw: str) -> SentimentResult:
    """Parse a model reply into a :class:`SentimentResult` (never raises)."""
    obj = _extract_json(raw)
    if obj is None:
        return _keyword_fallback(raw)

    label = str(obj.get("label", "")).strip().upper()
    if label not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        # Derive from score if label is missing/odd.
        try:
            s = float(obj.get("score", 0.0))
        except (TypeError, ValueError):
            s = 0.0
        label = "BULLISH" if s > 0.15 else "BEARISH" if s < -0.15 else "NEUTRAL"

    try:
        score = _clip(float(obj.get("score", 0.0)), -1.0, 1.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        confidence = _clip(float(obj.get("confidence", 0.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        confidence = 0.0

    rationale = str(obj.get("rationale", "")).strip() or "(no rationale provided)"
    return SentimentResult(label, score, confidence, rationale[:400], raw)


def analyze_sentiment(
    llm: LLMClient, text: str, instruction: Optional[str] = None
) -> SentimentResult:
    """Run a sentiment query end-to-end. Raises :class:`LLMError` on transport
    failures (the caller shows the message); parsing itself never raises."""
    text = (text or "").strip()
    if not text:
        raise LLMError("Enter some text to analyze.")
    raw = llm.chat(build_messages(text, instruction), temperature=0.0, max_tokens=300)
    return parse_sentiment(raw)
