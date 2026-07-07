"""Decision engine.

Combines the volatility + technical signals into a transparent, step-by-step
decision. Every component returns a :class:`Component` describing what it
looked at, the state it concluded, and how much it pushed the final score.
The GUI renders these components so the user can see *why* the bot acts.

Strategy in one sentence: in higher-volatility regimes, fade short-term
extremes (buy oversold dips, sell overbought spikes) in small size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from .alpaca_client import AccountState, Fundamentals
from .analysis import Analysis
from .news import NewsSentiment
from .options import OptionsActivity
from .params import StrategyParams


class Verdict(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class State(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BLOCKED = "BLOCKED"
    INFO = "INFO"


@dataclass
class Component:
    """One step in the decision process, shown as a panel in the GUI."""

    name: str
    state: State
    detail: str
    score: float = 0.0


@dataclass
class Decision:
    """Full decision output: components + aggregate verdict."""

    verdict: Verdict
    score: float
    confidence: float
    components: List[Component] = field(default_factory=list)
    summary: str = ""


# Score thresholds for acting. Tuned so a single weak signal won't trade;
# it takes a confluence (e.g. oversold RSI + low %B) to cross the line.
# These are legacy defaults; the authoritative values live in StrategyParams.
BUY_THRESHOLD = 3.0
SELL_THRESHOLD = -3.0


def _fmt(value: float, suffix: str = "", pct: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}{suffix}"


def decide(
    analysis: Analysis,
    fundamentals: Optional[Fundamentals],
    account: Optional[AccountState],
    params: Optional[StrategyParams] = None,
    options: Optional[OptionsActivity] = None,
    news: Optional[NewsSentiment] = None,
) -> Decision:
    """Run the full decision pipeline and return its components + verdict."""
    if params is None:
        params = StrategyParams()
    components: List[Component] = []

    if not analysis.has_data:
        components.append(
            Component(
                name="Data",
                state=State.INFO,
                detail="Waiting for market data from Alpaca...",
            )
        )
        return Decision(
            verdict=Verdict.HOLD,
            score=0.0,
            confidence=0.0,
            components=components,
            summary="No data yet.",
        )

    score = 0.0

    # 1) Volatility regime -- gates how aggressively we fade extremes.
    if analysis.vol_regime == "HIGH":
        vol_weight = params.weight_high
        vol_state = State.INFO
        vol_note = "High volatility -> mean-reversion edges are strongest."
    elif analysis.vol_regime == "LOW":
        vol_weight = params.weight_low
        vol_state = State.NEUTRAL
        vol_note = "Low volatility -> dampening signals; few clean setups."
    else:
        vol_weight = params.weight_normal
        vol_state = State.INFO
        vol_note = "Normal volatility."
    components.append(
        Component(
            name="Volatility regime",
            state=vol_state,
            detail=(
                f"{analysis.vol_regime} | realized vol {_fmt(analysis.realized_vol, pct=True)} "
                f"(pctile {_fmt(analysis.vol_percentile, pct=True)}), "
                f"ATR {_fmt(analysis.atr)}, BB width {_fmt(analysis.bb_width, pct=True)}"
            ),
        )
    )

    # 2) RSI -- momentum extreme.
    rsi = analysis.rsi
    if np.isfinite(rsi):
        if rsi < params.rsi_oversold:
            contrib = params.rsi_weight * vol_weight
            rsi_state, rsi_note = State.BULLISH, (
                f"Oversold (<{params.rsi_oversold:.0f}): buy pressure."
            )
        elif rsi > params.rsi_overbought:
            contrib = -params.rsi_weight * vol_weight
            rsi_state, rsi_note = State.BEARISH, (
                f"Overbought (>{params.rsi_overbought:.0f}): sell pressure."
            )
        else:
            contrib = 0.0
            rsi_state, rsi_note = State.NEUTRAL, "Mid-range: no extreme."
        score += contrib
        components.append(
            Component(
                name=f"RSI ({params.rsi_period})",
                state=rsi_state,
                detail=f"{_fmt(rsi)} - {rsi_note}",
                score=contrib,
            )
        )

    # 3) Bollinger %B -- position within the volatility envelope.
    pct_b = analysis.percent_b
    if np.isfinite(pct_b):
        if pct_b < params.bb_low:
            contrib = params.bb_weight * vol_weight
            bb_state, bb_note = State.BULLISH, "At/Below lower band: stretched down."
        elif pct_b > params.bb_high:
            contrib = -params.bb_weight * vol_weight
            bb_state, bb_note = State.BEARISH, "At/Above upper band: stretched up."
        else:
            contrib = 0.0
            bb_state, bb_note = State.NEUTRAL, "Inside bands."
        score += contrib
        components.append(
            Component(
                name="Bollinger %B",
                state=bb_state,
                detail=f"{_fmt(pct_b)} - {bb_note}",
                score=contrib,
            )
        )

    # 3b) Z-score mean reversion -- principled "stretch" measure.
    z = analysis.zscore
    if np.isfinite(z) and params.zscore_weight > 0:
        if z < -params.zscore_entry:
            contrib = params.zscore_weight * vol_weight
            z_state, z_note = State.BULLISH, (
                f"z {_fmt(z)} < -{params.zscore_entry:g}: stretched below mean."
            )
        elif z > params.zscore_entry:
            contrib = -params.zscore_weight * vol_weight
            z_state, z_note = State.BEARISH, (
                f"z {_fmt(z)} > {params.zscore_entry:g}: stretched above mean."
            )
        else:
            contrib = 0.0
            z_state, z_note = State.NEUTRAL, f"z {_fmt(z)}: near mean."
        score += contrib
        components.append(
            Component(
                name=f"Z-score ({params.zscore_period})",
                state=z_state,
                detail=z_note,
                score=contrib,
            )
        )

    # 4) Trend (fast vs slow SMA) -- light bias, not a primary driver.
    if np.isfinite(analysis.sma_fast) and np.isfinite(analysis.sma_slow):
        if analysis.sma_fast > analysis.sma_slow:
            contrib = params.trend_weight
            tr_state, tr_note = State.BULLISH, "Fast SMA above slow: up-bias."
        else:
            contrib = -params.trend_weight
            tr_state, tr_note = State.BEARISH, "Fast SMA below slow: down-bias."
        score += contrib
        components.append(
            Component(
                name=f"Trend (SMA {params.sma_fast}/{params.sma_slow})",
                state=tr_state,
                detail=(
                    f"fast {_fmt(analysis.sma_fast)} vs slow {_fmt(analysis.sma_slow)} - {tr_note}"
                ),
                score=contrib,
            )
        )

    # 5) Fundamentals / snapshot context -- gap + intraday range position.
    if fundamentals is not None:
        contrib = 0.0
        notes: List[str] = []
        fb_state = State.NEUTRAL

        prev_close = fundamentals.previous_close
        last = fundamentals.last_price
        if np.isfinite(prev_close) and prev_close > 0 and np.isfinite(last):
            gap = (last - prev_close) / prev_close
            notes.append(f"vs prev close {_fmt(gap, pct=True)}")
            if gap < -params.fund_gap_threshold:
                contrib += params.fund_gap_weight
                fb_state = State.BULLISH
                notes.append("down (dip)")
            elif gap > params.fund_gap_threshold:
                contrib -= params.fund_gap_weight
                fb_state = State.BEARISH
                notes.append("up (extended)")

        hi, lo = fundamentals.daily_high, fundamentals.daily_low
        if np.isfinite(hi) and np.isfinite(lo) and hi > lo and np.isfinite(last):
            range_pos = (last - lo) / (hi - lo)
            notes.append(f"day range pos {_fmt(range_pos, pct=True)}")
            if range_pos < params.fund_range_low:
                contrib += params.fund_range_weight
                fb_state = State.BULLISH
            elif range_pos > params.fund_range_high:
                contrib -= params.fund_range_weight
                fb_state = State.BEARISH

        if np.isfinite(fundamentals.daily_vwap) and np.isfinite(last):
            vwap = fundamentals.daily_vwap
            side = "above" if last >= vwap else "below"
            notes.append(f"{side} VWAP {_fmt(vwap)}")

        score += contrib
        components.append(
            Component(
                name="Fundamentals (snapshot)",
                state=fb_state,
                detail="; ".join(notes) if notes else "No snapshot context.",
                score=contrib,
            )
        )

    # 5c) Unusual Options Activity (UOA) -- live-only options-flow context.
    # Net options sentiment (bullish call flow vs bearish put flow) nudges the
    # score. Backtests pass options=None, so this has no effect there.
    if (
        options is not None
        and getattr(options, "available", False)
        and bool(params.use_options_flow)
    ):
        sent = float(options.sentiment)
        contrib = params.options_weight * sent
        if contrib > 0.05:
            of_state = State.BULLISH
        elif contrib < -0.05:
            of_state = State.BEARISH
        else:
            of_state = State.NEUTRAL
        pcr = options.put_call_volume_ratio
        detail = (
            f"sentiment {sent:+.2f}, P/C vol {_fmt(pcr)}, "
            f"{options.unusual_count} unusual contract(s)"
        )
        if options.flagged:
            top = options.flagged[0]
            detail += (
                f"; top: {top.option_type.upper()} {top.strike:g} "
                f"Vol/OI {top.vol_oi:.1f}x ({top.aggressor})"
            )
        score += contrib
        components.append(
            Component(
                name="Options flow (UOA)",
                state=of_state,
                detail=detail,
                score=contrib,
            )
        )

    # 5d) News sentiment -- works LIVE and in BACKTEST (causal lexicon score).
    # Net decay-weighted sentiment of recent headlines nudges the score.
    if (
        news is not None
        and getattr(news, "available", False)
        and bool(params.use_news_sentiment)
    ):
        sent = float(news.sentiment)
        contrib = params.news_weight * sent
        if contrib > 0.05:
            ns_state = State.BULLISH
        elif contrib < -0.05:
            ns_state = State.BEARISH
        else:
            ns_state = State.NEUTRAL
        head = (news.latest_headline or "")[:60]
        detail = (
            f"net {sent:+.2f}, {news.article_count} article(s)"
            + (f"; latest: \"{head}\" ({news.latest_score:+.2f})" if head else "")
        )
        score += contrib
        components.append(
            Component(
                name="News sentiment",
                state=ns_state,
                detail=detail,
                score=contrib,
            )
        )

    # 6) Risk / position limits -- can veto the raw signal.
    raw_verdict = (
        Verdict.BUY
        if score >= params.buy_threshold
        else Verdict.SELL
        if score <= params.sell_threshold
        else Verdict.HOLD
    )

    # 5b) Trend filter -- only fade pullbacks WITH the higher-timeframe trend.
    # Mean-reversion longs perform far better above a long-term MA; this vetoes
    # buying into a confirmed downtrend (it never blocks exits).
    if bool(params.use_trend_filter) and np.isfinite(analysis.trend_sma):
        above = analysis.last_price >= analysis.trend_sma
        if raw_verdict == Verdict.BUY and not above:
            raw_verdict = Verdict.HOLD
            components.append(
                Component(
                    name=f"Trend filter (SMA {params.trend_filter_period})",
                    state=State.BLOCKED,
                    detail=(
                        f"Buy vetoed: price {_fmt(analysis.last_price)} below "
                        f"long SMA {_fmt(analysis.trend_sma)} (downtrend)."
                    ),
                )
            )
        else:
            components.append(
                Component(
                    name=f"Trend filter (SMA {params.trend_filter_period})",
                    state=State.INFO,
                    detail=(
                        f"price {_fmt(analysis.last_price)} "
                        f"{'above' if above else 'below'} long SMA "
                        f"{_fmt(analysis.trend_sma)}."
                    ),
                )
            )

    max_position_shares = params.max_position_shares
    capped = max_position_shares and max_position_shares > 0
    verdict = raw_verdict
    if account is not None:
        qty = account.position_qty
        if raw_verdict == Verdict.BUY and capped and qty >= max_position_shares:
            verdict = Verdict.HOLD
            components.append(
                Component(
                    name="Risk / position",
                    state=State.BLOCKED,
                    detail=(
                        f"Buy signal blocked: holding {qty:.0f}/"
                        f"{max_position_shares} shares (at cap)."
                    ),
                )
            )
        elif raw_verdict == Verdict.SELL and qty <= 0:
            verdict = Verdict.HOLD
            components.append(
                Component(
                    name="Risk / position",
                    state=State.BLOCKED,
                    detail="Sell signal blocked: no shares held to sell.",
                )
            )
        else:
            cap_txt = f"/{max_position_shares}" if capped else ""
            components.append(
                Component(
                    name="Risk / position",
                    state=State.INFO,
                    detail=(
                        f"Holding {qty:.0f}{cap_txt} shares; "
                        f"unrealized P/L {_fmt(account.unrealized_pl, suffix=' USD')}."
                    ),
                )
            )

    confidence = float(min(1.0, abs(score) / (params.buy_threshold + 2.0)))
    summary = (
        f"{verdict.value} | score {score:+.2f} "
        f"(buy>=+{params.buy_threshold:.1f}, sell<={params.sell_threshold:.1f}), "
        f"confidence {_fmt(confidence, pct=True)}"
    )

    return Decision(
        verdict=verdict,
        score=score,
        confidence=confidence,
        components=components,
        summary=summary,
    )
