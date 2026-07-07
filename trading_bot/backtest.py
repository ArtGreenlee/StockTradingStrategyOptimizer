"""Event-driven backtest / simulation engine.

Replays historical bars through the *same* analysis + decision engine the live
bot uses, so a simulation faithfully reflects how the strategy would behave.

Key feature: **artificial trade latency**. A decision made on bar *i* does not
fill instantly -- it fills on the bar that lands ``latency_seconds`` later (at
that bar's opening price), plus optional slippage and commission. This lets you
see how execution delay erodes (or occasionally helps) a fast mean-reversion
strategy.

No look-ahead: indicators are causal, the regime percentile is computed over a
trailing window, and synthetic intraday fundamentals only use bars already
seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .alpaca_client import AccountState, Fundamentals
from .analysis import Analysis, compute_indicators, regime_from_percentile
from .decision import Verdict, decide
from .news import build_news_series
from .params import SimConfig, StrategyParams
from .risk import PositionState, exit_reason, target_entry_size


@dataclass
class SimTrade:
    """A single filled order in the simulation."""

    decision_time: datetime
    fill_time: datetime
    side: str
    signal_price: float  # close at decision time (the "ideal" fill)
    fill_price: float  # actual fill after latency + slippage
    qty: int
    realized_pl: Optional[float]  # for SELLs, P/L vs average cost
    cash_after: float
    position_after: int
    reason: str = "signal"  # "signal" | stop-loss | take-profit | trailing-stop | time-stop


@dataclass
class BacktestResult:
    """Everything the GUI needs to chart + summarize a run."""

    indicators: pd.DataFrame  # full indicator frame (for price/vol/RSI charts)
    equity: pd.Series
    buy_hold: pd.Series
    drawdown: pd.Series
    trades: List[SimTrade]
    buy_markers: List[Tuple[datetime, float]]
    sell_markers: List[Tuple[datetime, float]]
    stats: dict
    params: StrategyParams
    sim: SimConfig


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of each value within its trailing ``window`` (inclusive).

    Mirrors the live engine, which ranks current realized vol within the
    lookback window -- but here it is strictly causal (no future bars).
    """
    window = max(5, int(window))

    def _rank(a: np.ndarray) -> float:
        return float((a <= a[-1]).mean())

    return series.rolling(window, min_periods=5).apply(_rank, raw=True)


def _synth_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive causal intraday 'fundamentals' from the bar history.

    Alpaca does not provide historical snapshots, so we reconstruct the same
    fields the live snapshot supplies (previous close, intraday range, VWAP)
    using only bars at or before each row -- safe for backtesting.
    """
    out = pd.DataFrame(index=df.index)
    dates = pd.Series(df.index.date, index=df.index)
    vwap = df["vwap"] if "vwap" in df.columns else df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

    grp = df.groupby(dates)
    out["last_price"] = df["close"]
    out["daily_open"] = grp["open"].transform("first")
    out["daily_high"] = grp["high"].cummax()
    out["daily_low"] = grp["low"].cummin()
    out["daily_close"] = df["close"]  # proxy: latest close so far
    cum_vol = volume.groupby(dates).cumsum()
    cum_pv = (vwap * volume).groupby(dates).cumsum()
    out["daily_volume"] = cum_vol
    out["daily_vwap"] = (cum_pv / cum_vol.replace(0.0, np.nan)).fillna(df["close"])

    # Previous session close mapped onto every bar of the following session.
    session_last = df["close"].groupby(dates).last()
    prev_session = session_last.shift(1)
    out["previous_close"] = dates.map(prev_session)
    return out


def _analysis_at(ind: pd.DataFrame, i: int, regime: str, pctile: float) -> Analysis:
    """Build a lightweight Analysis for bar ``i`` from precomputed columns."""
    row = ind.iloc[i]
    return Analysis(
        frame=ind.iloc[i : i + 1],
        last_price=float(row["close"]),
        sma_fast=float(row.get("sma_fast", np.nan)),
        sma_slow=float(row.get("sma_slow", np.nan)),
        bb_upper=float(row.get("bb_upper", np.nan)),
        bb_lower=float(row.get("bb_lower", np.nan)),
        bb_mid=float(row.get("bb_mid", np.nan)),
        percent_b=float(row.get("percent_b", np.nan)),
        bb_width=float(row.get("bb_width", np.nan)),
        rsi=float(row.get("rsi", np.nan)),
        atr=float(row.get("atr", np.nan)),
        realized_vol=float(row.get("realized_vol", np.nan)),
        vol_regime=regime,
        vol_percentile=pctile,
        zscore=float(row.get("zscore", np.nan)),
        trend_sma=float(row.get("trend_sma", np.nan)),
    )


def _fundamentals_at(fund: pd.DataFrame, i: int) -> Fundamentals:
    row = fund.iloc[i]
    return Fundamentals(
        last_price=float(row["last_price"]),
        daily_open=float(row["daily_open"]),
        daily_high=float(row["daily_high"]),
        daily_low=float(row["daily_low"]),
        daily_close=float(row["daily_close"]),
        daily_volume=float(row["daily_volume"]),
        daily_vwap=float(row["daily_vwap"]),
        previous_close=float(row["previous_close"]),
        as_of=fund.index[i].to_pydatetime(),
    )


def _compute_stats(
    equity: pd.Series,
    buy_hold: pd.Series,
    trades: List[SimTrade],
    sim: SimConfig,
    exposure_fraction: float,
) -> dict:
    start_eq = sim.starting_cash
    final_eq = float(equity.iloc[-1]) if len(equity) else start_eq
    total_return = (final_eq / start_eq - 1.0) if start_eq else 0.0
    bh_return = (
        (float(buy_hold.iloc[-1]) / start_eq - 1.0) if start_eq and len(buy_hold) else 0.0
    )

    # Max drawdown on the equity curve.
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0.0, np.nan)
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Sharpe from per-bar returns, annualized for the bar size.
    rets = equity.pct_change().dropna()
    if len(rets) > 1 and rets.std(ddof=0) > 0:
        bars_per_year = 252.0 * (23_400.0 / max(1, sim.bar_seconds))
        sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    sells = [t for t in trades if t.realized_pl is not None]
    wins = [t for t in sells if t.realized_pl > 0]
    win_rate = (len(wins) / len(sells)) if sells else 0.0
    realized_total = float(sum(t.realized_pl for t in sells)) if sells else 0.0

    return {
        "final_equity": final_eq,
        "total_return": total_return,
        "buy_hold_return": bh_return,
        "num_trades": len(trades),
        "num_round_trips": len(sells),
        "win_rate": win_rate,
        "realized_pl": realized_total,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "exposure": exposure_fraction,
    }


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
    sim: SimConfig,
    progress: Optional[Callable[[int, int], None]] = None,
    news_df: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Replay ``df`` bar-by-bar and return a full :class:`BacktestResult`.

    Args:
        df: OHLCV(+vwap) bars indexed by tz-aware timestamp.
        params: tunable strategy parameters.
        sim: simulation parameters (cash, latency, slippage, commission).
        progress: optional callback ``(done, total)`` for a progress bar.
        news_df: optional news DataFrame (timestamp, headline, summary[,
            sentiment]) for the news-sentiment signal. Applied strictly
            causally (a headline only affects bars at/after its publish+lag).
    """
    if df is None or df.empty:
        raise ValueError("No data to backtest. Download history first.")

    df = df.sort_index()
    n = len(df)

    ind = compute_indicators(df, params)
    pctile_series = _rolling_percentile(ind["realized_vol"], sim.lookback_bars)
    fund = _synth_fundamentals(df)

    # Causal per-bar news sentiment (empty list of NewsSentiment if disabled /
    # no data). Built once; strictly no look-ahead.
    if news_df is not None and len(news_df) > 0 and bool(params.use_news_sentiment):
        news_series = build_news_series(
            news_df, df.index,
            lookback_minutes=params.news_lookback_minutes,
            half_life_minutes=params.news_half_life_minutes,
            publish_lag_seconds=params.news_publish_lag_seconds,
        )
    else:
        news_series = None

    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    atr_arr = ind["atr"].to_numpy(dtype=float)
    times = df.index

    latency_bars = sim.latency_bars
    slip = sim.slippage_bps / 10_000.0

    cash = float(sim.starting_cash)
    state = PositionState()
    # Pending fills: (fill_idx, side, qty, decision_idx, signal_price, reason).
    pending: List[Tuple[int, Verdict, int, int, float, str]] = []
    trades: List[SimTrade] = []
    buy_markers: List[Tuple[datetime, float]] = []
    sell_markers: List[Tuple[datetime, float]] = []
    equity = np.empty(n, dtype=float)
    in_market_bars = 0

    for i in range(n):
        price = close[i]
        atr = atr_arr[i]
        regime = regime_from_percentile(float(pctile_series.iloc[i]), params)

        # Advance the holding clock / trailing peak for the open position.
        state.step(price)

        has_pending = bool(pending)

        # 1) Exit checks first (defined-risk): force-close on ATR stop / take-
        #    profit / trailing / time stop. One position at a time, all-out.
        if state.shares > 0 and not has_pending:
            reason = exit_reason(state, price, atr, params)
            if reason is not None:
                pending.append(
                    (min(i + latency_bars, n - 1), Verdict.SELL, state.shares,
                     i, price, reason)
                )
                has_pending = True

        # 2) Signal-driven entries/exits (only when nothing is in flight).
        if not has_pending:
            analysis = _analysis_at(ind, i, regime, float(pctile_series.iloc[i]))
            fundamentals = _fundamentals_at(fund, i)
            equity_now = cash + state.shares * price
            acct = AccountState(
                cash=cash,
                portfolio_value=equity_now,
                buying_power=cash,
                position_qty=float(state.shares),
                avg_entry_price=state.avg_entry,
                unrealized_pl=(price - state.avg_entry) * state.shares,
                market_value=state.shares * price,
            )
            decision = decide(analysis, fundamentals, acct, params,
                               news=news_series[i] if news_series else None)
            if decision.verdict == Verdict.BUY and state.shares == 0:
                qty = target_entry_size(equity_now, cash, price, atr, params)
                if qty > 0:
                    pending.append(
                        (min(i + latency_bars, n - 1), Verdict.BUY, qty, i, price,
                         "signal")
                    )
            elif decision.verdict == Verdict.SELL and state.shares > 0:
                pending.append(
                    (min(i + latency_bars, n - 1), Verdict.SELL, state.shares,
                     i, price, "signal")
                )

        # 3) Settle any fills whose latency window ends on this bar.
        still_pending: List[Tuple[int, Verdict, int, int, float, str]] = []
        for entry in pending:
            fidx, side, oqty, di, sigp, reason = entry
            if fidx != i:
                still_pending.append(entry)
                continue
            base = open_[i] if latency_bars > 0 else close[i]
            if side == Verdict.BUY:
                fill_price = base * (1.0 + slip)
                state.open(oqty, fill_price)
                cash -= fill_price * oqty + sim.commission_per_trade
                realized = None
                buy_markers.append((times[i].to_pydatetime(), fill_price))
            else:
                fill_price = base * (1.0 - slip)
                realized = (fill_price - state.avg_entry) * oqty
                cash += fill_price * oqty - sim.commission_per_trade
                state.reset()
                sell_markers.append((times[i].to_pydatetime(), fill_price))
            trades.append(
                SimTrade(
                    decision_time=times[di].to_pydatetime(),
                    fill_time=times[i].to_pydatetime(),
                    side=side.value,
                    signal_price=sigp,
                    fill_price=fill_price,
                    qty=oqty,
                    realized_pl=realized,
                    cash_after=cash,
                    position_after=state.shares,
                    reason=reason,
                )
            )
        pending = still_pending

        equity[i] = cash + state.shares * price
        if state.shares > 0:
            in_market_bars += 1
        if progress is not None and (i % 250 == 0 or i == n - 1):
            progress(i + 1, n)

    equity_s = pd.Series(equity, index=times, name="equity")
    buy_hold = pd.Series(
        sim.starting_cash * (close / close[0]), index=times, name="buy_hold"
    )
    running_max = equity_s.cummax()
    drawdown = (equity_s - running_max) / running_max.replace(0.0, np.nan)
    stats = _compute_stats(
        equity_s, buy_hold, trades, sim, in_market_bars / n if n else 0.0
    )

    return BacktestResult(
        indicators=ind,
        equity=equity_s,
        buy_hold=buy_hold,
        drawdown=drawdown,
        trades=trades,
        buy_markers=buy_markers,
        sell_markers=sell_markers,
        stats=stats,
        params=params,
        sim=sim,
    )
