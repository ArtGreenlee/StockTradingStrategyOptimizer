"""Position sizing + ATR-based exit logic (shared by backtest and live).

Centralizing this here keeps the simulated and live behavior identical:

  * ``target_entry_size`` — volatility-scaled sizing. Risk a fixed fraction of
    equity over the ATR stop distance (Wilder ATR position sizing / fractional
    Kelly), bounded by a notional cap, optional hard share cap, and cash.
  * ``exit_reason`` — defined-risk exits: ATR stop-loss, ATR take-profit,
    ATR trailing stop, and a time-based stop. This is the biggest single
    performance lever versus only ever closing on an opposite signal.

This bot is long-only (it buys, then sells to flat), so the exit logic is
written for long positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .params import StrategyParams


@dataclass
class PositionState:
    """Tracks an open long position for exit evaluation."""

    shares: int = 0
    avg_entry: float = 0.0
    peak_price: float = 0.0  # highest price seen since entry (trailing stop)
    bars_held: int = 0

    def reset(self) -> None:
        self.shares = 0
        self.avg_entry = 0.0
        self.peak_price = 0.0
        self.bars_held = 0

    def open(self, shares: int, price: float) -> None:
        self.shares = int(shares)
        self.avg_entry = float(price)
        self.peak_price = float(price)
        self.bars_held = 0

    def step(self, price: float) -> None:
        """Advance one bar: update peak + holding period."""
        if self.shares > 0:
            self.peak_price = max(self.peak_price, float(price))
            self.bars_held += 1


def _atr_ok(atr: float) -> bool:
    return atr is not None and np.isfinite(atr) and atr > 0


def target_entry_size(
    equity: float,
    cash: float,
    price: float,
    atr: float,
    params: StrategyParams,
) -> int:
    """Return the number of shares to buy for a fresh long entry.

    Volatility sizing: shares ≈ (equity * risk%) / (stop_atr * ATR). Bounded by
    a notional cap (% of equity), an optional hard share cap, and available
    cash (no leverage). Falls back to the notional cap when ATR is unusable.
    """
    if price <= 0:
        return 0

    notional_cap = int((equity * params.max_position_pct / 100.0) // price)

    if bool(params.use_vol_sizing) and _atr_ok(atr):
        risk_dollars = equity * params.risk_per_trade_pct / 100.0
        stop_dist = max(params.stop_loss_atr, 0.5) * atr
        size = int(risk_dollars // stop_dist) if stop_dist > 0 else 0
    else:
        size = notional_cap

    size = min(size, notional_cap)
    if params.max_position_shares and params.max_position_shares > 0:
        size = min(size, int(params.max_position_shares))
    size = min(size, int(cash // price))  # no leverage
    return max(size, 0)


def exit_reason(
    state: PositionState, price: float, atr: float, params: StrategyParams
) -> Optional[str]:
    """Return why an open long should be closed now, or None to hold.

    Checks ATR stop-loss, ATR take-profit, ATR trailing stop, and a time stop,
    in that order.
    """
    if state.shares <= 0 or not bool(params.use_atr_exits):
        return None

    entry = state.avg_entry
    if _atr_ok(atr):
        if params.stop_loss_atr > 0 and price <= entry - params.stop_loss_atr * atr:
            return "stop-loss"
        if params.take_profit_atr > 0 and price >= entry + params.take_profit_atr * atr:
            return "take-profit"
        if (
            params.trailing_stop_atr > 0
            and state.peak_price > 0
            and price <= state.peak_price - params.trailing_stop_atr * atr
        ):
            return "trailing-stop"

    if params.max_hold_bars > 0 and state.bars_held >= params.max_hold_bars:
        return "time-stop"
    return None
