"""Unusual Options Activity (UOA) detection.

Background / how UOA is detected (researched, standard industry practice):

  * **Volume / Open Interest (Vol/OI)** — the flagship UOA signal. Open
    interest is the number of contracts currently outstanding; daily volume is
    how many traded today. When a contract's *volume exceeds its open
    interest* (Vol/OI > 1, and especially > 2-3), it means far more contracts
    changed hands today than existed at the open -- i.e. large NEW positioning,
    not routine closing. This is the classic "unusual" flag.
  * **Put/Call ratio** — total put volume / total call volume across the chain.
    A low ratio (call-heavy) skews bullish; a high ratio (put-heavy) skews
    bearish. Extreme readings are themselves "unusual" and sometimes
    contrarian.
  * **Aggressor side (sweep direction)** — comparing the last trade price to
    the bid/ask: trades at/above the ask are buyer-initiated (urgent demand);
    trades at/below the bid are seller-initiated. Heavy at-ask call buying is
    bullish; heavy at-ask put buying is bearish.
  * **Net premium (dollar volume)** — volume x price x 100. Big premium spent
    on calls vs puts shows where real money is committed.
  * **Implied volatility** — elevated IV signals demand/expectation of a move.
  * **Short-dated, out-of-the-money concentration** — speculative/informed
    bets cluster in near-term OTM contracts.

This module is pure (no network). The data layer hands it a tidy per-contract
DataFrame; here we compute the metrics, flag unusual contracts, and roll them
up into a single directional sentiment score the decision engine can use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# Required columns the data layer must provide (one row per option contract):
#   symbol, type ("call"/"put"), strike, expiration (datetime),
#   open_interest, volume, last_price, bid, ask, implied_volatility, delta
REQUIRED_COLUMNS = [
    "symbol", "type", "strike", "expiration", "open_interest", "volume",
    "last_price", "bid", "ask", "implied_volatility", "delta",
]


@dataclass
class FlaggedContract:
    """A single contract flagged as showing unusual activity."""

    symbol: str
    option_type: str  # "call" | "put"
    strike: float
    expiration: str
    volume: float
    open_interest: float
    vol_oi: float
    last_price: float
    implied_volatility: float
    aggressor: str  # "buy" | "sell" | "mid"
    premium: float  # dollar volume = volume * price * 100
    reasons: List[str] = field(default_factory=list)

    @property
    def bullish(self) -> bool:
        """Directional read for the underlying from this contract's flow."""
        if self.aggressor == "buy":
            return self.option_type == "call"
        if self.aggressor == "sell":
            # Sold-to-open puts are mildly bullish; sold calls mildly bearish.
            return self.option_type == "put"
        return self.option_type == "call"


@dataclass
class OptionsActivity:
    """Chain-level UOA summary + flagged contracts + directional sentiment."""

    available: bool
    as_of: Optional[object] = None
    underlying_price: float = float("nan")
    contracts_scanned: int = 0
    call_volume: float = 0.0
    put_volume: float = 0.0
    call_oi: float = 0.0
    put_oi: float = 0.0
    put_call_volume_ratio: float = float("nan")
    put_call_oi_ratio: float = float("nan")
    call_premium: float = 0.0  # buyer-initiated call $ volume
    put_premium: float = 0.0   # buyer-initiated put $ volume
    avg_iv: float = float("nan")
    max_vol_oi: float = float("nan")
    flagged: List[FlaggedContract] = field(default_factory=list)
    # Net sentiment in roughly [-1, 1]: +bullish call flow, -bearish put flow.
    sentiment: float = 0.0
    note: str = ""

    @property
    def unusual_count(self) -> int:
        return len(self.flagged)


def _aggressor(last: float, bid: float, ask: float) -> str:
    """Classify a trade as buyer-/seller-initiated from its quote context."""
    if not (np.isfinite(last) and np.isfinite(bid) and np.isfinite(ask)):
        return "mid"
    if ask <= bid:  # crossed/locked or missing -> inconclusive
        return "mid"
    mid = 0.5 * (bid + ask)
    span = ask - bid
    # Within the top/bottom third of the spread => at ask / at bid.
    if last >= ask - 0.34 * span:
        return "buy"
    if last <= bid + 0.34 * span:
        return "sell"
    return "mid"


def empty_activity(note: str) -> OptionsActivity:
    return OptionsActivity(available=False, note=note)


def analyze_options_activity(
    df: Optional[pd.DataFrame],
    underlying_price: float,
    params,
    as_of: Optional[object] = None,
) -> OptionsActivity:
    """Compute UOA metrics + flags + sentiment from a per-contract frame.

    Args:
        df: per-contract DataFrame with :data:`REQUIRED_COLUMNS`.
        underlying_price: current price of the underlying (for moneyness).
        params: :class:`StrategyParams` (thresholds + weights).
        as_of: timestamp of the data, for display.
    """
    if df is None or len(df) == 0:
        return empty_activity("No options data available.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return empty_activity(f"Options data missing columns: {missing}")

    d = df.copy()
    for col in ["strike", "open_interest", "volume", "last_price", "bid",
                "ask", "implied_volatility", "delta"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["type"] = d["type"].astype(str).str.lower()
    d = d[d["volume"].fillna(0) > 0]
    if d.empty:
        return empty_activity("No traded option contracts today.")

    d["vol_oi"] = d["volume"] / d["open_interest"].replace(0.0, np.nan)
    d["premium"] = d["volume"] * d["last_price"].fillna(0.0) * 100.0
    d["aggressor"] = [
        _aggressor(lp, b, a)
        for lp, b, a in zip(d["last_price"], d["bid"], d["ask"])
    ]

    calls = d[d["type"] == "call"]
    puts = d[d["type"] == "put"]
    call_vol = float(calls["volume"].sum())
    put_vol = float(puts["volume"].sum())
    call_oi = float(calls["open_interest"].sum())
    put_oi = float(puts["open_interest"].sum())

    pcr_vol = (put_vol / call_vol) if call_vol > 0 else float("nan")
    pcr_oi = (put_oi / call_oi) if call_oi > 0 else float("nan")

    # Buyer-initiated premium by side (the "smart money committed" view).
    call_prem = float(calls.loc[calls["aggressor"] == "buy", "premium"].sum())
    put_prem = float(puts.loc[puts["aggressor"] == "buy", "premium"].sum())

    # Flag unusual contracts: Vol/OI over threshold AND meaningful volume.
    flagged: List[FlaggedContract] = []
    for _, row in d.iterrows():
        vol = float(row["volume"])
        oi = float(row["open_interest"]) if np.isfinite(row["open_interest"]) else 0.0
        voloi = float(row["vol_oi"]) if np.isfinite(row["vol_oi"]) else (
            np.inf if vol > 0 and oi == 0 else 0.0
        )
        reasons: List[str] = []
        if vol >= params.uoa_min_volume and voloi >= params.uoa_voloi_threshold:
            reasons.append(f"Vol/OI {voloi:.1f}x (vol {vol:.0f} > OI {oi:.0f})")
        if vol >= params.uoa_min_volume and oi == 0:
            reasons.append("new strike (no prior OI)")
        prem = float(row["premium"])
        if prem >= params.uoa_min_premium:
            reasons.append(f"${prem:,.0f} premium")
        if not reasons:
            continue
        # Days to expiration for short-dated emphasis.
        flagged.append(
            FlaggedContract(
                symbol=str(row["symbol"]),
                option_type=str(row["type"]),
                strike=float(row["strike"]),
                expiration=str(row["expiration"])[:10],
                volume=vol,
                open_interest=oi,
                vol_oi=voloi,
                last_price=float(row["last_price"]) if np.isfinite(row["last_price"]) else float("nan"),
                implied_volatility=float(row["implied_volatility"]) if np.isfinite(row["implied_volatility"]) else float("nan"),
                aggressor=str(row["aggressor"]),
                premium=prem,
                reasons=reasons,
            )
        )

    # Sort flagged by premium (most capital first) and cap for display.
    flagged.sort(key=lambda f: f.premium, reverse=True)

    # ---- Directional sentiment ------------------------------------------
    # Combine two evidence sources, each squashed to roughly [-1, 1]:
    #  1) Buyer-initiated premium imbalance (calls vs puts).
    #  2) Flagged-contract bullish/bearish premium imbalance (the UOA itself).
    total_prem = call_prem + put_prem
    prem_imbalance = (call_prem - put_prem) / total_prem if total_prem > 0 else 0.0

    flag_bull = sum(f.premium for f in flagged if f.bullish)
    flag_bear = sum(f.premium for f in flagged if not f.bullish)
    flag_total = flag_bull + flag_bear
    flag_imbalance = (flag_bull - flag_bear) / flag_total if flag_total > 0 else 0.0

    sentiment = float(np.clip(0.5 * prem_imbalance + 0.5 * flag_imbalance, -1.0, 1.0))

    avg_iv = float(d["implied_volatility"].replace([np.inf, -np.inf], np.nan).dropna().mean()) \
        if d["implied_volatility"].notna().any() else float("nan")
    max_voloi = float(d["vol_oi"].replace([np.inf], np.nan).max()) \
        if d["vol_oi"].notna().any() else float("nan")

    return OptionsActivity(
        available=True,
        as_of=as_of,
        underlying_price=float(underlying_price) if np.isfinite(underlying_price) else float("nan"),
        contracts_scanned=int(len(d)),
        call_volume=call_vol,
        put_volume=put_vol,
        call_oi=call_oi,
        put_oi=put_oi,
        put_call_volume_ratio=pcr_vol,
        put_call_oi_ratio=pcr_oi,
        call_premium=call_prem,
        put_premium=put_prem,
        avg_iv=avg_iv,
        max_vol_oi=max_voloi,
        flagged=flagged,
        sentiment=sentiment,
        note="" if flagged else "No contracts crossed the unusual thresholds.",
    )
