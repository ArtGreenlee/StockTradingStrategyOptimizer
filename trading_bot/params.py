"""Tunable strategy + simulation parameters.

These dataclasses expose **every** knob the analysis, decision and simulation
layers use. The live bot builds :class:`StrategyParams` from ``Settings`` (so
its behavior is unchanged), while the Backtest tab lets the user edit any
field and re-run a simulation.

The defaults here are the single source of truth and intentionally match the
values the engine used when they were hard-coded constants.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # avoid import cycle at runtime
    from .config import Settings


@dataclass
class StrategyParams:
    """Every tunable indicator / scoring / risk parameter."""

    # --- Indicator periods ---
    sma_fast: int = 10
    sma_slow: int = 30
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    atr_period: int = 14
    vol_window: int = 20

    # --- Volatility regime cutoffs (percentile of realized vol) ---
    vol_high_pctile: float = 0.66
    vol_low_pctile: float = 0.33
    # Regime weights scale how hard mean-reversion signals push the score.
    weight_high: float = 1.0
    weight_normal: float = 0.8
    weight_low: float = 0.5

    # --- RSI signal ---
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_weight: float = 2.0

    # --- Bollinger %B signal ---
    bb_low: float = 0.10
    bb_high: float = 0.90
    bb_weight: float = 2.0

    # --- Trend (fast vs slow SMA) ---
    trend_weight: float = 1.0

    # --- Z-score mean reversion (Ornstein-Uhlenbeck style) ---
    # A more principled "stretch" measure than RSI/%B: z = (price - mean) / std.
    zscore_period: int = 20
    zscore_entry: float = 1.5  # |z| beyond this is a stretched, fade-able move
    zscore_weight: float = 2.0

    # --- Trend filter (trade pullbacks WITH the higher-timeframe trend) ---
    # Research (Connors/Alvarez) shows mean-reversion longs work far better
    # above a long-term moving average; fighting strong downtrends bleeds.
    use_trend_filter: bool = True
    trend_filter_period: int = 200

    # --- Fundamentals / snapshot context ---
    fund_gap_threshold: float = 0.02
    fund_gap_weight: float = 0.5
    fund_range_low: float = 0.20
    fund_range_high: float = 0.80
    fund_range_weight: float = 0.5

    # --- Decision thresholds ---
    buy_threshold: float = 3.0
    sell_threshold: float = -3.0

    # --- ATR-based risk management / exits (Wilder) ---
    # The single biggest performance lever: defined-risk exits instead of
    # waiting for an opposite signal. All distances are multiples of ATR.
    # Mean-reversion convention: a WIDE safety stop + a QUICK modest target
    # (high win rate, small wins), exiting near the reverted mean.
    use_atr_exits: bool = True
    stop_loss_atr: float = 3.0      # exit if price <= entry - N*ATR (wide safety)
    take_profit_atr: float = 1.5    # exit if price >= entry + N*ATR (quick)
    trailing_stop_atr: float = 0.0  # 0 = off; trail N*ATR below the peak
    max_hold_bars: int = 60         # 0 = off; time-based exit

    # --- Volatility-based position sizing (ATR risk parity / fractional Kelly) ---
    # Size each trade to risk a fixed % of equity over the stop distance, so
    # position size self-adjusts to volatility instead of a tiny fixed qty.
    use_vol_sizing: bool = True
    risk_per_trade_pct: float = 0.5   # % of equity risked to the stop
    max_position_pct: float = 50.0    # notional cap as % of equity (avoid concentration)

    # --- Risk / sizing ---
    order_qty: int = 1
    # Hard cap on shares. 0 = no share cap (rely on notional + cash). The live
    # bot overrides this from Settings for "small trade" safety.
    max_position_shares: int = 0

    # --- Unusual Options Activity (UOA) flow (live-only enrichment) ---
    # Options flow is a LIVE signal (Alpaca has no easy historical chain), so
    # it only affects live decisions; the backtest ignores it. The decision
    # engine nudges its score toward the net options sentiment.
    use_options_flow: bool = True
    # Scan window: contracts expiring within N days and within +/- this % of
    # spot (near-the-money, near-dated -- where informed/speculative flow sits).
    uoa_expiry_days: int = 45
    uoa_strike_pct: float = 0.15
    # Flag thresholds: a contract is "unusual" if traded volume exceeds this
    # multiple of its open interest AND clears a minimum volume; big premium
    # alone can also flag it.
    uoa_voloi_threshold: float = 2.0
    uoa_min_volume: int = 100
    uoa_min_premium: float = 50_000.0
    # How hard the net options sentiment pushes the decision score (the
    # sentiment is in [-1, 1], so contribution is +/- this weight).
    options_weight: float = 1.5

    # --- News sentiment (works LIVE and in BACKTEST; Alpaca news is historical) ---
    # News is scored with a finance lexicon (deterministic + causal), so it can
    # safely affect both live and backtested decisions. Sentiment is in [-1, 1].
    use_news_sentiment: bool = True
    # How far back published articles still count, and the decay half-life
    # (fresh headlines weigh more). Both in minutes.
    news_lookback_minutes: int = 240
    news_half_life_minutes: float = 60.0
    # Delay between publication and when the strategy may act on a headline.
    # Guards against look-ahead and models realistic reaction latency.
    news_publish_lag_seconds: int = 0
    # How hard net news sentiment pushes the decision score (+/- this weight).
    news_weight: float = 1.5


    @classmethod
    def from_settings(cls, settings: "Settings") -> "StrategyParams":
        """Build defaults, overriding sizing/risk from runtime settings."""
        return cls(
            order_qty=int(settings.order_qty),
            max_position_shares=int(settings.max_position_shares),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class SimConfig:
    """Simulation-only parameters (not used by live trading)."""

    starting_cash: float = 100_000.0
    # Artificial trade latency: delay between a decision and its fill. The
    # fill price is taken from the bar that lands after this delay, so latency
    # visibly costs (or occasionally helps) the strategy.
    latency_seconds: float = 0.0
    # Execution frictions.
    slippage_bps: float = 0.0
    commission_per_trade: float = 0.0
    # How many trailing bars the per-step analysis sees (mirrors the live
    # lookback window so regime/percentile semantics match live trading).
    lookback_bars: int = 390
    # Seconds per bar, derived from the selected timeframe at run time.
    bar_seconds: int = 60

    @property
    def latency_bars(self) -> int:
        if self.bar_seconds <= 0:
            return 0
        return int(self.latency_seconds // self.bar_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
