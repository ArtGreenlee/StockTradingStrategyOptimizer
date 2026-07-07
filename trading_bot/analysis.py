"""Technical + volatility analysis.

Turns a frame of 1-minute bars into the indicator series used both for
plotting and for the decision engine. The focus is *volatility* and
*mean reversion*, which drive the small trades this bot is designed for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .params import StrategyParams

# Legacy module-level defaults (kept for reference; the authoritative
# defaults now live in :class:`StrategyParams`).
SMA_FAST = 10
SMA_SLOW = 30
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_WINDOW = 20  # rolling realized-volatility window


@dataclass
class Analysis:
    """Computed indicator series + latest scalar values.

    ``frame`` carries the full set of series (for charts). The scalar fields
    are the most recent non-NaN values (for the decision engine + panels).
    """

    frame: pd.DataFrame
    last_price: float
    sma_fast: float
    sma_slow: float
    bb_upper: float
    bb_lower: float
    bb_mid: float
    percent_b: float
    bb_width: float
    rsi: float
    atr: float
    realized_vol: float
    vol_regime: str  # "LOW" | "NORMAL" | "HIGH"
    vol_percentile: float  # 0..1, current realized vol vs its own history
    zscore: float = float("nan")  # (price - rolling mean) / rolling std
    trend_sma: float = float("nan")  # long-term SMA for the trend filter

    @property
    def has_data(self) -> bool:
        return not self.frame.empty and np.isfinite(self.last_price)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When there are no losses RSI is 100; when no gains it's 0.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, rsi.where(avg_loss == 0.0, 0.0))
    return rsi


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _last(series: pd.Series, default: float = float("nan")) -> float:
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else default


def _empty_analysis() -> "Analysis":
    return Analysis(
        frame=pd.DataFrame(),
        last_price=float("nan"),
        sma_fast=float("nan"),
        sma_slow=float("nan"),
        bb_upper=float("nan"),
        bb_lower=float("nan"),
        bb_mid=float("nan"),
        percent_b=float("nan"),
        bb_width=float("nan"),
        rsi=float("nan"),
        atr=float("nan"),
        realized_vol=float("nan"),
        vol_regime="UNKNOWN",
        vol_percentile=float("nan"),
        zscore=float("nan"),
        trend_sma=float("nan"),
    )


def compute_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Return ``df`` plus all indicator columns (no regime/percentile).

    Every column here is causal (rolling / EWM), so it is safe to compute
    once over a full history and slice per-bar in a backtest without
    introducing look-ahead bias.
    """
    out = df.copy()
    close = out["close"]

    out["sma_fast"] = close.rolling(params.sma_fast, min_periods=1).mean()
    out["sma_slow"] = close.rolling(params.sma_slow, min_periods=1).mean()

    mid = close.rolling(params.bb_period, min_periods=params.bb_period).mean()
    std = close.rolling(params.bb_period, min_periods=params.bb_period).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + params.bb_std * std
    out["bb_lower"] = mid - params.bb_std * std
    band_range = (out["bb_upper"] - out["bb_lower"]).replace(0.0, np.nan)
    out["percent_b"] = (close - out["bb_lower"]) / band_range
    out["bb_width"] = band_range / mid.replace(0.0, np.nan)

    out["rsi"] = _rsi(close, params.rsi_period)
    out["atr"] = _atr(out, params.atr_period)

    # Z-score: how many rolling std-devs price sits from its rolling mean.
    z_mean = close.rolling(params.zscore_period, min_periods=params.zscore_period).mean()
    z_std = close.rolling(params.zscore_period, min_periods=params.zscore_period).std(ddof=0)
    out["zscore"] = (close - z_mean) / z_std.replace(0.0, np.nan)

    # Long-term trend SMA for the trend filter (min_periods=1 so it is defined
    # early; it simply tracks the running average until the full window fills).
    out["trend_sma"] = close.rolling(params.trend_filter_period, min_periods=1).mean()

    # Realized volatility = rolling std of log returns, annualized to a
    # comparable scale (approx US trading minutes per year).
    log_ret = np.log(close / close.shift(1))
    out["realized_vol"] = log_ret.rolling(
        params.vol_window, min_periods=params.vol_window
    ).std(ddof=0) * np.sqrt(252 * 390)

    return out


def regime_from_percentile(vol_percentile: float, params: StrategyParams) -> str:
    """Classify a realized-volatility percentile into a regime label."""
    if vol_percentile is None or np.isnan(vol_percentile):
        return "UNKNOWN"
    if vol_percentile >= params.vol_high_pctile:
        return "HIGH"
    if vol_percentile <= params.vol_low_pctile:
        return "LOW"
    return "NORMAL"


def analyze(df: pd.DataFrame, params: Optional[StrategyParams] = None) -> Analysis:
    """Compute all indicator series + latest values from minute bars."""
    if params is None:
        params = StrategyParams()
    if df is None or df.empty:
        return _empty_analysis()

    out = compute_indicators(df, params)

    realized_vol = _last(out["realized_vol"])
    vol_series = out["realized_vol"].dropna()
    if len(vol_series) >= 5 and np.isfinite(realized_vol):
        vol_percentile = float((vol_series <= realized_vol).mean())
    else:
        vol_percentile = float("nan")

    return Analysis(
        frame=out,
        last_price=_last(out["close"]),
        sma_fast=_last(out["sma_fast"]),
        sma_slow=_last(out["sma_slow"]),
        bb_upper=_last(out["bb_upper"]),
        bb_lower=_last(out["bb_lower"]),
        bb_mid=_last(out["bb_mid"]),
        percent_b=_last(out["percent_b"]),
        bb_width=_last(out["bb_width"]),
        rsi=_last(out["rsi"]),
        atr=_last(out["atr"]),
        realized_vol=realized_vol,
        vol_regime=regime_from_percentile(vol_percentile, params),
        vol_percentile=vol_percentile,
        zscore=_last(out["zscore"]),
        trend_sma=_last(out["trend_sma"]),
    )
