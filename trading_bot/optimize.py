"""Hyperparameter search for the strategy, optimizing for profit.

Runs many simulations over historical data, varying selected
:class:`StrategyParams` fields, and keeps the configuration with the highest
total return. Two methods are provided:

  * ``random``  -- sample every enabled parameter uniformly each trial.
  * ``guided``  -- stochastic hill-climb: perturb the current best on a random
                   subset of dimensions, accept improvements, occasionally
                   restart. Produces a converging parameter trajectory.

The search runs on a background thread and reports each trial through a
callback so the GUI can plot, live, how the best-so-far parameters evolve.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from .backtest import run_backtest
from .params import SimConfig, StrategyParams


@dataclass
class SearchSpec:
    """One searchable parameter and its bounds."""

    name: str
    low: float
    high: float
    is_int: bool
    enabled: bool = True

    def clip(self, value: float) -> float:
        value = min(self.high, max(self.low, value))
        return int(round(value)) if self.is_int else float(value)

    def sample(self, rng: np.random.Generator) -> float:
        return self.clip(rng.uniform(self.low, self.high))

    def normalize(self, value: float) -> float:
        span = self.high - self.low
        return 0.0 if span == 0 else (float(value) - self.low) / span


def default_search_space() -> List[SearchSpec]:
    """A curated search space. Core signal + risk knobs default ON; structural
    indicator *periods* default OFF (they reshape indicators and inflate the
    search volume / overfitting risk, so enable them deliberately).

    Bounds are intentionally generous: several were previously *clipping* good
    regions (e.g. the default ``stop_loss_atr``/``trend_filter_period`` sat near
    their old caps). Wider bounds enlarge the search, so raise trial count and
    lean on the out-of-sample % check when exploring the full space.
    """
    return [
        # --- Decision thresholds (max bullish score ~16 when weights maxed) ---
        SearchSpec("buy_threshold", 0.5, 10.0, False, True),
        SearchSpec("sell_threshold", -10.0, -0.5, False, True),
        # --- Core signals + weights (weights widened 4 -> 6 so one signal can
        #     dominate, which pairs with the higher decision thresholds) ---
        SearchSpec("zscore_entry", 0.5, 4.0, False, True),
        SearchSpec("zscore_weight", 0.0, 6.0, False, True),
        SearchSpec("rsi_oversold", 10, 45, True, True),
        SearchSpec("rsi_overbought", 55, 90, True, True),
        SearchSpec("rsi_weight", 0.0, 6.0, False, True),
        SearchSpec("bb_low", 0.0, 0.4, False, False),
        SearchSpec("bb_high", 0.6, 1.0, False, False),
        SearchSpec("bb_weight", 0.0, 6.0, False, True),
        SearchSpec("trend_weight", 0.0, 5.0, False, False),
        # --- Risk management / exits -- the biggest performance levers ---
        SearchSpec("stop_loss_atr", 0.5, 8.0, False, True),     # wider safety stops
        SearchSpec("take_profit_atr", 0.5, 6.0, False, True),
        SearchSpec("trailing_stop_atr", 0.0, 5.0, False, True),
        SearchSpec("max_hold_bars", 0, 1560, True, True),       # up to ~4 sessions
        SearchSpec("risk_per_trade_pct", 0.1, 5.0, False, True),
        # --- Structural indicator periods (default OFF) ---
        SearchSpec("trend_filter_period", 20, 400, True, False),
        SearchSpec("sma_fast", 3, 30, True, False),
        SearchSpec("sma_slow", 20, 250, True, False),
        SearchSpec("bb_period", 5, 60, True, False),
        SearchSpec("bb_std", 1.0, 3.5, False, False),
        SearchSpec("rsi_period", 5, 45, True, False),
        SearchSpec("zscore_period", 5, 90, True, False),
        SearchSpec("vol_window", 5, 90, True, False),
        SearchSpec("max_position_pct", 10, 100, False, False),
    ]


@dataclass
class TrialResult:
    """Outcome of a single evaluated configuration."""

    iteration: int
    objective: float  # candidate's total return (profit)
    best_objective: float  # best total return seen so far
    best_params: StrategyParams  # best-so-far configuration
    final_equity: float
    num_trades: int
    accepted: bool  # did this trial become the new best?
    trials_since_improvement: int = 0  # for early-stop patience display


@dataclass
class SearchResult:
    """Final result of a completed (or stopped) search."""

    best_params: StrategyParams
    best_objective: float
    best_stats: dict
    history: List[TrialResult]
    specs: List[SearchSpec]
    completed: bool
    objective_name: str = "return"
    # Out-of-sample (held-out tail) check of the best configuration.
    oos_objective: Optional[float] = None
    oos_stats: Optional[dict] = None
    # Why the search ended: "all-trials" | "patience" | "target" | "stopped".
    stop_reason: str = "all-trials"


# Supported optimization objectives, all "higher is better".
#   return       -- total return (raw profit)
#   sharpe       -- annualized Sharpe ratio
#   calmar       -- return / abs(max drawdown)
#   sharpe_calmar-- balanced blend of (squashed) Sharpe + Calmar; requires BOTH
#                   to be good, so it resists configs that game one metric.
OBJECTIVES = ("return", "sharpe", "calmar", "sharpe_calmar")


def _calmar(stats: dict) -> float:
    dd = abs(stats.get("max_drawdown", 0.0))
    ret = stats.get("total_return", 0.0)
    # No drawdown with positive return is ideal; scale to a large number.
    return ret / dd if dd > 1e-9 else (ret * 100.0 if ret > 0 else 0.0)


def _squash(value: float, scale: float) -> float:
    """Map an unbounded 'higher is better' metric to (-1, 1) via tanh.

    ``scale`` is the value that maps to tanh(1) ~= 0.76 (i.e. 'very good').
    """
    if value is None or not np.isfinite(value):
        return -1.0
    return float(np.tanh(value / scale))


def _activity_penalty(stats: dict, min_trades: int, min_exposure: float) -> float:
    """Return a multiplier in [0, 1] that ramps down for under-active configs.

    This is the guard against the classic Sharpe-optimization degeneracy where
    the optimizer "wins" by trading almost never (tiny exposure, a handful of
    lucky trades, near-zero variance -> astronomically high Sharpe). When a
    config falls short of the minimum trade count or exposure, its objective is
    scaled down proportionally; meeting both leaves it untouched.
    """
    factor = 1.0
    if min_trades and min_trades > 0:
        n = float(stats.get("num_trades", 0) or 0)
        factor *= min(1.0, n / float(min_trades))
    if min_exposure and min_exposure > 0:
        exp = float(stats.get("exposure", 0.0) or 0.0)
        factor *= min(1.0, exp / float(min_exposure))
    return max(0.0, factor)


def objective_value(
    stats: dict,
    objective: str,
    min_trades: int = 0,
    min_exposure: float = 0.0,
    calmar_weight: float = 0.5,
) -> float:
    """Map backtest stats to a scalar objective (higher = better).

    Args:
        stats: backtest result stats dict.
        objective: one of :data:`OBJECTIVES`.
        min_trades: if > 0, configs with fewer trades are penalized (a soft
            ramp, not a hard cutoff) -- this is the "penalized" mode.
        min_exposure: if > 0 (fraction 0..1), configs in the market less than
            this are penalized similarly.
        calmar_weight: for ``sharpe_calmar``, the share (0..1) given to Calmar
            (the rest goes to Sharpe).
    """
    if not stats:
        return float("-inf")

    if objective == "sharpe":
        base = stats.get("sharpe", 0.0)
        base = float(base) if base is not None and np.isfinite(base) else float("-inf")
    elif objective == "calmar":
        base = _calmar(stats)
    elif objective == "sharpe_calmar":
        s_n = _squash(stats.get("sharpe", 0.0), 3.0)   # Sharpe ~3 -> "very good"
        c_n = _squash(_calmar(stats), 3.0)             # Calmar ~3 -> "very good"
        if s_n > 0 and c_n > 0:
            # Weighted geometric mean: needs BOTH metrics good (imbalance hurts).
            w = min(1.0, max(0.0, calmar_weight))
            base = (s_n ** (1.0 - w)) * (c_n ** w)
        else:
            # If either is non-positive, the weaker one drags the score down.
            base = min(s_n, c_n)
    else:  # "return"
        base = stats.get("total_return", 0.0)
        base = float(base) if base is not None and np.isfinite(base) else float("-inf")

    if base == float("-inf"):
        return base

    # Activity guard / penalty: only ever REMOVES reward from a positive score,
    # so it can't make a bad (negative) config look good.
    if base > 0 and (min_trades > 0 or min_exposure > 0):
        base *= _activity_penalty(stats, min_trades, min_exposure)
    return float(base)


def _params_value(params: StrategyParams, name: str) -> float:
    return float(getattr(params, name))


def _apply(base: StrategyParams, values: Dict[str, float]) -> StrategyParams:
    return replace(base, **values)


def _evaluate(
    df, params: StrategyParams, sim: SimConfig, objective: str,
    min_trades: int = 0, min_exposure: float = 0.0, calmar_weight: float = 0.5,
    news_df=None,
) -> tuple:
    """Return (objective, stats) for a parameter set.

    Degenerate configs that error out (or produce non-finite results) are
    penalized so the optimizer avoids them.
    """
    try:
        result = run_backtest(df, params, sim, news_df=news_df)
    except Exception:
        return float("-inf"), {}
    obj = objective_value(
        result.stats, objective, min_trades, min_exposure, calmar_weight
    )
    return obj, result.stats


def run_search(
    df,
    base_params: StrategyParams,
    sim: SimConfig,
    specs: List[SearchSpec],
    n_trials: int,
    method: str = "guided",
    seed: int = 42,
    objective: str = "return",
    oos_fraction: float = 0.0,
    patience: int = 0,
    min_improvement: float = 0.0,
    target_objective: Optional[float] = None,
    min_trades: int = 0,
    min_exposure: float = 0.0,
    calmar_weight: float = 0.5,
    news_df=None,
    progress: Optional[Callable[[TrialResult], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> SearchResult:
    """Search ``specs`` for the best configuration under ``objective``.

    Args:
        df: historical OHLCV bars.
        base_params: starting point; non-searched fields are held fixed.
        sim: simulation config (cash, latency, slippage, commission).
        specs: candidate parameters (only ``enabled`` ones vary).
        n_trials: number of configurations to evaluate.
        method: ``"guided"`` (hill-climb) or ``"random"``.
        seed: RNG seed for reproducibility.
        objective: ``"return"``, ``"sharpe"``, ``"calmar"`` or
            ``"sharpe_calmar"`` (higher is better).
        oos_fraction: fraction of the tail held out for out-of-sample scoring
            (0 disables). Optimization runs only on the in-sample portion; the
            best config is then scored on the unseen tail to expose overfitting.
        min_trades: penalize configs with fewer than this many trades (guards
            against the "trade almost never" Sharpe degeneracy).
        min_exposure: penalize configs in the market less than this fraction
            (0..1) of the time.
        calmar_weight: for ``sharpe_calmar``, the share (0..1) given to Calmar.
        patience: early stop after this many consecutive trials without an
            improvement greater than ``min_improvement`` (0 disables).
        min_improvement: minimum gain in the objective that counts as an
            improvement (filters out noise so patience isn't reset by tiny
            gains).
        target_objective: early stop once the best objective reaches this value
            (``None`` disables).
        progress: called with each :class:`TrialResult`.
        stop_event: if set, the search stops early and returns what it found.
    """
    active = [s for s in specs if s.enabled]
    if not active:
        raise ValueError("Select at least one parameter to search.")
    if df is None or len(df) == 0:
        raise ValueError("No data to search over. Download history first.")

    # Train / test split (out-of-sample tail).
    train_df = df
    test_df = None
    train_news = news_df
    test_news = news_df
    if oos_fraction and 0.0 < oos_fraction < 0.9 and len(df) >= 100:
        cut = int(len(df) * (1.0 - oos_fraction))
        train_df, test_df = df.iloc[:cut], df.iloc[cut:]
        # Split news by the same time boundary so OOS news is causal-correct.
        if news_df is not None and len(news_df) > 0 and "timestamp" in news_df.columns:
            boundary = df.index[cut]
            ts = news_df["timestamp"]
            train_news = news_df[ts < boundary]
            test_news = news_df  # test window sees all prior news for warm-up

    rng = np.random.default_rng(seed)
    history: List[TrialResult] = []

    # Seed the search with the base configuration.
    center = {s.name: s.clip(_params_value(base_params, s.name)) for s in active}
    best_values = dict(center)
    best_params = _apply(base_params, best_values)
    best_obj, best_stats = _evaluate(
        train_df, best_params, sim, objective,
        min_trades, min_exposure, calmar_weight, train_news,
    )

    # Per-dimension Gaussian step for the guided method (15% of each range).
    steps = {s.name: 0.15 * (s.high - s.low) for s in active}

    # Early-stopping bookkeeping.
    trials_since_improve = 0
    stop_reason = "all-trials"
    patience = max(0, int(patience))
    min_improvement = max(0.0, float(min_improvement))

    # The seed configuration may already meet the target.
    if target_objective is not None and best_obj >= target_objective:
        stop_reason = "target"

    if stop_reason != "target":
        for i in range(1, max(1, int(n_trials)) + 1):
            if stop_event is not None and stop_event.is_set():
                stop_reason = "stopped"
                break

            if method == "random":
                cand = {s.name: s.sample(rng) for s in active}
            else:  # guided hill-climb
                if rng.random() < 0.1:
                    # Occasional random restart to escape local optima.
                    cand = {s.name: s.sample(rng) for s in active}
                else:
                    cand = dict(center)
                    k = rng.integers(1, len(active) + 1)
                    for s in rng.choice(active, size=int(k), replace=False):
                        step = rng.normal(0.0, steps[s.name])
                        cand[s.name] = s.clip(center[s.name] + step)

            cand_params = _apply(base_params, cand)
            obj, stats = _evaluate(
                train_df, cand_params, sim, objective,
                min_trades, min_exposure, calmar_weight, train_news,
            )

            # An "improvement" must beat the best by more than min_improvement
            # so noise doesn't keep resetting the patience counter.
            improved = obj > best_obj + min_improvement
            accepted = obj > best_obj
            if accepted:
                best_obj, best_values = obj, dict(cand)
                best_params = cand_params
                best_stats = stats
            if method != "random" and obj >= best_obj:
                # Move the hill-climb center toward improving candidates.
                center = dict(cand)

            trials_since_improve = 0 if improved else trials_since_improve + 1

            trial = TrialResult(
                iteration=i,
                objective=obj,
                best_objective=best_obj,
                best_params=best_params,
                final_equity=float(best_stats.get("final_equity", sim.starting_cash)),
                num_trades=int(best_stats.get("num_trades", 0)),
                accepted=accepted,
                trials_since_improvement=trials_since_improve,
            )
            history.append(trial)
            if progress is not None:
                progress(trial)

            # Early-stop checks (after reporting the trial).
            if target_objective is not None and best_obj >= target_objective:
                stop_reason = "target"
                break
            if patience > 0 and trials_since_improve >= patience:
                stop_reason = "patience"
                break

    # Out-of-sample scoring of the best configuration.
    oos_obj = None
    oos_stats = None
    if test_df is not None and len(test_df) >= 20:
        oos_obj, oos_stats = _evaluate(
            test_df, best_params, sim, objective,
            min_trades, min_exposure, calmar_weight, test_news,
        )

    # "completed" means the search ran its full course (incl. early stops that
    # are legitimate convergence outcomes); only a user Stop is "incomplete".
    completed = stop_reason != "stopped"
    return SearchResult(
        best_params=best_params,
        best_objective=best_obj,
        best_stats=best_stats,
        history=history,
        specs=active,
        completed=completed,
        objective_name=objective,
        oos_objective=oos_obj,
        oos_stats=oos_stats,
        stop_reason=stop_reason,
    )


def search_process_entry(out_queue, stop_event, kwargs: dict) -> None:
    """Run a full search in a child process, streaming results to ``out_queue``.

    This is the top-level (picklable) target for ``multiprocessing.Process``.
    Running the CPU-bound search in its own process means it cannot contend for
    the GIL with the GUI's main thread, so the UI stays fully responsive no
    matter how heavy the backtests are.

    Messages put on ``out_queue``:
        ("trial", TrialResult)         -- one per evaluated configuration
        ("search_done", SearchResult)  -- final result (incl. early stops)
        ("search_error", str)          -- on failure
    """
    try:
        def progress(trial: TrialResult) -> None:
            out_queue.put(("trial", trial))

        result = run_search(
            progress=progress, stop_event=stop_event, **kwargs
        )
        out_queue.put(("search_done", result))
    except Exception as exc:  # report instead of dying silently
        out_queue.put(
            ("search_error", f"Search failed: {type(exc).__name__}: {exc}")
        )

