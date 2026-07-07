"""Persistent store for hyperparameter-search results.

Every completed (or stopped) search is written to a JSON file under
``search_history/`` at the project root, so you can revisit and reload any past
configuration. Each record captures the winning :class:`StrategyParams` plus
enough context (ticker, objective, in-sample + out-of-sample metrics, stop
reason, trial count, timestamp) to judge it later.

The format is plain JSON -- portable, diff-able, and easy to inspect by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .params import StrategyParams

# search_history/ lives at the project root (one level above this package).
HISTORY_DIR = Path(__file__).resolve().parent.parent / "search_history"


@dataclass
class SearchRecord:
    """Metadata + winning parameters for one stored search."""

    path: Path
    timestamp: str
    ticker: str
    objective: str
    best_objective: float
    best_return: float
    oos_objective: Optional[float]
    oos_return: Optional[float]
    stop_reason: str
    trials: int
    timeframe: str
    params: Dict[str, Any]
    # Per-trial objective trajectory: list of {iteration, objective,
    # best_objective, num_trades}.
    objective_history: List[Dict[str, Any]] = None  # type: ignore[assignment]
    # Best-parameter snapshots at each improvement: {iteration,
    # best_objective, params}.
    param_evolution: List[Dict[str, Any]] = None  # type: ignore[assignment]

    @property
    def label(self) -> str:
        """One-line human summary for list displays."""
        when = self.timestamp.replace("T", " ")[:19]
        oos = (
            f", OOS {_fmt(self.objective, self.oos_objective)}"
            if self.oos_objective is not None
            else ""
        )
        return (
            f"{when} | {self.ticker} {self.timeframe} | {self.objective} "
            f"{_fmt(self.objective, self.best_objective)}{oos} | "
            f"ret {self.best_return * 100:+.1f}% | {self.trials} trials"
        )


def _fmt(objective: str, value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if objective == "return":
        return f"{value * 100:+.1f}%"
    return f"{value:.2f}"


def _coerce(value: Any) -> Any:
    """Make numpy / pandas scalars JSON-serializable."""
    try:
        import numpy as np

        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
    except Exception:
        pass
    return value


def _clean_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not stats:
        return {}
    return {k: _coerce(v) for k, v in stats.items()}


def record_search(
    result,
    ticker: str,
    timeframe: str = "",
    directory: Optional[Path] = None,
) -> Path:
    """Persist a :class:`SearchResult` to a timestamped JSON file.

    Returns the path written. Never raises on serialization issues that would
    lose the params -- it coerces values defensively.
    """
    directory = directory or HISTORY_DIR
    directory.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    safe_ticker = "".join(c for c in (ticker or "NA") if c.isalnum()) or "NA"
    fname = f"{now.strftime('%Y%m%dT%H%M%S')}_{safe_ticker}.json"
    path = directory / fname

    # Per-trial objective trajectory ("history of sharpe scores") + a snapshot
    # of the best parameters each time they improved ("best parameters per
    # run"). The trajectory is compact; param snapshots are only stored when
    # the best actually changed, so files stay small even for long searches.
    trials = list(getattr(result, "history", []) or [])
    objective_history = [
        {
            "iteration": int(getattr(t, "iteration", i + 1)),
            "objective": _coerce(getattr(t, "objective", None)),
            "best_objective": _coerce(getattr(t, "best_objective", None)),
            "num_trades": int(getattr(t, "num_trades", 0) or 0),
        }
        for i, t in enumerate(trials)
    ]
    param_evolution: list = []
    prev_best = None
    for i, t in enumerate(trials):
        best = getattr(t, "best_objective", None)
        if prev_best is None or (best is not None and best != prev_best):
            bp = getattr(t, "best_params", None)
            param_evolution.append(
                {
                    "iteration": int(getattr(t, "iteration", i + 1)),
                    "best_objective": _coerce(best),
                    "params": bp.to_dict() if bp is not None else {},
                }
            )
            prev_best = best

    payload = {
        "timestamp": now.isoformat(timespec="seconds"),
        "ticker": ticker,
        "timeframe": timeframe,
        "objective": getattr(result, "objective_name", "return"),
        "best_objective": _coerce(getattr(result, "best_objective", 0.0)),
        "oos_objective": _coerce(getattr(result, "oos_objective", None)),
        "best_stats": _clean_stats(getattr(result, "best_stats", {})),
        "oos_stats": _clean_stats(getattr(result, "oos_stats", None)),
        "stop_reason": getattr(result, "stop_reason", "all-trials"),
        "trials": len(trials),
        "searched_params": [s.name for s in getattr(result, "specs", []) or []],
        "params": result.best_params.to_dict(),
        "objective_history": objective_history,
        "param_evolution": param_evolution,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _record_from_payload(path: Path, payload: Dict[str, Any]) -> SearchRecord:
    best_stats = payload.get("best_stats", {}) or {}
    oos_stats = payload.get("oos_stats", {}) or {}
    return SearchRecord(
        path=path,
        timestamp=str(payload.get("timestamp", "")),
        ticker=str(payload.get("ticker", "")),
        objective=str(payload.get("objective", "return")),
        best_objective=float(payload.get("best_objective", 0.0) or 0.0),
        best_return=float(best_stats.get("total_return", 0.0) or 0.0),
        oos_objective=(
            None if payload.get("oos_objective") is None
            else float(payload.get("oos_objective"))
        ),
        oos_return=(
            None if not oos_stats else float(oos_stats.get("total_return", 0.0) or 0.0)
        ),
        stop_reason=str(payload.get("stop_reason", "")),
        trials=int(payload.get("trials", 0) or 0),
        timeframe=str(payload.get("timeframe", "")),
        params=dict(payload.get("params", {}) or {}),
        objective_history=list(payload.get("objective_history", []) or []),
        param_evolution=list(payload.get("param_evolution", []) or []),
    )


def list_searches(directory: Optional[Path] = None) -> List[SearchRecord]:
    """Return stored searches, most recent first. Skips unreadable files."""
    directory = directory or HISTORY_DIR
    if not directory.exists():
        return []
    records: List[SearchRecord] = []
    for path in directory.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            records.append(_record_from_payload(path, payload))
        except Exception:
            continue  # ignore corrupt / partial files
    records.sort(key=lambda r: r.timestamp, reverse=True)
    return records


def params_from_record(record: SearchRecord) -> StrategyParams:
    """Rebuild a :class:`StrategyParams` from a stored record's params dict."""
    valid = {f.name for f in fields(StrategyParams)}
    kwargs = {k: v for k, v in record.params.items() if k in valid}
    return StrategyParams(**kwargs)


def delete_search(path: Path) -> bool:
    """Delete a stored search file. Returns True on success."""
    try:
        Path(path).unlink()
        return True
    except Exception:
        return False
