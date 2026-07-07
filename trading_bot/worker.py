"""Background worker thread.

Polls Alpaca on an interval, runs analysis + the decision engine, optionally
executes small paper trades, and pushes a fully-formed :class:`Update` onto a
thread-safe queue. The GUI thread drains that queue to redraw in real time.

All network I/O lives here so the Tkinter main thread never blocks.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from alpaca.trading.enums import OrderSide

from .alpaca_client import AccountState, AlpacaClient, Fundamentals
from .analysis import Analysis, analyze
from .config import Settings
from .decision import Decision, Verdict, decide
from .news import NewsSentiment, aggregate_news, empty_news, items_from_frame
from .options import OptionsActivity, analyze_options_activity, empty_activity
from .params import StrategyParams
from .risk import PositionState, exit_reason, target_entry_size


@dataclass
class TradeRecord:
    """A trade attempt (or its failure) for the GUI trade log."""

    timestamp: datetime
    side: str
    price: float
    message: str
    ok: bool


@dataclass
class Update:
    """One snapshot of bot state delivered to the GUI."""

    timestamp: datetime
    analysis: Optional[Analysis] = None
    fundamentals: Optional[Fundamentals] = None
    account: Optional[AccountState] = None
    decision: Optional[Decision] = None
    options: Optional[OptionsActivity] = None
    news: Optional[NewsSentiment] = None
    new_trades: List[TradeRecord] = field(default_factory=list)
    error: Optional[str] = None


class BotWorker(threading.Thread):
    """Runs the poll/analyze/decide/trade loop until stopped."""

    def __init__(
        self,
        client: AlpacaClient,
        settings: Settings,
        out_queue: "queue.Queue[Update]",
        params: Optional[StrategyParams] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._client = client
        self._settings = settings
        self._queue = out_queue
        # Live trading parameters. Defaults derive from settings, but the GUI
        # can swap them at runtime (e.g. importing tuned values from a
        # backtest). Assignment is atomic, so no lock is needed.
        self._params = params or StrategyParams.from_settings(settings)
        self._stop = threading.Event()
        # Tracks the open long for ATR-exit evaluation (peak / holding time).
        self._state = PositionState()
        # Auto-trade starts OFF: the bot observes and explains until the
        # user explicitly enables trading from the GUI.
        self._auto_trade = threading.Event()
        # Options flow (UOA) is comparatively heavy (3 API calls) and slow to
        # change, so it is refreshed at most once per this interval and cached.
        self._options_cache: Optional[OptionsActivity] = None
        self._options_last_fetch = 0.0
        self._options_min_interval = 60.0  # seconds
        # News is also slow-moving; refresh on its own cadence + cache.
        self._news_items: list = []
        self._news_last_fetch = 0.0
        self._news_min_interval = 60.0  # seconds

    # ---- Controls (called from GUI thread) ------------------------------

    def stop(self) -> None:
        self._stop.set()

    def set_params(self, params: StrategyParams) -> None:
        """Swap the live strategy parameters (thread-safe atomic assignment)."""
        self._params = params

    @property
    def params(self) -> StrategyParams:
        return self._params

    def set_auto_trade(self, enabled: bool) -> None:
        if enabled:
            self._auto_trade.set()
        else:
            self._auto_trade.clear()

    @property
    def auto_trade_enabled(self) -> bool:
        return self._auto_trade.is_set()

    # ---- Main loop -------------------------------------------------------

    def run(self) -> None:
        while not self._stop.is_set():
            update = self._poll_once()
            self._queue.put(update)
            # Sleep in small slices so Stop is responsive.
            self._stop.wait(timeout=self._settings.poll_interval_seconds)

    def _poll_once(self) -> Update:
        now = datetime.now()
        try:
            bars = self._client.get_minute_bars()
            analysis = analyze(bars, self._params)
            fundamentals = self._client.get_fundamentals()
            account = self._client.get_account_state()
        except Exception as exc:  # network / auth / rate-limit errors
            return Update(timestamp=now, error=f"{type(exc).__name__}: {exc}")

        # Options flow (UOA): refresh on its own slow cadence and reuse the
        # cache otherwise, so it never dominates API usage or slows the loop.
        options = self._maybe_fetch_options(analysis)
        # News sentiment (same lexicon scorer the backtest uses).
        news = self._maybe_fetch_news()

        decision = decide(
            analysis=analysis,
            fundamentals=fundamentals,
            account=account,
            params=self._params,
            options=options,
            news=news,
        )

        # Sync our exit-tracking state with the broker's actual position.
        self._sync_state(account, analysis.last_price)

        trades: List[TradeRecord] = []
        if self._auto_trade.is_set():
            trades = self._maybe_trade(decision, analysis, account)
            # Refresh account after trading so the GUI shows the new position.
            if trades:
                try:
                    account = self._client.get_account_state()
                    self._sync_state(account, analysis.last_price)
                except Exception:
                    pass

        return Update(
            timestamp=now,
            analysis=analysis,
            fundamentals=fundamentals,
            account=account,
            decision=decision,
            options=options,
            news=news,
            new_trades=trades,
        )

    def _maybe_fetch_news(self) -> Optional[NewsSentiment]:
        """Fetch + score recent news on a throttled cadence; aggregate live."""
        if not bool(self._params.use_news_sentiment):
            return None
        now = time.monotonic()
        if now - self._news_last_fetch >= self._news_min_interval or not self._news_items:
            self._news_last_fetch = now
            try:
                df = self._client.get_recent_news(
                    lookback_minutes=int(self._params.news_lookback_minutes)
                )
                self._news_items = items_from_frame(df)
            except Exception:
                # Keep prior items on transient errors.
                pass
        if not self._news_items:
            return empty_news("No recent news.")
        return aggregate_news(
            self._news_items,
            datetime.now(timezone.utc),
            lookback_minutes=self._params.news_lookback_minutes,
            half_life_minutes=self._params.news_half_life_minutes,
        )

    def _maybe_fetch_options(self, analysis: Analysis) -> Optional[OptionsActivity]:
        """Fetch + analyze options flow on a throttled cadence (cached)."""
        if not bool(self._params.use_options_flow):
            return None
        now = time.monotonic()
        if (
            self._options_cache is not None
            and now - self._options_last_fetch < self._options_min_interval
        ):
            return self._options_cache

        self._options_last_fetch = now
        price = analysis.last_price if analysis.has_data else float("nan")
        try:
            frame, as_of, note = self._client.get_options_activity(
                underlying_price=price,
                expiry_days=self._params.uoa_expiry_days,
                strike_pct=self._params.uoa_strike_pct,
            )
            if frame is None:
                self._options_cache = empty_activity(note)
            else:
                self._options_cache = analyze_options_activity(
                    frame, price, self._params, as_of=as_of
                )
        except Exception as exc:
            self._options_cache = empty_activity(
                f"Options scan failed: {type(exc).__name__}: {exc}"
            )
        return self._options_cache

    def _sync_state(self, account: AccountState, price: float) -> None:
        """Reconcile the local PositionState with the broker account."""
        qty = int(account.position_qty)
        if qty <= 0:
            self._state.reset()
            return
        if self._state.shares == 0:
            # Newly observed position (e.g. just filled or pre-existing).
            self._state.open(qty, account.avg_entry_price or price)
        else:
            self._state.shares = qty
            if account.avg_entry_price:
                self._state.avg_entry = account.avg_entry_price
        self._state.step(price)

    def _maybe_trade(
        self, decision: Decision, analysis: Analysis, account: AccountState
    ) -> List[TradeRecord]:
        """Apply ATR exits + volatility-sized entries (paper orders)."""
        price = analysis.last_price
        atr = analysis.atr

        # 1) Defined-risk exit takes priority over any signal.
        if self._state.shares > 0:
            reason = exit_reason(self._state, price, atr, self._params)
            if reason is not None:
                return [self._execute(Verdict.SELL, self._state.shares, price, reason)]

        # 2) Otherwise act on the decision verdict.
        if decision.verdict == Verdict.BUY and self._state.shares == 0:
            qty = target_entry_size(
                account.portfolio_value, account.cash, price, atr, self._params
            )
            if qty > 0:
                return [self._execute(Verdict.BUY, qty, price, "signal")]
        elif decision.verdict == Verdict.SELL and self._state.shares > 0:
            return [self._execute(Verdict.SELL, self._state.shares, price, "signal")]
        return []

    def _execute(
        self, verdict: Verdict, qty: int, price: float, reason: str
    ) -> TradeRecord:
        side = OrderSide.BUY if verdict == Verdict.BUY else OrderSide.SELL
        try:
            message = self._client.submit_market_order(side=side, qty=int(qty))
            if reason != "signal":
                message = f"[{reason}] {message}"
            return TradeRecord(
                timestamp=datetime.now(),
                side=verdict.value,
                price=price,
                message=message,
                ok=True,
            )
        except Exception as exc:
            return TradeRecord(
                timestamp=datetime.now(),
                side=verdict.value,
                price=price,
                message=f"Order failed: {type(exc).__name__}: {exc}",
                ok=False,
            )
