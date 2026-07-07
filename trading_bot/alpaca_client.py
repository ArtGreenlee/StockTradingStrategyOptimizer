"""Thin wrapper around the Alpaca SDK (paper trading only).

Responsibilities:
  * Build the data + trading clients.
  * Fetch historical 1-minute bars (IEX feed, works on free accounts).
  * Fetch a snapshot (latest trade + daily/previous-daily bar) used as the
    "fundamental" data points the bot reasons about.
  * Read account + position state.
  * Submit small market orders against the PAPER account.

Every order goes through ``TradingClient(..., paper=True)`` so no real money
is ever at risk.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import (
    NewsRequest,
    OptionSnapshotRequest,
    StockBarsRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, OrderSide, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest

from .config import Settings
from .metrics import METRICS

# Selectable bar sizes for the backtest downloader: label -> (TimeFrame,
# seconds-per-bar). Seconds-per-bar feeds the simulator's latency model.
TIMEFRAME_CHOICES: "dict[str, Tuple[TimeFrame, int]]" = {
    "1Min": (TimeFrame(amount=1, unit=TimeFrameUnit.Minute), 60),
    "5Min": (TimeFrame(amount=5, unit=TimeFrameUnit.Minute), 300),
    "15Min": (TimeFrame(amount=15, unit=TimeFrameUnit.Minute), 900),
    "30Min": (TimeFrame(amount=30, unit=TimeFrameUnit.Minute), 1800),
    "1Hour": (TimeFrame(amount=1, unit=TimeFrameUnit.Hour), 3600),
    "1Day": (TimeFrame(amount=1, unit=TimeFrameUnit.Day), 23_400),
}


@dataclass
class AccountState:
    """Snapshot of paper account + position for the traded ticker."""

    cash: float
    portfolio_value: float
    buying_power: float
    position_qty: float
    avg_entry_price: float
    unrealized_pl: float
    market_value: float


@dataclass
class Fundamentals:
    """Snapshot-level data points used as 'fundamental' context.

    Alpaca's free feed does not expose ratios like P/E, so we surface the
    real, available snapshot data (daily OHLCV, VWAP, previous close) and let
    the analysis layer derive context (gap, range position, volume vs avg).
    """

    last_price: float
    daily_open: float
    daily_high: float
    daily_low: float
    daily_close: float
    daily_volume: float
    daily_vwap: float
    previous_close: float
    as_of: Optional[datetime]


class AlpacaClient:
    """Wraps data + paper trading clients for a single symbol."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Ticker is mutable at runtime so the GUI can switch symbols without
        # rebuilding the SDK clients.
        self.ticker = settings.ticker
        self._data = StockHistoricalDataClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
        )
        # paper=True is mandatory and never configurable elsewhere.
        self._trading = TradingClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            paper=True,
        )
        # Options data client (for Unusual Options Activity detection).
        self._options = OptionHistoricalDataClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
        )
        # News data client (Benzinga-powered; historical + live).
        self._news = NewsClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
        )
        # Route all outbound traffic through a corporate proxy if configured.
        self._apply_network_settings()
        # Count every REST request these clients make (for the GUI telemetry).
        self._instrument_metrics()

    def set_ticker(self, ticker: str) -> None:
        """Switch the traded symbol (used by the GUI ticker selector)."""
        cleaned = ticker.strip().upper()
        if cleaned:
            self.ticker = cleaned

    # ---- Networking / proxy ---------------------------------------------

    def _apply_network_settings(self) -> None:
        """Apply proxy + TLS settings to both SDK ``requests`` sessions.

        The Alpaca SDK builds each client on a ``requests.Session`` exposed as
        ``_session``. We mutate those sessions directly so every REST call
        (data + trading) honors the corporate proxy and CA settings.
        """
        proxies = self._settings.proxies()
        sessions = [
            getattr(self._data, "_session", None),
            getattr(self._trading, "_session", None),
            getattr(self._options, "_session", None),
            getattr(self._news, "_session", None),
        ]

        for session in sessions:
            if session is None:
                continue
            if proxies:
                session.proxies.update(proxies)
            if self._settings.proxy_ca_bundle:
                # Secure handling of a TLS-intercepting proxy: trust the
                # corporate root CA instead of disabling verification.
                session.verify = self._settings.proxy_ca_bundle
            elif not self._settings.proxy_verify_ssl:
                session.verify = False

        # If verification is disabled, silence the noisy per-request warning
        # but make the insecure state obvious via ``network_summary``.
        if not self._settings.proxy_verify_ssl and not self._settings.proxy_ca_bundle:
            try:
                from urllib3.exceptions import InsecureRequestWarning

                warnings.simplefilter("ignore", InsecureRequestWarning)
            except Exception:
                pass

    @property
    def network_summary(self) -> str:
        """Short description of the active proxy/TLS configuration."""
        return self._settings.proxy_description

    def _instrument_metrics(self) -> None:
        """Hook each SDK ``requests.Session`` to count every REST request.

        Wrapping ``Session.request`` captures all HTTP calls (the SDK's get/
        post/etc. all route through it), including pagination -- giving an
        accurate API-call rate for the GUI without instrumenting each method.
        """
        for client in (self._data, self._trading, self._options, self._news):
            session = getattr(client, "_session", None)
            if session is None or getattr(session, "_metrics_wrapped", False):
                continue
            original = session.request

            def make_counted(orig):
                def counted(*args, **kwargs):
                    METRICS.record_api_call()
                    return orig(*args, **kwargs)
                return counted

            session.request = make_counted(original)
            session._metrics_wrapped = True

    def verify_connectivity(self) -> Tuple[bool, str]:
        """Make one lightweight call to confirm the API is reachable.

        Returns ``(ok, message)``. Used by the GUI's "Test connection"
        button so proxy issues surface immediately rather than on first poll.
        """
        try:
            account = self._trading.get_account()
            return True, (
                f"Connected via {self._settings.proxy_description}. "
                f"Account status: {account.status}."
            )
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # ---- Market data -----------------------------------------------------

    def get_minute_bars(self) -> pd.DataFrame:
        """Return recent 1-minute bars as a tidy DataFrame.

        Columns: open, high, low, close, volume, vwap; indexed by timestamp
        (tz-aware UTC). Returns an empty frame if no data is available.
        """
        end = datetime.now(timezone.utc)
        # Pad the window so weekends/holidays still yield a full session.
        start = end - timedelta(minutes=self._settings.lookback_minutes, days=5)

        request = StockBarsRequest(
            symbol_or_symbols=self.ticker,
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = self._data.get_stock_bars(request)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap"]
            )

        # When a single symbol is requested the frame is multi-indexed
        # (symbol, timestamp); drop the symbol level for convenience.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(self.ticker, level="symbol")

        df = df.tail(self._settings.lookback_minutes).copy()
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def download_history(
        self,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame,
    ) -> pd.DataFrame:
        """Download historical bars for an arbitrary range + timeframe.

        Used by the Backtest tab. Returns a tidy OHLCV(+vwap) DataFrame indexed
        by tz-aware UTC timestamp; raises on API error so the GUI can report it.
        """
        request = StockBarsRequest(
            symbol_or_symbols=self.ticker,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = self._data.get_stock_bars(request)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap"]
            )
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(self.ticker, level="symbol")
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    # ---- News (sentiment) -----------------------------------------------

    def get_news(
        self, start: datetime, end: datetime, limit_total: int = 5000
    ) -> pd.DataFrame:
        """Fetch news articles for the ticker over a date range.

        Returns a tidy DataFrame (columns: ``timestamp`` tz-aware UTC,
        ``headline``, ``summary``, ``source``) sorted ascending. Works for both
        backtest history and live (small range). Paginates until exhausted or
        ``limit_total`` is reached. Returns an empty frame on no data.
        """
        rows = []
        page_token = None
        try:
            while len(rows) < limit_total:
                req = NewsRequest(
                    symbols=self.ticker,
                    start=start,
                    end=end,
                    sort="asc",
                    limit=50,
                    include_content=False,
                    exclude_contentless=True,
                    page_token=page_token,
                )
                resp = self._news.get_news(req)
                articles = getattr(resp, "news", None) or getattr(resp, "data", None) or []
                # SDK may return a NewsSet with a "data" dict; normalize.
                if isinstance(articles, dict):
                    articles = articles.get("news", [])
                for a in articles:
                    ts = getattr(a, "created_at", None) or getattr(a, "updated_at", None)
                    rows.append({
                        "timestamp": pd.Timestamp(ts, tz="UTC") if ts is not None else pd.NaT,
                        "headline": getattr(a, "headline", "") or "",
                        "summary": getattr(a, "summary", "") or "",
                        "source": getattr(a, "source", "") or "",
                    })
                page_token = getattr(resp, "next_page_token", None)
                if not page_token or not articles:
                    break
        except Exception:
            # No news entitlement / network error -> empty (caller degrades).
            if not rows:
                return pd.DataFrame(columns=["timestamp", "headline", "summary", "source"])

        if not rows:
            return pd.DataFrame(columns=["timestamp", "headline", "summary", "source"])
        df = pd.DataFrame(rows).dropna(subset=["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_recent_news(self, lookback_minutes: int = 240) -> pd.DataFrame:
        """Fetch recent news for the ticker (live use)."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=max(5, lookback_minutes))
        return self.get_news(start, end, limit_total=200)

    def get_fundamentals(self) -> Optional[Fundamentals]:
        """Return snapshot data for the ticker, or None if unavailable."""
        request = StockSnapshotRequest(
            symbol_or_symbols=self.ticker, feed=DataFeed.IEX
        )
        snapshot = self._data.get_stock_snapshot(request)
        snap = snapshot.get(self.ticker)
        if snap is None:
            return None

        latest_trade = snap.latest_trade
        daily = snap.daily_bar
        prev = snap.previous_daily_bar

        last_price = float(latest_trade.price) if latest_trade else float("nan")
        return Fundamentals(
            last_price=last_price,
            daily_open=float(daily.open) if daily else float("nan"),
            daily_high=float(daily.high) if daily else float("nan"),
            daily_low=float(daily.low) if daily else float("nan"),
            daily_close=float(daily.close) if daily else float("nan"),
            daily_volume=float(daily.volume) if daily else float("nan"),
            daily_vwap=float(daily.vwap) if daily and daily.vwap else float("nan"),
            previous_close=float(prev.close) if prev else float("nan"),
            as_of=latest_trade.timestamp if latest_trade else None,
        )

    # ---- Options (Unusual Options Activity) -----------------------------

    def get_options_activity(
        self,
        underlying_price: float,
        expiry_days: int = 45,
        strike_pct: float = 0.15,
        max_contracts: int = 250,
    ) -> Tuple[Optional[pd.DataFrame], Optional[object], str]:
        """Fetch a near-the-money options chain for UOA detection.

        Returns ``(frame, as_of, note)``. ``frame`` is one row per contract
        with columns expected by :func:`trading_bot.options.analyze_options_activity`
        (symbol, type, strike, expiration, open_interest, volume, last_price,
        bid, ask, implied_volatility, delta). On any failure (e.g. no options
        data entitlement) returns ``(None, None, reason)`` so the caller can
        degrade gracefully.

        Three lightweight calls: list contracts (open interest), snapshot chain
        (IV/greeks/quote/last trade), and a batched daily-bar pull (volume).
        """
        if not (underlying_price and underlying_price > 0):
            return None, None, "No underlying price yet for options scan."

        lo = underlying_price * (1.0 - strike_pct)
        hi = underlying_price * (1.0 + strike_pct)
        today = datetime.now(timezone.utc).date()
        exp_lte = today + timedelta(days=max(1, expiry_days))

        # 1) Contracts (reference data incl. open interest).
        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[self.ticker],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=today,
                expiration_date_lte=exp_lte,
                strike_price_gte=str(round(lo, 2)),
                strike_price_lte=str(round(hi, 2)),
                limit=max_contracts,
            )
            resp = self._trading.get_option_contracts(req)
            contracts = getattr(resp, "option_contracts", None) or []
        except Exception as exc:
            return None, None, f"Options unavailable: {type(exc).__name__}: {exc}"

        if not contracts:
            return None, None, "No option contracts in the scan window."

        rows = {}
        for c in contracts:
            rows[c.symbol] = {
                "symbol": c.symbol,
                "type": getattr(c.type, "value", str(c.type)).lower(),
                "strike": float(c.strike_price),
                "expiration": str(c.expiration_date),
                "open_interest": float(c.open_interest) if c.open_interest is not None else 0.0,
                "volume": 0.0,
                "last_price": float(c.close_price) if c.close_price is not None else float("nan"),
                "bid": float("nan"),
                "ask": float("nan"),
                "implied_volatility": float("nan"),
                "delta": float("nan"),
            }

        symbols = list(rows.keys())[:max_contracts]

        # 2) Snapshot chain (IV, greeks, latest trade + quote).
        feed = self._settings.options_feed or "indicative"
        try:
            snaps = self._options.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=symbols, feed=feed)
            )
        except Exception:
            snaps = {}
        for sym, snap in (snaps or {}).items():
            if sym not in rows or snap is None:
                continue
            r = rows[sym]
            if getattr(snap, "implied_volatility", None) is not None:
                r["implied_volatility"] = float(snap.implied_volatility)
            greeks = getattr(snap, "greeks", None)
            if greeks is not None and getattr(greeks, "delta", None) is not None:
                r["delta"] = float(greeks.delta)
            lt = getattr(snap, "latest_trade", None)
            if lt is not None and getattr(lt, "price", None) is not None:
                r["last_price"] = float(lt.price)
                r["volume"] = max(r["volume"], float(getattr(lt, "size", 0) or 0))
            lq = getattr(snap, "latest_quote", None)
            if lq is not None:
                if getattr(lq, "bid_price", None) is not None:
                    r["bid"] = float(lq.bid_price)
                if getattr(lq, "ask_price", None) is not None:
                    r["ask"] = float(lq.ask_price)

        # 3) Day volume via batched daily bars (authoritative volume).
        try:
            from alpaca.data.requests import OptionBarsRequest

            bars = self._options.get_option_bars(
                OptionBarsRequest(
                    symbol_or_symbols=symbols,
                    timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),
                    start=today - timedelta(days=4),
                )
            )
            bdf = bars.df
            if bdf is not None and not bdf.empty:
                if isinstance(bdf.index, pd.MultiIndex):
                    last_vol = bdf.groupby(level="symbol")["volume"].last()
                    for sym, vol in last_vol.items():
                        if sym in rows:
                            rows[sym]["volume"] = max(rows[sym]["volume"], float(vol))
        except Exception:
            pass  # fall back to last-trade size already stored

        frame = pd.DataFrame(list(rows.values()))
        note = f"Scanned {len(frame)} contracts via {feed} feed."
        return frame, datetime.now(timezone.utc), note

    # ---- Account / positions --------------------------------------------

    def get_account_state(self) -> AccountState:
        """Return account balances + current position in the traded ticker."""
        account = self._trading.get_account()

        position_qty = 0.0
        avg_entry = 0.0
        unrealized_pl = 0.0
        market_value = 0.0
        try:
            position = self._trading.get_open_position(self.ticker)
            position_qty = float(position.qty)
            avg_entry = float(position.avg_entry_price)
            unrealized_pl = float(position.unrealized_pl)
            market_value = float(position.market_value)
        except Exception:
            # No open position for this symbol -> all zeros.
            pass

        return AccountState(
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            position_qty=position_qty,
            avg_entry_price=avg_entry,
            unrealized_pl=unrealized_pl,
            market_value=market_value,
        )

    # ---- Orders ----------------------------------------------------------

    def submit_market_order(self, side: OrderSide, qty: int) -> str:
        """Submit a small market order to the PAPER account.

        Returns a human-readable confirmation string. Raises on API error.
        """
        order = MarketOrderRequest(
            symbol=self.ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        submitted = self._trading.submit_order(order)
        return (
            f"{side.value.upper()} {qty} {self.ticker} "
            f"(order {str(submitted.id)[:8]}, status={submitted.status})"
        )
