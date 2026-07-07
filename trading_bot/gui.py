"""Tkinter GUI with real-time charts + a transparent decision panel.

Left side  : every step of the decision process (volatility, RSI, Bollinger,
             trend, fundamentals, risk) plus account state, controls and a
             trade log.
Right side : live matplotlib charts (price + Bollinger/SMA + trade markers,
             volatility, RSI, volume) that redraw on every poll.

The GUI only renders. A :class:`BotWorker` thread does all network work and
hands snapshots over a queue, which we drain on the Tk main loop.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from tkinter import scrolledtext, ttk
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .alpaca_client import AccountState, AlpacaClient
from .analysis import Analysis
from .backtest_panel import BacktestPanel
from .config import Settings
from .decision import Component, Decision, State, Verdict
from .metrics import METRICS
from .orchestrator_panel import OrchestratorPanel
from .params import StrategyParams
from .theme import BG, BLUE, FG, GRAY, GREEN, MUTED, ORANGE, PANEL, RED
from .worker import BotWorker, TradeRecord, Update

STATE_COLORS = {
    State.BULLISH: GREEN,
    State.BEARISH: RED,
    State.NEUTRAL: GRAY,
    State.BLOCKED: ORANGE,
    State.INFO: BLUE,
}

VERDICT_COLORS = {
    Verdict.BUY: GREEN,
    Verdict.SELL: RED,
    Verdict.HOLD: GRAY,
}


class TradingBotGUI:
    def __init__(self, settings: Settings, client: AlpacaClient) -> None:
        self._settings = settings
        self._client = client
        self._queue: "queue.Queue[Update]" = queue.Queue()
        # Carries plain-text notifications (e.g. connection-test results) from
        # background threads back to the Tk main loop for safe logging.
        self._notify_queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[BotWorker] = None
        # Active live-trading strategy parameters (importable from Backtest).
        self._live_params = StrategyParams.from_settings(settings)

        # Trade markers persist across redraws: (timestamp, price).
        self._buy_markers: List[Tuple[datetime, float]] = []
        self._sell_markers: List[Tuple[datetime, float]] = []
        # Symbols of options contracts already logged as unusual (dedupe).
        self._seen_uoa: set = set()

        self._root = tk.Tk()
        self._root.title(f"Volatility Bot - {settings.ticker} (PAPER)")
        self._root.configure(bg=BG)
        self._root.geometry("1480x900")
        self._root.minsize(1180, 760)

        self._build_styles()
        self._build_topbar()
        self._build_body()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start STOPPED so the app doesn't hit the Alpaca API until the user
        # explicitly connects (press Start). The queue drainer still runs so
        # the UI is ready to render the moment polling begins.
        self._root.after(250, self._drain_queue)
        self._root.after(1000, self._update_telemetry)

    # ---- Styling ---------------------------------------------------------

    def _build_styles(self) -> None:
        style = ttk.Style(self._root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Panel.TLabel", background=PANEL, foreground=FG)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure(
            "Header.TLabel", background=BG, foreground=FG,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.map("TCheckbutton", background=[("active", BG)])
        # Notebook (mode tabs).
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab", background=PANEL, foreground=MUTED,
            padding=(14, 6), font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", BG)], foreground=[("selected", FG)],
        )
        style.configure(
            "TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG,
            arrowcolor=FG,
        )
        # The combobox FIELD is themed above, but its drop-down popup is a Tk
        # Listbox that ignores ttk styling (defaults to white-on-white). Style
        # it via the option database, and map the readonly/selected states so
        # the field text stays readable.
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL), ("disabled", PANEL)],
            foreground=[("readonly", FG), ("disabled", MUTED)],
            selectbackground=[("readonly", PANEL)],
            selectforeground=[("readonly", FG)],
            arrowcolor=[("readonly", FG)],
        )
        self._root.option_add("*TCombobox*Listbox.background", PANEL)
        self._root.option_add("*TCombobox*Listbox.foreground", FG)
        self._root.option_add("*TCombobox*Listbox.selectBackground", BLUE)
        self._root.option_add("*TCombobox*Listbox.selectForeground", "white")
        self._root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 9))
        style.configure("TProgressbar", background=GREEN, troughcolor=PANEL)

    # ---- Top bar ---------------------------------------------------------

    def _build_topbar(self) -> None:
        bar = ttk.Frame(self._root, padding=(12, 10))
        bar.pack(side=tk.TOP, fill=tk.X)

        # Editable ticker selector.
        self._ticker_var = tk.StringVar(value=self._settings.ticker)
        ticker_entry = tk.Entry(
            bar, textvariable=self._ticker_var, width=6, bg=PANEL, fg=FG,
            insertbackground=FG, relief=tk.FLAT, justify="center",
            font=("Segoe UI", 16, "bold"),
        )
        ticker_entry.pack(side=tk.LEFT)
        ticker_entry.bind("<Return>", lambda e: self._on_apply_ticker())
        tk.Button(
            bar, text="Set", command=self._on_apply_ticker, bg=PANEL, fg=FG,
            activebackground=GRAY, relief=tk.FLAT, padx=6,
        ).pack(side=tk.LEFT, padx=(4, 0))

        self._price_var = tk.StringVar(value="--")
        ttk.Label(
            bar, textvariable=self._price_var, style="TLabel",
            font=("Segoe UI", 18),
        ).pack(side=tk.LEFT, padx=(12, 0))

        # PAPER badge so it's always obvious no real money is involved.
        paper = tk.Label(
            bar, text=" PAPER MODE ", bg=BLUE, fg="white",
            font=("Segoe UI", 9, "bold"),
        )
        paper.pack(side=tk.LEFT, padx=16)

        # Proxy badge: green when routing through a proxy, muted when direct.
        insecure = not self._settings.proxy_verify_ssl and not self._settings.proxy_ca_bundle
        if self._settings.proxy_enabled:
            proxy_text, proxy_bg = " PROXY ", (ORANGE if insecure else GREEN)
        else:
            proxy_text, proxy_bg = " DIRECT ", GRAY
        self._proxy_badge = tk.Label(
            bar, text=proxy_text, bg=proxy_bg, fg="white",
            font=("Segoe UI", 9, "bold"),
        )
        self._proxy_badge.pack(side=tk.LEFT, padx=(0, 8))

        # Verdict badge.
        self._verdict_lbl = tk.Label(
            bar, text=" HOLD ", bg=GRAY, fg="white",
            font=("Segoe UI", 14, "bold"), padx=10, pady=2,
        )
        self._verdict_lbl.pack(side=tk.LEFT, padx=8)

        # Telemetry: Alpaca API call rate + LLM token usage.
        telemetry = tk.Frame(bar, bg=BG)
        telemetry.pack(side=tk.LEFT, padx=(12, 0))
        self._api_rate_var = tk.StringVar(value="API  --/s")
        tk.Label(
            telemetry, textvariable=self._api_rate_var, bg=PANEL, fg=GREEN,
            font=("Consolas", 9), padx=8, pady=2,
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._token_rate_var = tk.StringVar(value="LLM  -- tok/min")
        tk.Label(
            telemetry, textvariable=self._token_rate_var, bg=PANEL, fg=BLUE,
            font=("Consolas", 9), padx=8, pady=2,
        ).pack(side=tk.LEFT)

        # Controls on the right.
        controls = ttk.Frame(bar)
        controls.pack(side=tk.RIGHT)

        self._auto_trade_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls, text="Auto-trade (paper)",
            variable=self._auto_trade_var, command=self._on_toggle_auto,
        ).pack(side=tk.RIGHT, padx=8)

        self._start_btn = tk.Button(
            controls, text="Start", command=self._on_start_stop,
            bg=PANEL, fg=FG, activebackground=GRAY, relief=tk.FLAT, padx=12,
        )
        self._start_btn.pack(side=tk.RIGHT, padx=8)

        self._test_btn = tk.Button(
            controls, text="Test connection", command=self._on_test_connection,
            bg=PANEL, fg=FG, activebackground=GRAY, relief=tk.FLAT, padx=12,
        )
        self._test_btn.pack(side=tk.RIGHT, padx=8)

        self._status_var = tk.StringVar(value="stopped - press Start to connect")
        ttk.Label(
            bar, textvariable=self._status_var, style="Muted.TLabel"
        ).pack(side=tk.RIGHT, padx=12)

    # ---- Body (left decision column + right charts) ----------------------

    def _build_body(self) -> None:
        body = ttk.Frame(self._root, padding=(10, 0, 10, 10))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Two modes live in a tabbed notebook: live trading and backtesting.
        self._notebook = ttk.Notebook(body)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        live_tab = ttk.Frame(self._notebook, style="TFrame", padding=(8, 8))
        self._notebook.add(live_tab, text="  Live Trading  ")

        left = ttk.Frame(live_tab, width=430)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        ttk.Label(left, text="Decision process", style="Header.TLabel").pack(
            anchor="w", pady=(0, 6)
        )
        self._summary_var = tk.StringVar(value="Initializing...")
        ttk.Label(
            left, textvariable=self._summary_var, style="Muted.TLabel",
            wraplength=410, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        # Scrollable component list.
        comp_wrap = ttk.Frame(left, style="Panel.TFrame")
        comp_wrap.pack(fill=tk.BOTH, expand=True)
        self._comp_canvas = tk.Canvas(
            comp_wrap, bg=PANEL, highlightthickness=0, bd=0
        )
        scrollbar = ttk.Scrollbar(
            comp_wrap, orient="vertical", command=self._comp_canvas.yview
        )
        self._comp_inner = ttk.Frame(self._comp_canvas, style="Panel.TFrame")
        self._comp_inner.bind(
            "<Configure>",
            lambda e: self._comp_canvas.configure(
                scrollregion=self._comp_canvas.bbox("all")
            ),
        )
        self._comp_canvas.create_window((0, 0), window=self._comp_inner, anchor="nw")
        self._comp_canvas.configure(yscrollcommand=scrollbar.set)
        self._comp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Account panel.
        ttk.Label(left, text="Account & position", style="Header.TLabel").pack(
            anchor="w", pady=(10, 4)
        )
        self._account_var = tk.StringVar(value="Loading account...")
        ttk.Label(
            left, textvariable=self._account_var, style="Muted.TLabel",
            wraplength=410, justify="left",
        ).pack(anchor="w")

        # Active strategy parameters (importable from the Backtest tab).
        params_header = ttk.Frame(left, style="TFrame")
        params_header.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(
            params_header, text="Strategy parameters", style="Header.TLabel"
        ).pack(side=tk.LEFT)
        tk.Button(
            params_header, text="Import from Backtest",
            command=self._on_import_params, bg=BLUE, fg="white",
            activebackground=GRAY, relief=tk.FLAT, padx=8,
        ).pack(side=tk.RIGHT)
        self._params_text = tk.Text(
            left, height=10, bg=PANEL, fg=FG, insertbackground=FG,
            relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD,
        )
        self._params_text.pack(fill=tk.X)
        self._params_text.configure(state=tk.DISABLED)

        # Unusual Options Activity (UOA) flow panel.
        ttk.Label(left, text="Options flow (UOA)", style="Header.TLabel").pack(
            anchor="w", pady=(10, 4)
        )
        self._uoa_text = tk.Text(
            left, height=8, bg=PANEL, fg=FG, insertbackground=FG,
            relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD,
        )
        self._uoa_text.pack(fill=tk.X)
        self._uoa_text.insert(tk.END, "Options flow loads after you press Start.")
        self._uoa_text.configure(state=tk.DISABLED)

        # News sentiment panel.
        ttk.Label(left, text="News sentiment", style="Header.TLabel").pack(
            anchor="w", pady=(10, 4)
        )
        self._news_text = tk.Text(
            left, height=5, bg=PANEL, fg=FG, insertbackground=FG,
            relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD,
        )
        self._news_text.pack(fill=tk.X)
        self._news_text.insert(tk.END, "News loads after you press Start.")
        self._news_text.configure(state=tk.DISABLED)

        # Trade log.
        ttk.Label(left, text="Trade log", style="Header.TLabel").pack(
            anchor="w", pady=(10, 4)
        )
        self._log = scrolledtext.ScrolledText(
            left, height=8, bg=PANEL, fg=FG, insertbackground=FG,
            relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD,
        )
        self._log.pack(fill=tk.X)
        self._log.configure(state=tk.DISABLED)

        # Charts on the right.
        right = ttk.Frame(live_tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self._build_charts(right)

        # Second mode: backtest / simulation tab.
        bt_tab = ttk.Frame(self._notebook, style="TFrame")
        self._notebook.add(bt_tab, text="  Backtest / Simulation  ")
        self._backtest = BacktestPanel(bt_tab, self._client, self._settings)

        # Third mode: Orchestrator (LLM/Copilot sentiment analysis).
        orch_tab = ttk.Frame(self._notebook, style="TFrame")
        self._notebook.add(orch_tab, text="  Orchestrator  ")
        self._orchestrator = OrchestratorPanel(orch_tab, self._settings)

        self._refresh_params_display()

    def _build_charts(self, parent: ttk.Frame) -> None:
        self._fig = Figure(figsize=(9, 8), dpi=100, facecolor=BG)
        self._fig.subplots_adjust(
            left=0.08, right=0.97, top=0.96, bottom=0.06, hspace=0.32
        )
        gs = self._fig.add_gridspec(4, 1, height_ratios=[3, 2, 2, 1.4])
        self._ax_price = self._fig.add_subplot(gs[0])
        self._ax_vol = self._fig.add_subplot(gs[1], sharex=self._ax_price)
        self._ax_rsi = self._fig.add_subplot(gs[2], sharex=self._ax_price)
        self._ax_volume = self._fig.add_subplot(gs[3], sharex=self._ax_price)

        for ax in (self._ax_price, self._ax_vol, self._ax_rsi, self._ax_volume):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRAY)
            ax.grid(True, color="#2a313c", linewidth=0.6)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_charts()

    def _draw_empty_charts(self) -> None:
        self._ax_price.set_title(
            "Waiting for market data...", color=FG, fontsize=11
        )
        self._canvas.draw_idle()

    # ---- Worker lifecycle ------------------------------------------------

    def _start_worker(self) -> None:
        self._worker = BotWorker(
            self._client, self._settings, self._queue, params=self._live_params
        )
        self._worker.set_auto_trade(self._auto_trade_var.get())
        self._worker.start()
        self._status_var.set(
            f"running - polling every {self._settings.poll_interval_seconds}s"
        )
        self._start_btn.configure(text="Stop")

    def _on_start_stop(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.stop()
            self._worker = None
            self._status_var.set("stopped")
            self._start_btn.configure(text="Start")
        else:
            self._start_worker()

    def _on_toggle_auto(self) -> None:
        enabled = self._auto_trade_var.get()
        if self._worker is not None:
            self._worker.set_auto_trade(enabled)
        self._log_line(
            f"Auto-trade {'ENABLED' if enabled else 'disabled'} (paper)."
        )

    def _on_test_connection(self) -> None:
        """Verify API reachability (through the proxy) without blocking the UI."""
        self._test_btn.configure(state=tk.DISABLED, text="Testing...")
        self._notify_queue.put(
            f"Testing connection ({self._settings.proxy_description})..."
        )

        def worker() -> None:
            ok, message = self._client.verify_connectivity()
            tag = "OK" if ok else "FAIL"
            self._notify_queue.put(f"Connection test {tag}: {message}")
            self._notify_queue.put("__test_done__")

        threading.Thread(target=worker, daemon=True).start()

    # ---- Ticker + parameter controls ------------------------------------

    def _on_apply_ticker(self) -> None:
        """Switch the traded symbol live, rebuilding the polling loop."""
        new = self._ticker_var.get().strip().upper()
        if not new or not new.isalnum():
            self._log_line(f"Invalid ticker: '{new}'. Use letters/digits only.")
            self._ticker_var.set(self._settings.ticker)
            return
        if new == self._settings.ticker:
            return

        was_running = self._worker is not None and self._worker.is_alive()
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        self._settings = replace(self._settings, ticker=new)
        self._client.set_ticker(new)
        self._backtest.set_ticker(new)

        # Reset the live view for the new symbol.
        self._ticker_var.set(new)
        self._price_var.set("--")
        self._buy_markers.clear()
        self._sell_markers.clear()
        self._root.title(f"Volatility Bot - {new} (PAPER)")
        self._draw_empty_charts()
        self._log_line(f"Ticker changed to {new}.")

        if was_running:
            self._start_worker()
        else:
            self._status_var.set("stopped")

    def _on_import_params(self) -> None:
        """Pull the current editor params from the Backtest tab into live trading."""
        try:
            params = self._backtest.get_strategy_params()
        except Exception as exc:
            self._log_line(f"Import failed: {exc}")
            return
        self._live_params = params
        if self._worker is not None:
            self._worker.set_params(params)
        self._refresh_params_display()
        self._log_line(
            "Imported strategy parameters from the Backtest tab into live trading."
        )

    def _refresh_params_display(self) -> None:
        p = self._live_params
        exits = "on" if p.use_atr_exits else "off"
        sizing = "vol" if p.use_vol_sizing else "fixed"
        tfilt = f"SMA{p.trend_filter_period}" if p.use_trend_filter else "off"
        cap = f"{p.max_position_shares} sh" if p.max_position_shares else "no cap"
        lines = [
            f"buy / sell score : +{p.buy_threshold:g} / {p.sell_threshold:g}",
            f"RSI({p.rsi_period})       : <{p.rsi_oversold:g} / >{p.rsi_overbought:g}  w={p.rsi_weight:g}",
            f"Bollinger %B     : <{p.bb_low:g} / >{p.bb_high:g}  w={p.bb_weight:g}",
            f"Z-score({p.zscore_period})   : |z|>{p.zscore_entry:g}  w={p.zscore_weight:g}",
            f"trend filter     : {tfilt}",
            f"ATR exits ({exits})  : stop {p.stop_loss_atr:g}x / TP {p.take_profit_atr:g}x"
            f" / trail {p.trailing_stop_atr:g}x / hold {p.max_hold_bars}",
            f"sizing ({sizing})     : risk {p.risk_per_trade_pct:g}%/trade, "
            f"max {p.max_position_pct:g}% equity, {cap}",
        ]
        self._params_text.configure(state=tk.NORMAL)
        self._params_text.delete("1.0", tk.END)
        self._params_text.insert(tk.END, "\n".join(lines))
        self._params_text.configure(state=tk.DISABLED)

    # ---- Queue draining / rendering -------------------------------------

    def _update_telemetry(self) -> None:
        """Refresh the API-call-rate and LLM-token-usage readouts (~1 Hz)."""
        snap = METRICS.snapshot(calls_window=5.0)
        self._api_rate_var.set(
            f"API {snap.api_calls_per_second:.1f}/s  ({snap.api_calls_total:,})"
        )
        tpm = snap.tokens_per_minute
        total = snap.tokens_total
        total_txt = f"{total / 1000:.1f}k" if total >= 1000 else str(total)
        self._token_rate_var.set(f"LLM {tpm:,} tok/min  ({total_txt})")
        self._root.after(1000, self._update_telemetry)

    def _drain_queue(self) -> None:
        latest: Optional[Update] = None
        trades: List[TradeRecord] = []
        try:
            while True:
                update = self._queue.get_nowait()
                latest = update
                trades.extend(update.new_trades)
        except queue.Empty:
            pass

        # Drain plain-text notifications from background threads.
        try:
            while True:
                note = self._notify_queue.get_nowait()
                if note == "__test_done__":
                    self._test_btn.configure(state=tk.NORMAL, text="Test connection")
                else:
                    stamp = datetime.now().strftime("%H:%M:%S")
                    self._log_line(f"[{stamp}] {note}")
        except queue.Empty:
            pass

        if latest is not None:
            self._render(latest, trades)

        self._root.after(250, self._drain_queue)

    def _render(self, update: Update, trades: List[TradeRecord]) -> None:
        stamp = update.timestamp.strftime("%H:%M:%S")
        if update.error:
            self._status_var.set(f"error @ {stamp}")
            self._log_line(f"[{stamp}] ERROR: {update.error}")
            return

        self._status_var.set(f"updated {stamp}")

        if update.analysis is not None:
            self._render_price_header(update.analysis)
        if update.decision is not None:
            self._render_decision(update.decision)
        if update.account is not None:
            self._render_account(update.account)
        if update.options is not None:
            self._render_options(update.options)
        if update.news is not None:
            self._render_news(update.news)

        # Record trade markers against the latest bar timestamp.
        if trades and update.analysis is not None and update.analysis.has_data:
            x = update.analysis.frame.index[-1].to_pydatetime()
            for trade in trades:
                if trade.ok and trade.side == Verdict.BUY.value:
                    self._buy_markers.append((x, trade.price))
                elif trade.ok and trade.side == Verdict.SELL.value:
                    self._sell_markers.append((x, trade.price))
                tag = "OK" if trade.ok else "FAIL"
                self._log_line(
                    f"[{trade.timestamp.strftime('%H:%M:%S')}] {tag} "
                    f"{trade.message}"
                )

        if update.analysis is not None:
            self._render_charts(update.analysis)

    def _render_price_header(self, analysis: Analysis) -> None:
        if analysis.has_data:
            self._price_var.set(f"${analysis.last_price:,.2f}")

    def _render_decision(self, decision: Decision) -> None:
        self._summary_var.set(decision.summary)
        color = VERDICT_COLORS.get(decision.verdict, GRAY)
        self._verdict_lbl.configure(
            text=f" {decision.verdict.value} ", bg=color
        )

        for child in self._comp_inner.winfo_children():
            child.destroy()

        for comp in decision.components:
            self._build_component_card(comp)

    def _build_component_card(self, comp: Component) -> None:
        color = STATE_COLORS.get(comp.state, GRAY)
        card = tk.Frame(self._comp_inner, bg=PANEL)
        card.pack(fill=tk.X, padx=6, pady=4)

        accent = tk.Frame(card, bg=color, width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y)

        text_wrap = tk.Frame(card, bg=PANEL)
        text_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4), pady=4)

        header = tk.Frame(text_wrap, bg=PANEL)
        header.pack(fill=tk.X)
        tk.Label(
            header, text=comp.name, bg=PANEL, fg=FG,
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            header, text=comp.state.value, bg=PANEL, fg=color,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.RIGHT)
        if comp.score:
            tk.Label(
                header, text=f"{comp.score:+.1f}", bg=PANEL, fg=MUTED,
                font=("Consolas", 9),
            ).pack(side=tk.RIGHT, padx=6)

        tk.Label(
            text_wrap, text=comp.detail, bg=PANEL, fg=MUTED,
            font=("Segoe UI", 9), wraplength=360, justify="left",
        ).pack(anchor="w", pady=(2, 0))

    def _render_account(self, account: AccountState) -> None:
        self._account_var.set(
            f"Cash: ${account.cash:,.2f}   "
            f"Portfolio: ${account.portfolio_value:,.2f}\n"
            f"Position: {account.position_qty:.0f} shares "
            f"@ avg ${account.avg_entry_price:,.2f}\n"
            f"Market value: ${account.market_value:,.2f}   "
            f"Unrealized P/L: ${account.unrealized_pl:,.2f}"
        )

    def _render_options(self, options) -> None:
        """Render the Options-flow (UOA) panel + log notable new flags."""
        lines: List[str] = []
        if not getattr(options, "available", False):
            lines.append(options.note or "Options flow unavailable.")
        else:
            bias = (
                "BULLISH" if options.sentiment > 0.05
                else "BEARISH" if options.sentiment < -0.05
                else "neutral"
            )
            pcr = options.put_call_volume_ratio
            pcr_txt = f"{pcr:.2f}" if pcr == pcr else "n/a"  # NaN check
            iv_txt = f"{options.avg_iv * 100:.1f}%" if options.avg_iv == options.avg_iv else "n/a"
            lines.append(
                f"Sentiment {options.sentiment:+.2f} ({bias})  "
                f"| {options.unusual_count} unusual"
            )
            lines.append(
                f"P/C vol {pcr_txt}  call {options.call_volume:,.0f} / "
                f"put {options.put_volume:,.0f}"
            )
            lines.append(
                f"Buy-prem  call ${options.call_premium:,.0f} / "
                f"put ${options.put_premium:,.0f}  | avgIV {iv_txt}"
            )
            for f in options.flagged[:4]:
                lines.append(
                    f"  {f.option_type.upper()} {f.strike:g} exp {f.expiration} "
                    f"V/OI {f.vol_oi:.1f}x {f.aggressor} ${f.premium:,.0f}"
                )
            # Log newly-seen flagged contracts once (avoid log spam).
            for f in options.flagged:
                if f.symbol not in self._seen_uoa:
                    self._seen_uoa.add(f.symbol)
                    self._log_line(
                        f"[UOA] {f.option_type.upper()} {f.strike:g} "
                        f"{f.expiration}: {'; '.join(f.reasons)} ({f.aggressor})"
                    )

        self._uoa_text.configure(state=tk.NORMAL)
        self._uoa_text.delete("1.0", tk.END)
        self._uoa_text.insert(tk.END, "\n".join(lines))
        self._uoa_text.configure(state=tk.DISABLED)

    def _render_news(self, news) -> None:
        """Render the News-sentiment panel."""
        lines: List[str] = []
        if not getattr(news, "available", False):
            lines.append(news.note or "No recent news.")
        else:
            bias = (
                "BULLISH" if news.sentiment > 0.05
                else "BEARISH" if news.sentiment < -0.05
                else "neutral"
            )
            lines.append(
                f"Net {news.sentiment:+.2f} ({bias})  | "
                f"{news.article_count} article(s)"
            )
            head = (news.latest_headline or "").strip()
            if head:
                when = (
                    news.latest_time.strftime("%m-%d %H:%M")
                    if news.latest_time else ""
                )
                lines.append(f"Latest ({news.latest_score:+.2f}) {when}:")
                lines.append(f"  {head[:90]}")
        self._news_text.configure(state=tk.NORMAL)
        self._news_text.delete("1.0", tk.END)
        self._news_text.insert(tk.END, "\n".join(lines))
        self._news_text.configure(state=tk.DISABLED)

    # ---- Charts ----------------------------------------------------------

    def _render_charts(self, analysis: Analysis) -> None:
        if not analysis.has_data:
            return
        df = analysis.frame
        x = df.index

        # Price + Bollinger + SMAs + trade markers.
        ax = self._ax_price
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.grid(True, color="#2a313c", linewidth=0.6)
        ax.plot(x, df["close"], color=FG, linewidth=1.3, label="Close")
        if "sma_fast" in df:
            ax.plot(x, df["sma_fast"], color=BLUE, linewidth=1.0, label="SMA 10")
        if "sma_slow" in df:
            ax.plot(x, df["sma_slow"], color=ORANGE, linewidth=1.0, label="SMA 30")
        if "bb_upper" in df and "bb_lower" in df:
            ax.plot(x, df["bb_upper"], color=GRAY, linewidth=0.8, linestyle="--")
            ax.plot(x, df["bb_lower"], color=GRAY, linewidth=0.8, linestyle="--")
            ax.fill_between(
                x, df["bb_lower"], df["bb_upper"], color=BLUE, alpha=0.07
            )
        self._plot_markers(ax, x)
        ax.set_title(
            f"{self._settings.ticker} price - 1-min bars",
            color=FG, fontsize=11,
        )
        ax.legend(
            loc="upper left", fontsize=7, facecolor=PANEL,
            edgecolor=GRAY, labelcolor=FG,
        )
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRAY)

        # Volatility: realized vol + Bollinger band width (twin axis).
        ax = self._ax_vol
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.grid(True, color="#2a313c", linewidth=0.6)
        if "realized_vol" in df:
            ax.plot(
                x, df["realized_vol"] * 100, color=GREEN, linewidth=1.1,
                label="Realized vol % (annualized)",
            )
        ax.set_ylabel("vol %", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRAY)
        ax.set_title(
            f"Volatility - regime: {analysis.vol_regime}", color=FG, fontsize=10
        )
        ax.legend(
            loc="upper left", fontsize=7, facecolor=PANEL,
            edgecolor=GRAY, labelcolor=FG,
        )

        # RSI.
        ax = self._ax_rsi
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.grid(True, color="#2a313c", linewidth=0.6)
        if "rsi" in df:
            ax.plot(x, df["rsi"], color=BLUE, linewidth=1.1, label="RSI 14")
        ax.axhline(70, color=RED, linewidth=0.8, linestyle="--")
        ax.axhline(30, color=GREEN, linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 100)
        ax.set_ylabel("RSI", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRAY)
        ax.legend(
            loc="upper left", fontsize=7, facecolor=PANEL,
            edgecolor=GRAY, labelcolor=FG,
        )

        # Volume.
        ax = self._ax_volume
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.grid(True, color="#2a313c", linewidth=0.6)
        if "volume" in df:
            ax.bar(x, df["volume"], color=GRAY, width=0.0006)
        ax.set_ylabel("vol", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRAY)

        self._fig.autofmt_xdate(rotation=0, ha="center")
        self._canvas.draw_idle()

    def _plot_markers(self, ax, x_index) -> None:
        if x_index is None or len(x_index) == 0:
            return
        x_min, x_max = x_index[0], x_index[-1]
        buys = [(t, p) for (t, p) in self._buy_markers if x_min <= t <= x_max]
        sells = [(t, p) for (t, p) in self._sell_markers if x_min <= t <= x_max]
        if buys:
            bx, by = zip(*buys)
            ax.scatter(
                bx, by, marker="^", color=GREEN, s=90, zorder=5,
                edgecolors="white", linewidths=0.6, label="Buy",
            )
        if sells:
            sx, sy = zip(*sells)
            ax.scatter(
                sx, sy, marker="v", color=RED, s=90, zorder=5,
                edgecolors="white", linewidths=0.6, label="Sell",
            )

    # ---- Logging / lifecycle --------------------------------------------

    def _log_line(self, text: str) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)
        self._log.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        # Reap any running hyperparameter-search subprocess.
        try:
            self._backtest.shutdown()
        except Exception:
            pass
        self._root.after(150, self._root.destroy)

    def run(self) -> None:
        self._log_line(
            f"Ready for {self._settings.ticker} (PAPER). Press Start to connect "
            f"to Alpaca and begin polling. Auto-trade is OFF until you enable it."
        )
        self._log_line(f"Network: {self._settings.proxy_description}.")
        if not self._settings.proxy_verify_ssl and not self._settings.proxy_ca_bundle:
            self._log_line(
                "WARNING: TLS verification is DISABLED. This is insecure; "
                "prefer setting PROXY_CA_BUNDLE to your corporate root CA."
            )
        self._root.mainloop()
