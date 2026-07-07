"""Backtest / simulation tab.

A self-contained Tkinter panel that lets the user:
  * download historical bars for the configured ticker,
  * edit **every** strategy + simulation parameter (generated dynamically from
    the dataclasses, so new knobs appear automatically),
  * set an artificial trade latency, and
  * run a simulation and view equity curve, trade markers and summary stats.

All network + compute work runs on background threads; results return via a
queue drained on the Tk event loop so the UI never freezes.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import tkinter as tk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .alpaca_client import TIMEFRAME_CHOICES, AlpacaClient
from .backtest import BacktestResult, run_backtest
from .config import Settings
from .optimize import (
    SearchResult,
    SearchSpec,
    TrialResult,
    default_search_space,
    search_process_entry,
)
from .params import SimConfig, StrategyParams
from .search_store import (
    list_searches,
    params_from_record,
    record_search,
    delete_search,
)
from .theme import BG, BLUE, FG, GRAY, GREEN, MUTED, ORANGE, PANEL, RED

# Parameter groups for a tidy editor. Any StrategyParams field not listed here
# is appended to an "Other" group so the editor always covers every knob.
_STRATEGY_GROUPS: List[Tuple[str, List[str]]] = [
    ("Indicators", ["sma_fast", "sma_slow", "bb_period", "bb_std",
                     "rsi_period", "atr_period", "vol_window"]),
    ("Volatility regime", ["vol_high_pctile", "vol_low_pctile",
                           "weight_high", "weight_normal", "weight_low"]),
    ("Signals & weights", ["rsi_oversold", "rsi_overbought", "rsi_weight",
                           "bb_low", "bb_high", "bb_weight", "trend_weight",
                           "zscore_period", "zscore_entry", "zscore_weight",
                           "fund_gap_threshold", "fund_gap_weight",
                           "fund_range_low", "fund_range_high",
                           "fund_range_weight"]),
    ("Trend filter", ["use_trend_filter", "trend_filter_period"]),
    ("Exits (ATR risk mgmt)", ["use_atr_exits", "stop_loss_atr",
                               "take_profit_atr", "trailing_stop_atr",
                               "max_hold_bars"]),
    ("Position sizing", ["use_vol_sizing", "risk_per_trade_pct",
                         "max_position_pct", "order_qty", "max_position_shares"]),
    ("News sentiment", ["use_news_sentiment", "news_lookback_minutes",
                        "news_half_life_minutes", "news_publish_lag_seconds",
                        "news_weight"]),
    ("Decision thresholds", ["buy_threshold", "sell_threshold"]),
]
_SIM_FIELDS = ["starting_cash", "latency_seconds", "slippage_bps",
               "commission_per_trade", "lookback_bars"]


class BacktestPanel:
    """Builds and drives the Backtest tab."""

    def __init__(
        self, parent: tk.Widget, client: AlpacaClient, settings: Settings
    ) -> None:
        self._parent = parent
        self._client = client
        self._settings = settings
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._df = None  # downloaded history (pandas DataFrame)
        self._news_df = None  # downloaded news (pandas DataFrame), optional
        self._result: Optional[BacktestResult] = None
        self._busy = False

        # tk vars keyed "S.<field>" (strategy) and "M.<field>" (sim).
        self._vars: Dict[str, tk.StringVar] = {}
        self._types: Dict[str, type] = {}
        # The research tool uses the full performance-oriented defaults (no hard
        # share cap; notional + ATR sizing). Live trading keeps its own cap.
        self._defaults_strategy = StrategyParams()
        self._defaults_sim = SimConfig()

        # Hyperparameter-search state.
        self._search_space = default_search_space()
        self._search_is_int = {s.name: s.is_int for s in self._search_space}
        self._search_enabled: Dict[str, tk.BooleanVar] = {}
        self._search_low: Dict[str, tk.StringVar] = {}
        self._search_high: Dict[str, tk.StringVar] = {}
        self._search_history: List[TrialResult] = []
        self._search_specs: List[SearchSpec] = []
        self._best_search_params: Optional[StrategyParams] = None
        self._search_stop: Optional[object] = None  # mp.Event while searching
        self._search_proc: Optional[mp.process.BaseProcess] = None
        self._search_mpq: Optional[object] = None  # mp.Queue while searching
        self._searching = False

        self._build()
        self._parent.after(200, self._poll)

    # ---- Layout ----------------------------------------------------------

    def _build(self) -> None:
        root = ttk.Frame(self._parent, style="TFrame", padding=(8, 8))
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root, width=460, style="TFrame")
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        self._build_data_section(left)
        self._build_params_section(left)
        self._build_run_section(left)

        right = ttk.Frame(root, style="TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self._build_results(right)

    def _build_data_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="1. Historical data", style="Header.TLabel").pack(
            anchor="w"
        )
        box = ttk.Frame(parent, style="TFrame")
        box.pack(fill=tk.X, pady=(4, 8))

        self._ticker_var = tk.StringVar(value=f"Ticker: {self._settings.ticker}")
        ttk.Label(box, textvariable=self._ticker_var,
                  style="Muted.TLabel").grid(row=0, column=0, sticky="w", columnspan=2)

        ttk.Label(box, text="Timeframe", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self._timeframe_var = tk.StringVar(value="1Min")
        ttk.Combobox(
            box, textvariable=self._timeframe_var, values=list(TIMEFRAME_CHOICES),
            state="readonly", width=10,
        ).grid(row=1, column=1, sticky="w", padx=4)

        today = datetime.now(timezone.utc).date()
        ttk.Label(box, text="Start (YYYY-MM-DD)", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self._start_var = tk.StringVar(value=str(today - timedelta(days=7)))
        tk.Entry(box, textvariable=self._start_var, width=14, bg=PANEL, fg=FG,
                 insertbackground=FG, relief=tk.FLAT).grid(
            row=2, column=1, sticky="w", padx=4
        )

        ttk.Label(box, text="End (YYYY-MM-DD)", style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", pady=2
        )
        self._end_var = tk.StringVar(value=str(today))
        tk.Entry(box, textvariable=self._end_var, width=14, bg=PANEL, fg=FG,
                 insertbackground=FG, relief=tk.FLAT).grid(
            row=3, column=1, sticky="w", padx=4
        )

        self._download_btn = tk.Button(
            box, text="Download data", command=self._on_download,
            bg=BLUE, fg="white", activebackground=GRAY, relief=tk.FLAT, padx=10,
        )
        self._download_btn.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 2))

        # Include Alpaca news (scored causally) in the simulation.
        self._news_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="Download + use news sentiment",
            variable=self._news_enabled,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self._data_status = tk.StringVar(value="No data downloaded yet.")
        ttk.Label(parent, textvariable=self._data_status, style="Muted.TLabel",
                  wraplength=440, justify="left").pack(anchor="w")
        self._news_status = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._news_status, style="Muted.TLabel",
                  wraplength=440, justify="left").pack(anchor="w")

    def _build_params_section(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="TFrame")
        header.pack(fill=tk.X, pady=(10, 2))
        ttk.Label(header, text="2. Tune parameters", style="Header.TLabel").pack(
            side=tk.LEFT
        )
        tk.Button(header, text="Reset", command=self._reset_params, bg=PANEL,
                  fg=FG, activebackground=GRAY, relief=tk.FLAT, padx=8).pack(
            side=tk.RIGHT
        )

        # Scrollable area for the (many) parameter fields.
        wrap = ttk.Frame(parent, style="Panel.TFrame")
        wrap.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(wrap, bg=PANEL, highlightthickness=0, bd=0, height=260)
        bar = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bar.pack(side=tk.RIGHT, fill=tk.Y)

        # Build groups for strategy params (covering every field).
        covered = {name for _, names in _STRATEGY_GROUPS for name in names}
        leftover = [f.name for f in fields(StrategyParams) if f.name not in covered]
        groups = list(_STRATEGY_GROUPS)
        if leftover:
            groups.append(("Other", leftover))

        for title, names in groups:
            self._add_group_header(inner, title)
            for name in names:
                self._add_param_row(
                    inner, "S", name, getattr(self._defaults_strategy, name)
                )

        self._add_group_header(inner, "Simulation (incl. latency)")
        for name in _SIM_FIELDS:
            self._add_param_row(inner, "M", name, getattr(self._defaults_sim, name))

    def _build_run_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="3. Run simulation", style="Header.TLabel").pack(
            anchor="w", pady=(10, 2)
        )
        self._run_btn = tk.Button(
            parent, text="Run simulation", command=self._on_run,
            bg=GREEN, fg="white", activebackground=GRAY, relief=tk.FLAT, padx=12,
        )
        self._run_btn.pack(anchor="w")
        self._progress = ttk.Progressbar(parent, mode="determinate", length=440)
        self._progress.pack(fill=tk.X, pady=(6, 2))
        self._run_status = tk.StringVar(value="Idle.")
        ttk.Label(parent, textvariable=self._run_status, style="Muted.TLabel",
                  wraplength=440, justify="left").pack(anchor="w")

    def _build_results(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        single = ttk.Frame(notebook, style="TFrame")
        notebook.add(single, text="  Single run  ")
        self._build_single_run(single)

        search = ttk.Frame(notebook, style="TFrame")
        notebook.add(search, text="  Hyperparameter search  ")
        self._build_search_tab(search)

        history = ttk.Frame(notebook, style="TFrame")
        notebook.add(history, text="  Search history  ")
        self._build_history_tab(history)

    def _build_single_run(self, parent: ttk.Frame) -> None:
        self._fig = Figure(figsize=(8, 7), dpi=100, facecolor=BG)
        self._fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.06,
                                  hspace=0.45)
        gs = self._fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.4])
        self._ax_price = self._fig.add_subplot(gs[0])
        self._ax_equity = self._fig.add_subplot(gs[1])
        self._ax_dd = self._fig.add_subplot(gs[2])
        for ax in (self._ax_price, self._ax_equity, self._ax_dd):
            self._style_axis(ax)
        self._ax_price.set_title("Run a simulation to see results",
                                 color=FG, fontsize=11)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._stats_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._stats_var, bg=PANEL, fg=FG,
                 font=("Consolas", 9), justify="left", anchor="w").pack(
            fill=tk.X, pady=(6, 0)
        )

    def _build_search_tab(self, parent: ttk.Frame) -> None:
        # Top: search controls. Bottom: param-bounds list (left) + charts (right).
        controls = ttk.Frame(parent, style="TFrame", padding=(2, 4))
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Optimize for profit", style="Header.TLabel").grid(
            row=0, column=0, columnspan=6, sticky="w"
        )

        ttk.Label(controls, text="Trials", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 2)
        )
        self._search_trials_var = tk.StringVar(value="150")
        tk.Entry(controls, textvariable=self._search_trials_var, width=7, bg=PANEL,
                 fg=FG, insertbackground=FG, relief=tk.FLAT).grid(row=1, column=1)

        ttk.Label(controls, text="Method", style="Muted.TLabel").grid(
            row=1, column=2, sticky="w", padx=(10, 2)
        )
        self._search_method_var = tk.StringVar(value="guided")
        ttk.Combobox(controls, textvariable=self._search_method_var,
                     values=["guided", "random"], state="readonly", width=8).grid(
            row=1, column=3
        )

        ttk.Label(controls, text="Seed", style="Muted.TLabel").grid(
            row=1, column=4, sticky="w", padx=(10, 2)
        )
        self._search_seed_var = tk.StringVar(value="42")
        tk.Entry(controls, textvariable=self._search_seed_var, width=7, bg=PANEL,
                 fg=FG, insertbackground=FG, relief=tk.FLAT).grid(row=1, column=5)

        # Objective + out-of-sample split (guards against overfitting).
        ttk.Label(controls, text="Objective", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 2), pady=(4, 0)
        )
        self._search_objective_var = tk.StringVar(value="sharpe_calmar")
        ttk.Combobox(controls, textvariable=self._search_objective_var,
                     values=["return", "sharpe", "calmar", "sharpe_calmar"],
                     state="readonly", width=12).grid(row=2, column=1, pady=(4, 0))
        ttk.Label(controls, text="OOS test %", style="Muted.TLabel").grid(
            row=2, column=2, sticky="w", padx=(10, 2), pady=(4, 0)
        )
        self._search_oos_var = tk.StringVar(value="30")
        tk.Entry(controls, textvariable=self._search_oos_var, width=7, bg=PANEL,
                 fg=FG, insertbackground=FG, relief=tk.FLAT).grid(
            row=2, column=3, pady=(4, 0)
        )

        # Activity guard (anti-degeneracy) + dual-objective Calmar weight.
        ttk.Label(controls, text="Min trades", style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 2), pady=(4, 0)
        )
        self._search_min_trades_var = tk.StringVar(value="20")
        tk.Entry(controls, textvariable=self._search_min_trades_var, width=7,
                 bg=PANEL, fg=FG, insertbackground=FG, relief=tk.FLAT).grid(
            row=3, column=1, pady=(4, 0)
        )
        ttk.Label(controls, text="Min expo %", style="Muted.TLabel").grid(
            row=3, column=2, sticky="w", padx=(10, 2), pady=(4, 0)
        )
        self._search_min_expo_var = tk.StringVar(value="5")
        tk.Entry(controls, textvariable=self._search_min_expo_var, width=7,
                 bg=PANEL, fg=FG, insertbackground=FG, relief=tk.FLAT).grid(
            row=3, column=3, pady=(4, 0)
        )
        ttk.Label(controls, text="Calmar wt", style="Muted.TLabel").grid(
            row=3, column=4, sticky="w", padx=(10, 2), pady=(4, 0)
        )
        self._search_calmar_wt_var = tk.StringVar(value="0.5")
        tk.Entry(controls, textvariable=self._search_calmar_wt_var, width=7,
                 bg=PANEL, fg=FG, insertbackground=FG, relief=tk.FLAT).grid(
            row=3, column=5, pady=(4, 0)
        )
        ttk.Label(
            controls,
            text="Anti-degeneracy guard: penalizes configs with < Min trades or "
                 "< Min expo % (in-market time). Calmar wt blends Sharpe+Calmar "
                 "for the 'sharpe_calmar' objective (0=Sharpe, 1=Calmar).",
            style="Muted.TLabel", wraplength=900, justify="left",
        ).grid(row=4, column=0, columnspan=6, sticky="w")

        # Early stopping: patience (no-improvement window), min improvement
        # (noise threshold), and an optional target objective.
        ttk.Label(controls, text="Patience", style="Muted.TLabel").grid(
            row=5, column=0, sticky="w", padx=(0, 2), pady=(4, 0)
        )
        self._search_patience_var = tk.StringVar(value="40")
        tk.Entry(controls, textvariable=self._search_patience_var, width=7,
                 bg=PANEL, fg=FG, insertbackground=FG, relief=tk.FLAT).grid(
            row=5, column=1, pady=(4, 0)
        )
        ttk.Label(controls, text="Min improve", style="Muted.TLabel").grid(
            row=5, column=2, sticky="w", padx=(10, 2), pady=(4, 0)
        )
        self._search_min_improve_var = tk.StringVar(value="0")
        tk.Entry(controls, textvariable=self._search_min_improve_var, width=7,
                 bg=PANEL, fg=FG, insertbackground=FG, relief=tk.FLAT).grid(
            row=5, column=3, pady=(4, 0)
        )
        ttk.Label(controls, text="Target", style="Muted.TLabel").grid(
            row=5, column=4, sticky="w", padx=(10, 2), pady=(4, 0)
        )
        self._search_target_var = tk.StringVar(value="")
        tk.Entry(controls, textvariable=self._search_target_var, width=7,
                 bg=PANEL, fg=FG, insertbackground=FG, relief=tk.FLAT).grid(
            row=5, column=5, pady=(4, 0)
        )
        ttk.Label(
            controls,
            text="Early stop: patience = trials w/o improvement (0 = off); "
                 "target = stop when objective reaches it (blank = off).",
            style="Muted.TLabel",
        ).grid(row=6, column=0, columnspan=6, sticky="w")

        self._search_start_btn = tk.Button(
            controls, text="Start search", command=self._on_start_search,
            bg=GREEN, fg="white", activebackground=GRAY, relief=tk.FLAT, padx=10,
        )
        self._search_start_btn.grid(row=7, column=0, columnspan=2, sticky="w",
                                    pady=(6, 2))
        self._search_stop_btn = tk.Button(
            controls, text="Stop", command=self._on_stop_search, bg=RED,
            fg="white", activebackground=GRAY, relief=tk.FLAT, padx=10,
            state=tk.DISABLED,
        )
        self._search_stop_btn.grid(row=7, column=2, sticky="w", pady=(6, 2))
        self._apply_best_btn = tk.Button(
            controls, text="Apply best to editor", command=self._apply_best_to_editor,
            bg=BLUE, fg="white", activebackground=GRAY, relief=tk.FLAT, padx=10,
            state=tk.DISABLED,
        )
        self._apply_best_btn.grid(row=7, column=3, columnspan=3, sticky="w",
                                  pady=(6, 2))

        self._search_progress = ttk.Progressbar(controls, mode="determinate",
                                                 length=520)
        self._search_progress.grid(row=8, column=0, columnspan=6, sticky="we",
                                   pady=(4, 2))
        self._search_best_var = tk.StringVar(value="No search run yet.")
        ttk.Label(controls, textvariable=self._search_best_var, style="TLabel").grid(
            row=9, column=0, columnspan=6, sticky="w"
        )
        self._search_status = tk.StringVar(value="Select parameters, then Start search.")
        ttk.Label(controls, textvariable=self._search_status, style="Muted.TLabel",
                  wraplength=900, justify="left").grid(
            row=10, column=0, columnspan=6, sticky="w"
        )

        body = ttk.Frame(parent, style="TFrame")
        body.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # Left: which parameters to search + their bounds.
        bounds_wrap = ttk.Frame(body, width=320, style="Panel.TFrame")
        bounds_wrap.pack(side=tk.LEFT, fill=tk.Y)
        bounds_wrap.pack_propagate(False)
        tk.Label(bounds_wrap, text="Search space (enable + bounds)", bg=PANEL,
                 fg=BLUE, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=6,
                                                             pady=(4, 2))
        canvas = tk.Canvas(bounds_wrap, bg=PANEL, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(bounds_wrap, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        hdr = tk.Frame(inner, bg=PANEL)
        hdr.pack(fill=tk.X, padx=4)
        tk.Label(hdr, text="", bg=PANEL, width=2).pack(side=tk.LEFT)
        tk.Label(hdr, text="param", bg=PANEL, fg=MUTED, width=16, anchor="w",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Label(hdr, text="min", bg=PANEL, fg=MUTED, width=7,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Label(hdr, text="max", bg=PANEL, fg=MUTED, width=7,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        for spec in self._search_space:
            self._add_search_row(inner, spec)

        # Enable mouse-wheel scrolling over the whole search-space subtree.
        self._bind_mousewheel(canvas, inner)

        # Right: live search charts.
        chart_wrap = ttk.Frame(body, style="TFrame")
        chart_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self._search_fig = Figure(figsize=(7, 6), dpi=100, facecolor=BG)
        self._search_fig.subplots_adjust(left=0.1, right=0.97, top=0.94,
                                         bottom=0.08, hspace=0.35)
        sgs = self._search_fig.add_gridspec(2, 1, height_ratios=[3, 2])
        self._ax_traj = self._search_fig.add_subplot(sgs[0])
        self._ax_obj = self._search_fig.add_subplot(sgs[1])
        for ax in (self._ax_traj, self._ax_obj):
            self._style_axis(ax)
        self._ax_traj.set_title("Run a search to see parameter evolution",
                                color=FG, fontsize=10)
        self._search_canvas = FigureCanvasTkAgg(self._search_fig, master=chart_wrap)
        self._search_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---- Search history tab ---------------------------------------------

    def _build_history_tab(self, parent: ttk.Frame) -> None:
        """List every stored search and let the user load its parameters."""
        header = ttk.Frame(parent, style="TFrame", padding=(2, 4))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Saved searches", style="Header.TLabel").pack(
            side=tk.LEFT
        )
        tk.Button(header, text="Refresh", command=self._refresh_history,
                  bg=PANEL, fg=FG, activebackground=GRAY, relief=tk.FLAT,
                  padx=10).pack(side=tk.RIGHT, padx=4)
        tk.Button(header, text="Delete", command=self._on_delete_history,
                  bg=RED, fg="white", activebackground=GRAY, relief=tk.FLAT,
                  padx=10).pack(side=tk.RIGHT, padx=4)
        tk.Button(header, text="Load into editor",
                  command=self._on_load_history, bg=GREEN, fg="white",
                  activebackground=GRAY, relief=tk.FLAT, padx=10).pack(
            side=tk.RIGHT, padx=4
        )

        # Body: list (left) + trajectory chart (right).
        body = ttk.Frame(parent, style="TFrame")
        body.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        list_wrap = ttk.Frame(body, width=360, style="TFrame")
        list_wrap.pack(side=tk.LEFT, fill=tk.Y)
        list_wrap.pack_propagate(False)
        self._history_list = tk.Listbox(
            list_wrap, bg=PANEL, fg=FG, selectbackground=BLUE,
            selectforeground="white", relief=tk.FLAT, font=("Consolas", 9),
            activestyle="none", highlightthickness=0,
        )
        hsb = ttk.Scrollbar(list_wrap, orient="vertical",
                            command=self._history_list.yview)
        self._history_list.configure(yscrollcommand=hsb.set)
        self._history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._history_list.bind("<<ListboxSelect>>", self._on_history_select)
        self._history_list.bind("<Double-Button-1>",
                                lambda e: self._on_load_history())

        # Trajectory chart: objective score per trial + best-so-far, with
        # markers where the best parameters improved.
        chart_wrap = ttk.Frame(body, style="TFrame")
        chart_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self._hist_fig = Figure(figsize=(6, 3.2), dpi=100, facecolor=BG)
        self._hist_fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.16)
        self._ax_hist = self._hist_fig.add_subplot(111)
        self._style_axis(self._ax_hist)
        self._ax_hist.set_title("Select a saved search to see its score history",
                                color=FG, fontsize=10)
        self._hist_canvas = FigureCanvasTkAgg(self._hist_fig, master=chart_wrap)
        self._hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Details of the selected record.
        self._history_detail = tk.Text(
            parent, height=8, bg=PANEL, fg=FG, relief=tk.FLAT,
            font=("Consolas", 9), wrap=tk.WORD,
        )
        self._history_detail.pack(fill=tk.X, pady=(6, 0))
        self._history_detail.configure(state=tk.DISABLED)

        self._history_status = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._history_status, style="Muted.TLabel",
                  wraplength=620, justify="left").pack(anchor="w", pady=(4, 0))

        self._history_records: List = []
        self._refresh_history()

    def _refresh_history(self) -> None:
        try:
            self._history_records = list_searches()
        except Exception as exc:
            self._history_records = []
            self._history_status.set(f"Could not read history: {exc}")
        self._history_list.delete(0, tk.END)
        for rec in self._history_records:
            self._history_list.insert(tk.END, rec.label)
        if not self._history_records:
            self._history_status.set(
                "No saved searches yet. Run a hyperparameter search; results are "
                "stored automatically."
            )
        else:
            self._history_status.set(
                f"{len(self._history_records)} saved search(es). Select one, then "
                f"'Load into editor'."
            )

    def _selected_record(self):
        sel = self._history_list.curselection()
        if not sel or sel[0] >= len(self._history_records):
            return None
        return self._history_records[sel[0]]

    def _on_history_select(self, _event=None) -> None:
        rec = self._selected_record()
        if rec is None:
            return
        lines = [
            f"Saved : {rec.timestamp.replace('T', ' ')}",
            f"Ticker: {rec.ticker}   Timeframe: {rec.timeframe or 'n/a'}",
            f"Objective: {rec.objective}   ({rec.stop_reason}, {rec.trials} trials)",
            f"In-sample : {self._fmt_objective(rec.objective, rec.best_objective)}"
            f"   return {rec.best_return * 100:+.2f}%",
        ]
        if rec.oos_objective is not None:
            lines.append(
                f"Out-of-sample: {self._fmt_objective(rec.objective, rec.oos_objective)}"
                f"   return {(rec.oos_return or 0.0) * 100:+.2f}%"
            )
        # A few headline params.
        p = rec.params
        lines.append(
            "Params: "
            + ", ".join(
                f"{k}={p[k]}" for k in (
                    "buy_threshold", "sell_threshold", "stop_loss_atr",
                    "take_profit_atr", "max_hold_bars", "risk_per_trade_pct",
                ) if k in p
            )
        )
        # Best-parameter evolution log ("best parameters per run").
        evo = rec.param_evolution or []
        if evo:
            lines.append(f"Best-params improvements ({len(evo)}):")
            for e in evo[-6:]:  # last few improvements
                ev_p = e.get("params", {})
                key_bits = ", ".join(
                    f"{k}={ev_p[k]}" for k in ("buy_threshold", "stop_loss_atr",
                                               "rsi_oversold") if k in ev_p
                )
                lines.append(
                    f"  trial {e.get('iteration')}: "
                    f"{self._fmt_objective(rec.objective, e.get('best_objective'))}"
                    f"  [{key_bits}]"
                )
        self._history_detail.configure(state=tk.NORMAL)
        self._history_detail.delete("1.0", tk.END)
        self._history_detail.insert(tk.END, "\n".join(lines))
        self._history_detail.configure(state=tk.DISABLED)

        self._render_history_trajectory(rec)

    def _render_history_trajectory(self, rec) -> None:
        """Plot the saved per-trial objective + best-so-far, with improvements."""
        ax = self._ax_hist
        ax.clear()
        self._style_axis(ax)
        hist = rec.objective_history or []
        if not hist:
            ax.set_title("No score history stored for this run", color=FG, fontsize=10)
            self._hist_canvas.draw_idle()
            return

        name = rec.objective
        scale = 100.0 if name == "return" else 1.0
        iters = [h.get("iteration", i + 1) for i, h in enumerate(hist)]
        obj = [
            (h.get("objective") * scale) if h.get("objective") is not None else np.nan
            for h in hist
        ]
        best = [
            (h.get("best_objective") * scale) if h.get("best_objective") is not None else np.nan
            for h in hist
        ]
        ax.scatter(iters, obj, s=12, color=MUTED, alpha=0.5, label=f"Trial {name}")
        ax.plot(iters, best, color=GREEN, linewidth=1.6, label=f"Best {name}")
        # Mark where the best parameters improved.
        for e in (rec.param_evolution or []):
            it = e.get("iteration")
            bo = e.get("best_objective")
            if it is not None and bo is not None:
                ax.scatter([it], [bo * scale], marker="^", color=BLUE, s=55,
                           zorder=5, edgecolors="white", linewidths=0.5)
        ax.set_xlabel("trial", color=MUTED, fontsize=8)
        ax.set_ylabel(name, color=MUTED, fontsize=8)
        ax.set_title(
            f"Score history -- {rec.ticker} ({name}); ^ = best-params improvement",
            color=FG, fontsize=9,
        )
        ax.legend(loc="lower right", fontsize=7, facecolor=PANEL,
                  edgecolor=GRAY, labelcolor=FG)
        self._hist_canvas.draw_idle()

    def _on_load_history(self) -> None:
        rec = self._selected_record()
        if rec is None:
            self._history_status.set("Select a saved search first.")
            return
        try:
            params = params_from_record(rec)
        except Exception as exc:
            self._history_status.set(f"Could not load params: {exc}")
            return
        self.set_params_into_editor(params)
        self._best_search_params = params  # so 'Apply' / Live import can reuse
        self._history_status.set(
            f"Loaded params from {rec.timestamp.replace('T', ' ')[:19]} into the "
            f"editor (tab 2). Run a simulation, or 'Import from Backtest' on the "
            f"Live tab."
        )

    def _on_delete_history(self) -> None:
        rec = self._selected_record()
        if rec is None:
            self._history_status.set("Select a saved search to delete.")
            return
        if delete_search(rec.path):
            self._history_status.set(f"Deleted {rec.path.name}.")
            self._refresh_history()
        else:
            self._history_status.set(f"Could not delete {rec.path.name}.")

    def _bind_mousewheel(self, canvas: tk.Canvas, *subtrees: tk.Widget) -> None:
        """Scroll ``canvas`` with the mouse wheel while the cursor is over it.

        Binds directly to the canvas and every descendant widget (recursively),
        so the wheel works even when hovering over rows/entries -- and without a
        global ``bind_all`` that would fight the other scroll areas in the app.
        """
        def _on_wheel(event: "tk.Event") -> str:
            num = getattr(event, "num", 0)
            if num == 4:                       # Linux scroll up
                canvas.yview_scroll(-1, "units")
            elif num == 5:                     # Linux scroll down
                canvas.yview_scroll(1, "units")
            else:                              # Windows / macOS
                canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

        def _bind(widget: tk.Widget) -> None:
            widget.bind("<MouseWheel>", _on_wheel)
            widget.bind("<Button-4>", _on_wheel)
            widget.bind("<Button-5>", _on_wheel)
            for child in widget.winfo_children():
                _bind(child)

        _bind(canvas)
        for subtree in subtrees:
            _bind(subtree)

    def _add_search_row(self, parent: ttk.Frame, spec: SearchSpec) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill=tk.X, padx=4, pady=1)
        en = tk.BooleanVar(value=spec.enabled)
        self._search_enabled[spec.name] = en
        tk.Checkbutton(row, variable=en, bg=PANEL, activebackground=PANEL,
                       selectcolor=GRAY, bd=0, highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(row, text=spec.name, bg=PANEL, fg=FG, width=16, anchor="w",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        low = tk.StringVar(value=self._format_value(spec.low))
        high = tk.StringVar(value=self._format_value(spec.high))
        self._search_low[spec.name] = low
        self._search_high[spec.name] = high
        tk.Entry(row, textvariable=low, width=7, bg=BG, fg=FG,
                 insertbackground=FG, relief=tk.FLAT).pack(side=tk.LEFT, padx=1)
        tk.Entry(row, textvariable=high, width=7, bg=BG, fg=FG,
                 insertbackground=FG, relief=tk.FLAT).pack(side=tk.LEFT, padx=1)

    # ---- Parameter widgets ----------------------------------------------

    def _add_group_header(self, parent: ttk.Frame, title: str) -> None:
        tk.Label(parent, text=title, bg=PANEL, fg=BLUE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=6, pady=(8, 2))

    def _add_param_row(self, parent: ttk.Frame, scope: str, name: str, value) -> None:
        key = f"{scope}.{name}"
        self._types[key] = type(value)
        var = tk.StringVar(value=self._format_value(value))
        self._vars[key] = var

        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(row, text=name, bg=PANEL, fg=FG, font=("Segoe UI", 9),
                 width=22, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var, width=12, bg=BG, fg=FG,
                 insertbackground=FG, relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _reset_params(self) -> None:
        for key, var in self._vars.items():
            scope, name = key.split(".", 1)
            default = (
                getattr(self._defaults_strategy, name)
                if scope == "S"
                else getattr(self._defaults_sim, name)
            )
            var.set(self._format_value(default))
        self._run_status.set("Parameters reset to defaults.")

    def _collect_params(self) -> Tuple[StrategyParams, SimConfig]:
        """Read all entry widgets into fresh dataclasses (raises on bad input)."""
        s_kwargs: Dict[str, object] = {}
        m_kwargs: Dict[str, object] = {}
        for key, var in self._vars.items():
            scope, name = key.split(".", 1)
            typ = self._types[key]
            raw = var.get().strip()
            try:
                if typ is bool:
                    value = raw.lower() in {"1", "true", "yes", "on", "y", "t"}
                elif typ is int:
                    value = int(round(float(raw)))
                else:
                    value = float(raw)
            except ValueError as exc:
                raise ValueError(f"'{name}' = '{raw}' is not a valid number") from exc
            (s_kwargs if scope == "S" else m_kwargs)[name] = value

        params = StrategyParams(**s_kwargs)
        sim = SimConfig(**m_kwargs)
        # Derive seconds-per-bar from the selected timeframe.
        _, bar_seconds = TIMEFRAME_CHOICES[self._timeframe_var.get()]
        sim.bar_seconds = bar_seconds
        return params, sim

    # ---- Actions ---------------------------------------------------------

    def _on_download(self) -> None:
        if self._busy:
            return
        try:
            start = datetime.strptime(self._start_var.get().strip(), "%Y-%m-%d")
            end = datetime.strptime(self._end_var.get().strip(), "%Y-%m-%d")
        except ValueError:
            self._data_status.set("Dates must be YYYY-MM-DD.")
            return
        if end < start:
            self._data_status.set("End date must be on or after start date.")
            return

        start = start.replace(tzinfo=timezone.utc)
        # Include the whole end day.
        end = end.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        timeframe, _ = TIMEFRAME_CHOICES[self._timeframe_var.get()]
        label = self._timeframe_var.get()

        self._busy = True
        self._download_btn.configure(state=tk.DISABLED, text="Downloading...")
        self._data_status.set(f"Downloading {label} bars...")
        want_news = bool(self._news_enabled.get())
        self._news_status.set("Downloading news..." if want_news else "")

        def work() -> None:
            try:
                df = self._client.download_history(start, end, timeframe)
                news_df = None
                news_note = ""
                if want_news:
                    try:
                        news_df = self._client.get_news(start, end)
                        news_note = (
                            f"Loaded {len(news_df):,} news article(s) for the range."
                            if news_df is not None and len(news_df) > 0
                            else "No news returned for this range (or no entitlement)."
                        )
                    except Exception as exc:
                        news_note = f"News unavailable: {type(exc).__name__}: {exc}"
                self._queue.put(("data", df, label, news_df, news_note))
            except Exception as exc:  # network / auth / rate limit
                self._queue.put(("error", f"Download failed: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _on_run(self) -> None:
        if self._busy:
            return
        if self._df is None or len(self._df) == 0:
            self._run_status.set("Download data before running a simulation.")
            return
        try:
            params, sim = self._collect_params()
        except ValueError as exc:
            self._run_status.set(f"Parameter error: {exc}")
            return

        df = self._df
        news_df = self._news_df if self._news_enabled.get() else None
        self._busy = True
        self._run_btn.configure(state=tk.DISABLED, text="Running...")
        self._progress.configure(value=0, maximum=len(df))
        news_txt = (
            f" + {len(news_df):,} news" if news_df is not None and len(news_df) else ""
        )
        self._run_status.set(
            f"Simulating {len(df)} bars{news_txt} | latency {sim.latency_seconds:g}s "
            f"(~{sim.latency_bars} bar(s))..."
        )

        def work() -> None:
            try:
                def prog(done: int, total: int) -> None:
                    self._queue.put(("progress", done, total))

                result = run_backtest(df, params, sim, progress=prog,
                                      news_df=news_df)
                self._queue.put(("result", result))
            except Exception as exc:
                self._queue.put(("error", f"Simulation failed: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    # ---- Queue draining --------------------------------------------------

    def _poll(self) -> None:
        # Trial messages are coalesced: a fast search can enqueue many per
        # cycle, but we update widgets / redraw the chart only ONCE here, so
        # the Tk main thread never blocks on a pile of synchronous draws.
        trials: List[TrialResult] = []

        # 1) Thread queue: download + single-run results, errors.
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "data":
                    self._on_data_ready(msg[1], msg[2], msg[3], msg[4])
                elif kind == "progress":
                    self._progress.configure(value=msg[1], maximum=msg[2])
                elif kind == "result":
                    self._on_result_ready(msg[1])
                elif kind == "error":
                    self._on_error(msg[1])
        except queue.Empty:
            pass

        # 2) Process queue: the hyperparameter search (runs in a child process
        #    so its CPU load can't freeze the GUI). Drain trials + terminal msg.
        search_done: Optional[tuple] = None
        if self._search_mpq is not None:
            try:
                while True:
                    msg = self._search_mpq.get_nowait()
                    kind = msg[0]
                    if kind == "trial":
                        trials.append(msg[1])
                    elif kind == "search_done":
                        search_done = ("done", msg[1])
                    elif kind == "search_error":
                        search_done = ("error", msg[1])
            except queue.Empty:
                pass
            except (OSError, ValueError):
                # Queue closed during teardown -- ignore.
                pass

        if trials:
            self._absorb_trials(trials)

        if search_done is not None:
            if search_done[0] == "done":
                self._on_search_done(search_done[1])
            else:
                self._on_search_error(search_done[1])
            self._cleanup_search_proc()

        self._parent.after(150, self._poll)

    def _on_data_ready(self, df, label: str, news_df=None, news_note="") -> None:
        self._busy = False
        self._download_btn.configure(state=tk.NORMAL, text="Download data")
        self._df = df
        self._news_df = news_df
        self._news_status.set(news_note or "")
        if df is None or len(df) == 0:
            self._data_status.set(
                "No bars returned for that range/timeframe (markets closed or "
                "no IEX data). Try a wider range."
            )
            return
        first = df.index[0].strftime("%Y-%m-%d %H:%M")
        last = df.index[-1].strftime("%Y-%m-%d %H:%M")
        self._data_status.set(
            f"Loaded {len(df):,} {label} bars: {first} -> {last} UTC."
        )

    def _on_error(self, message: str) -> None:
        self._busy = False
        self._download_btn.configure(state=tk.NORMAL, text="Download data")
        self._run_btn.configure(state=tk.NORMAL, text="Run simulation")
        self._run_status.set(message)
        self._data_status.set(message)

    def _on_result_ready(self, result: BacktestResult) -> None:
        self._busy = False
        self._result = result
        self._run_btn.configure(state=tk.NORMAL, text="Run simulation")
        s = result.stats
        self._run_status.set(
            f"Done. {s['num_trades']} fills, {s['num_round_trips']} round-trips."
        )
        self._stats_var.set(self._format_stats(result))
        self._render_charts(result)

    @staticmethod
    def _format_stats(result: BacktestResult) -> str:
        s = result.stats
        sim = result.sim
        return (
            f"Start ${sim.starting_cash:,.0f} -> End ${s['final_equity']:,.2f}   "
            f"Return {s['total_return'] * 100:+.2f}%   "
            f"(Buy&Hold {s['buy_hold_return'] * 100:+.2f}%)\n"
            f"Trades {s['num_trades']}   Round-trips {s['num_round_trips']}   "
            f"Win rate {s['win_rate'] * 100:.1f}%   "
            f"Realized P/L ${s['realized_pl']:,.2f}\n"
            f"Max drawdown {s['max_drawdown'] * 100:.2f}%   "
            f"Sharpe {s['sharpe']:.2f}   "
            f"Exposure {s['exposure'] * 100:.1f}%   "
            f"Latency ~{sim.latency_bars} bar(s)"
        )

    # ---- Hyperparameter search ------------------------------------------

    def _collect_specs(self) -> List[SearchSpec]:
        """Build the enabled search specs from the bounds widgets."""
        specs: List[SearchSpec] = []
        for spec in self._search_space:
            if not self._search_enabled[spec.name].get():
                continue
            try:
                low = float(self._search_low[spec.name].get())
                high = float(self._search_high[spec.name].get())
            except ValueError as exc:
                raise ValueError(f"Bad bounds for '{spec.name}'.") from exc
            if high < low:
                raise ValueError(f"'{spec.name}': max must be >= min.")
            specs.append(
                SearchSpec(spec.name, low, high, self._search_is_int[spec.name], True)
            )
        if not specs:
            raise ValueError("Enable at least one parameter to search.")
        return specs

    def _on_start_search(self) -> None:
        if self._busy or self._searching:
            return
        if self._df is None or len(self._df) == 0:
            self._search_status.set("Download data before searching.")
            return
        try:
            base, sim = self._collect_params()
            specs = self._collect_specs()
            n_trials = int(float(self._search_trials_var.get()))
            seed = int(float(self._search_seed_var.get()))
            oos = max(0.0, min(0.8, float(self._search_oos_var.get()) / 100.0))
            patience = max(0, int(float(self._search_patience_var.get() or 0)))
            min_improve = max(0.0, float(self._search_min_improve_var.get() or 0))
            target_raw = self._search_target_var.get().strip()
            target = float(target_raw) if target_raw else None
            min_trades = max(0, int(float(self._search_min_trades_var.get() or 0)))
            min_exposure = max(
                0.0, min(1.0, float(self._search_min_expo_var.get() or 0) / 100.0)
            )
            calmar_wt = max(0.0, min(1.0, float(self._search_calmar_wt_var.get() or 0.5)))
        except ValueError as exc:
            self._search_status.set(f"Cannot start: {exc}")
            return
        if n_trials < 1:
            self._search_status.set("Trials must be >= 1.")
            return

        method = self._search_method_var.get()
        objective = self._search_objective_var.get()
        self._active_objective = objective
        self._search_specs = specs
        self._search_history = []
        self._best_search_params = None
        self._searching = True
        self._search_start_btn.configure(state=tk.DISABLED)
        self._search_stop_btn.configure(state=tk.NORMAL)
        self._apply_best_btn.configure(state=tk.DISABLED)
        self._search_progress.configure(value=0, maximum=n_trials)
        self._search_best_var.set("Searching...")
        early = []
        if patience > 0:
            early.append(f"patience {patience}")
        if target is not None:
            early.append(f"target {target:g}")
        early_txt = f", early stop: {', '.join(early)}" if early else ""
        guard = []
        if min_trades > 0:
            guard.append(f"min {min_trades} trades")
        if min_exposure > 0:
            guard.append(f"min {min_exposure * 100:g}% expo")
        guard_txt = f", guard: {', '.join(guard)}" if guard else ""
        use_news = self._news_enabled.get() and self._news_df is not None \
            and len(self._news_df) > 0
        news_txt = f", {len(self._news_df):,} news" if use_news else ""
        self._search_status.set(
            f"Optimizing {len(specs)} parameter(s) for {objective} over "
            f"{len(self._df):,} bars{news_txt}, up to {n_trials} trials ({method}), "
            f"{int(oos * 100)}%% held out for out-of-sample test"
            f"{guard_txt}{early_txt}..."
        )

        # The search is CPU-bound; run it in a SEPARATE PROCESS so it can never
        # contend for the GIL with the GUI thread. Results stream back over a
        # multiprocessing queue that _poll() drains on the Tk event loop.
        kwargs = dict(
            df=self._df, base_params=base, sim=sim, specs=specs,
            n_trials=n_trials, method=method, seed=seed, objective=objective,
            oos_fraction=oos, patience=patience, min_improvement=min_improve,
            target_objective=target, min_trades=min_trades,
            min_exposure=min_exposure, calmar_weight=calmar_wt,
            news_df=self._news_df if use_news else None,
        )
        try:
            ctx = mp.get_context("spawn")
            self._search_mpq = ctx.Queue()
            self._search_stop = ctx.Event()
            self._search_proc = ctx.Process(
                target=search_process_entry,
                args=(self._search_mpq, self._search_stop, kwargs),
                daemon=True,
            )
            self._search_proc.start()
        except Exception as exc:
            self._searching = False
            self._search_start_btn.configure(state=tk.NORMAL)
            self._search_stop_btn.configure(state=tk.DISABLED)
            self._search_status.set(f"Could not start search process: {exc}")
            self._cleanup_search_proc()

    def _on_stop_search(self) -> None:
        if self._search_stop is not None:
            self._search_stop.set()
        self._search_status.set("Stopping search (finishing current trial)...")

    def _cleanup_search_proc(self) -> None:
        """Tear down the search process + queue after it finishes or is stopped."""
        proc = self._search_proc
        if proc is not None:
            try:
                proc.join(timeout=0.5)
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        if self._search_mpq is not None:
            try:
                self._search_mpq.close()
            except Exception:
                pass
        self._search_proc = None
        self._search_mpq = None
        self._search_stop = None

    def shutdown(self) -> None:
        """Stop and reap the search process (called on app close)."""
        if self._search_stop is not None:
            try:
                self._search_stop.set()
            except Exception:
                pass
        proc = self._search_proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    @staticmethod
    def _fmt_objective(name: str, value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        if name == "return":
            return f"{value * 100:+.2f}%"
        return f"{value:.2f}"

    def _absorb_trials(self, trials: List[TrialResult]) -> None:
        """Record a batch of trials and refresh the UI once (keeps GUI fluid).

        Only the most recent trial drives the readout/progress, and the chart
        is redrawn a single time per poll regardless of batch size.
        """
        self._search_history.extend(trials)
        last = trials[-1]
        self._best_search_params = last.best_params
        self._search_progress.configure(value=last.iteration)
        name = getattr(self, "_active_objective", "return")
        self._search_best_var.set(
            f"Best {name} {self._fmt_objective(name, last.best_objective)}  |  "
            f"equity ${last.final_equity:,.0f}  |  {last.num_trades} trades  |  "
            f"trial {last.iteration}  |  {last.trials_since_improvement} since improve"
        )
        self._render_search_charts()

    def _on_search_done(self, result: SearchResult) -> None:
        self._searching = False
        self._search_start_btn.configure(state=tk.NORMAL)
        self._search_stop_btn.configure(state=tk.DISABLED)
        self._best_search_params = result.best_params
        self._apply_best_btn.configure(state=tk.NORMAL)
        self._render_search_charts(force=True)

        # Persist every search so its parameters can be reloaded later.
        saved_note = ""
        try:
            path = record_search(
                result, ticker=self._settings.ticker,
                timeframe=self._timeframe_var.get(),
            )
            saved_note = f"  Saved to history ({path.name})."
            self._refresh_history()
        except Exception as exc:
            saved_note = f"  (Could not save to history: {exc})"

        reason_txt = {
            "all-trials": "completed all trials",
            "patience": "early-stopped (no improvement / patience reached)",
            "target": "early-stopped (target objective reached)",
            "stopped": "stopped by user",
        }.get(result.stop_reason, result.stop_reason)
        name = result.objective_name
        is_txt = self._fmt_objective(name, result.best_objective)
        is_ret = result.best_stats.get("total_return", 0.0) * 100
        msg = (
            f"Search {reason_txt}: in-sample {name} {is_txt} "
            f"(return {is_ret:+.2f}%, ${result.best_stats.get('final_equity', 0):,.0f}) "
            f"over {len(result.history)} trials."
        )
        if result.oos_objective is not None and result.oos_stats is not None:
            oos_txt = self._fmt_objective(name, result.oos_objective)
            oos_ret = result.oos_stats.get("total_return", 0.0) * 100
            msg += (
                f"  OUT-OF-SAMPLE {name} {oos_txt} (return {oos_ret:+.2f}%). "
                f"If OOS is far worse than in-sample, the config is overfit."
            )
        msg += "  Click 'Apply best to editor'."
        msg += saved_note
        self._search_status.set(msg)

    def _on_search_error(self, message: str) -> None:
        self._searching = False
        self._search_start_btn.configure(state=tk.NORMAL)
        self._search_stop_btn.configure(state=tk.DISABLED)
        self._search_status.set(message)

    def _apply_best_to_editor(self) -> None:
        if self._best_search_params is None:
            return
        self.set_params_into_editor(self._best_search_params)
        self._search_status.set(
            "Best parameters loaded into the editor (tab 2). Run a simulation, "
            "or use 'Import from Backtest' on the Live tab."
        )

    # ---- Public API (used by the Live tab / ticker selector) ------------

    def get_strategy_params(self) -> StrategyParams:
        """Return the strategy params currently in the editor (raises on bad input)."""
        params, _ = self._collect_params()
        return params

    def set_params_into_editor(self, params: StrategyParams) -> None:
        """Load a StrategyParams into the editable fields."""
        for f in fields(StrategyParams):
            key = f"S.{f.name}"
            if key in self._vars:
                self._vars[key].set(self._format_value(getattr(params, f.name)))

    def set_ticker(self, ticker: str) -> None:
        """Update the panel for a new symbol and clear stale downloaded data."""
        self._settings = replace(self._settings, ticker=ticker)
        self._ticker_var.set(f"Ticker: {ticker}")
        self._df = None
        self._news_df = None
        self._news_status.set("")
        self._data_status.set(
            f"Ticker changed to {ticker}. Download data to backtest this symbol."
        )

    # ---- Charts ----------------------------------------------------------

    def _style_axis(self, ax) -> None:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRAY)
        ax.grid(True, color="#2a313c", linewidth=0.6)

    def _render_search_charts(self, force: bool = False) -> None:
        hist = self._search_history
        if not hist or not self._search_specs:
            return
        iters = [t.iteration for t in hist]

        # Top: best-so-far value of each searched parameter, normalized to its
        # [low, high] range, so very differently-scaled knobs share one axis.
        ax = self._ax_traj
        ax.clear()
        self._style_axis(ax)
        for spec in self._search_specs:
            ys = [spec.normalize(getattr(t.best_params, spec.name)) for t in hist]
            ax.plot(iters, ys, linewidth=1.2, label=spec.name)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("normalized\nbest-so-far", color=MUTED, fontsize=8)
        ax.set_title("Parameter evolution during search (optimizing profit)",
                     color=FG, fontsize=10)
        ax.legend(loc="upper left", ncol=2, fontsize=6, facecolor=PANEL,
                  edgecolor=GRAY, labelcolor=FG)

        # Bottom: best-so-far profit curve + per-trial profit scatter.
        ax = self._ax_obj
        ax.clear()
        self._style_axis(ax)
        best = np.array([t.best_objective for t in hist], dtype=float)
        cand = np.array(
            [t.objective if np.isfinite(t.objective) else np.nan for t in hist],
            dtype=float,
        )
        name = getattr(self, "_active_objective", "return")
        scale = 100.0 if name == "return" else 1.0
        ax.scatter(iters, cand * scale, s=10, color=MUTED, alpha=0.5,
                   label=f"Trial {name}")
        ax.plot(iters, best * scale, color=GREEN, linewidth=1.4,
                label=f"Best {name}")
        finite = (cand * scale)[np.isfinite(cand)]
        if finite.size:
            lo = min(float(np.percentile(finite, 5)), float((best * scale).min()))
            hi = max(float(finite.max()), float((best * scale).max()))
            pad = max(0.5, (hi - lo) * 0.1)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(name, color=MUTED, fontsize=8)
        ax.set_xlabel("trial", color=MUTED, fontsize=8)
        ax.legend(loc="upper left", fontsize=7, facecolor=PANEL,
                  edgecolor=GRAY, labelcolor=FG)

        # Live updates use draw_idle() (non-blocking, coalesces repeated
        # requests) so frequent redraws never stall the UI; the final render
        # forces a synchronous draw() so the last frame is guaranteed painted.
        if force:
            self._search_canvas.draw()
        else:
            self._search_canvas.draw_idle()

    def _render_charts(self, result: BacktestResult) -> None:
        ind = result.indicators
        x = ind.index

        ax = self._ax_price
        ax.clear()
        self._style_axis(ax)
        ax.plot(x, ind["close"], color=FG, linewidth=1.0, label="Close")
        if "sma_fast" in ind:
            ax.plot(x, ind["sma_fast"], color=BLUE, linewidth=0.9, label="SMA fast")
        if "sma_slow" in ind:
            ax.plot(x, ind["sma_slow"], color=ORANGE, linewidth=0.9, label="SMA slow")
        if "bb_upper" in ind and "bb_lower" in ind:
            ax.fill_between(x, ind["bb_lower"], ind["bb_upper"],
                            color=BLUE, alpha=0.07)
        if result.buy_markers:
            bx, by = zip(*result.buy_markers)
            ax.scatter(bx, by, marker="^", color=GREEN, s=42, zorder=5,
                       edgecolors="white", linewidths=0.4, label="Buy")
        if result.sell_markers:
            sx, sy = zip(*result.sell_markers)
            ax.scatter(sx, sy, marker="v", color=RED, s=42, zorder=5,
                       edgecolors="white", linewidths=0.4, label="Sell")
        ax.set_title(f"{self._settings.ticker} simulated trades",
                     color=FG, fontsize=11)
        ax.legend(loc="upper left", fontsize=7, facecolor=PANEL,
                  edgecolor=GRAY, labelcolor=FG)

        ax = self._ax_equity
        ax.clear()
        self._style_axis(ax)
        ax.plot(result.equity.index, result.equity.values, color=GREEN,
                linewidth=1.2, label="Strategy equity")
        ax.plot(result.buy_hold.index, result.buy_hold.values, color=MUTED,
                linewidth=1.0, linestyle="--", label="Buy & hold")
        ax.set_ylabel("equity $", color=MUTED, fontsize=8)
        ax.set_title("Equity curve", color=FG, fontsize=10)
        ax.legend(loc="upper left", fontsize=7, facecolor=PANEL,
                  edgecolor=GRAY, labelcolor=FG)

        ax = self._ax_dd
        ax.clear()
        self._style_axis(ax)
        ax.fill_between(result.drawdown.index, result.drawdown.values * 100, 0,
                        color=RED, alpha=0.4)
        ax.set_ylabel("drawdown %", color=MUTED, fontsize=8)

        self._fig.autofmt_xdate(rotation=0, ha="center")
        # Force a synchronous draw: draw_idle() can drop the single repaint a
        # one-shot run needs (e.g. when this sub-tab isn't the visible one),
        # leaving a stale chart after a re-run.
        self._canvas.draw()
