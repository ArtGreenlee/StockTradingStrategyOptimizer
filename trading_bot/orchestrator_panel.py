"""Orchestrator tab: query the LLM/Copilot endpoint for sentiment analysis.

The user pastes text (a headline, earnings note, social post, etc.), clicks
**Analyze sentiment**, and the panel sends it -- with a sentiment-analysis
prompt -- to the configured LLM endpoint. The request runs on a background
thread; the parsed result (label, score, confidence, rationale) returns over a
queue drained on the Tk main loop, so the UI never freezes.

If no endpoint is configured the panel says so and the button is disabled,
matching the graceful-degradation pattern used elsewhere in the app.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import tkinter as tk
from tkinter import ttk

from .config import Settings
from .llm_client import LLMClient, LLMError
from .orchestrator import SYSTEM_PROMPT, SentimentResult, analyze_sentiment
from .theme import BG, BLUE, FG, GRAY, GREEN, MUTED, ORANGE, PANEL, RED

_LABEL_COLORS = {"BULLISH": GREEN, "BEARISH": RED, "NEUTRAL": GRAY}


class OrchestratorPanel:
    """Builds and drives the Orchestrator (sentiment-analysis) tab."""

    def __init__(self, parent: tk.Widget, settings: Settings) -> None:
        self._parent = parent
        self._settings = settings
        self._llm = LLMClient(settings)
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._busy = False

        self._build()
        self._parent.after(200, self._poll)

    # ---- Layout ----------------------------------------------------------

    def _build(self) -> None:
        root = ttk.Frame(self._parent, style="TFrame", padding=(10, 8))
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="Sentiment Orchestrator", style="Header.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            root,
            text=("Send text to the LLM/Copilot endpoint and get a market "
                  "sentiment read (bullish / bearish / neutral)."),
            style="Muted.TLabel", wraplength=900, justify="left",
        ).pack(anchor="w", pady=(0, 6))

        # Endpoint status.
        self._endpoint_var = tk.StringVar(value=f"Endpoint: {self._llm.description}")
        ttk.Label(root, textvariable=self._endpoint_var, style="Muted.TLabel",
                  wraplength=900, justify="left").pack(anchor="w")

        # Input text field.
        ttk.Label(root, text="Text to analyze", style="Header.TLabel").pack(
            anchor="w", pady=(10, 2)
        )
        self._input = tk.Text(
            root, height=7, bg=PANEL, fg=FG, insertbackground=FG,
            relief=tk.FLAT, font=("Segoe UI", 10), wrap=tk.WORD,
        )
        self._input.pack(fill=tk.X)
        self._input.insert(
            tk.END,
            "Example: Apple raised full-year guidance and announced a record "
            "buyback; analysts upgraded the stock.",
        )

        # Optional editable instruction/prompt (advanced).
        adv = ttk.Frame(root, style="TFrame")
        adv.pack(fill=tk.X, pady=(8, 0))
        self._show_prompt = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            adv, text="Edit sentiment prompt (advanced)",
            variable=self._show_prompt, command=self._toggle_prompt,
        ).pack(side=tk.LEFT)

        self._prompt_box = tk.Text(
            root, height=6, bg=PANEL, fg=MUTED, insertbackground=FG,
            relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD,
        )
        self._prompt_box.insert(tk.END, SYSTEM_PROMPT)
        # Hidden until the advanced toggle is enabled.

        # Controls.
        controls = ttk.Frame(root, style="TFrame")
        controls.pack(fill=tk.X, pady=(8, 2))
        self._analyze_btn = tk.Button(
            controls, text="Analyze sentiment", command=self._on_analyze,
            bg=GREEN, fg="white", activebackground=GRAY, relief=tk.FLAT, padx=12,
        )
        self._analyze_btn.pack(side=tk.LEFT)
        tk.Button(
            controls, text="Clear", command=self._on_clear, bg=PANEL, fg=FG,
            activebackground=GRAY, relief=tk.FLAT, padx=10,
        ).pack(side=tk.LEFT, padx=6)
        self._status_var = tk.StringVar(value="Idle.")
        ttk.Label(controls, textvariable=self._status_var, style="Muted.TLabel").pack(
            side=tk.LEFT, padx=10
        )

        # Result: verdict badge + metrics + rationale.
        result_wrap = ttk.Frame(root, style="TFrame")
        result_wrap.pack(fill=tk.X, pady=(10, 0))
        self._verdict_lbl = tk.Label(
            result_wrap, text=" — ", bg=GRAY, fg="white",
            font=("Segoe UI", 16, "bold"), padx=14, pady=4,
        )
        self._verdict_lbl.pack(side=tk.LEFT)
        self._metrics_var = tk.StringVar(value="")
        ttk.Label(result_wrap, textvariable=self._metrics_var, style="TLabel",
                  font=("Consolas", 11)).pack(side=tk.LEFT, padx=14)

        ttk.Label(root, text="Rationale", style="Header.TLabel").pack(
            anchor="w", pady=(10, 2)
        )
        self._rationale = tk.Text(
            root, height=4, bg=PANEL, fg=FG, relief=tk.FLAT,
            font=("Segoe UI", 10), wrap=tk.WORD,
        )
        self._rationale.pack(fill=tk.X)
        self._rationale.configure(state=tk.DISABLED)

        ttk.Label(root, text="Raw model response", style="Header.TLabel").pack(
            anchor="w", pady=(10, 2)
        )
        self._raw = tk.Text(
            root, height=8, bg=PANEL, fg=MUTED, relief=tk.FLAT,
            font=("Consolas", 9), wrap=tk.WORD,
        )
        self._raw.pack(fill=tk.BOTH, expand=True)
        self._raw.configure(state=tk.DISABLED)

        if not self._llm.configured:
            self._analyze_btn.configure(state=tk.DISABLED)
            self._status_var.set("LLM endpoint not configured — see .env (LLM_*).")

    def _toggle_prompt(self) -> None:
        if self._show_prompt.get():
            self._prompt_box.pack(fill=tk.X, pady=(4, 0))
        else:
            self._prompt_box.pack_forget()

    # ---- Actions ---------------------------------------------------------

    def _on_clear(self) -> None:
        self._input.delete("1.0", tk.END)

    def _on_analyze(self) -> None:
        if self._busy:
            return
        text = self._input.get("1.0", tk.END).strip()
        if not text:
            self._status_var.set("Enter some text to analyze.")
            return
        instruction = (
            self._prompt_box.get("1.0", tk.END).strip()
            if self._show_prompt.get() else None
        )

        self._busy = True
        self._analyze_btn.configure(state=tk.DISABLED, text="Analyzing...")
        self._status_var.set(f"Querying {self._settings.llm_model}...")

        def work() -> None:
            try:
                result = analyze_sentiment(self._llm, text, instruction)
                self._queue.put(("result", result))
            except LLMError as exc:
                self._queue.put(("error", str(exc)))
            except Exception as exc:  # unexpected
                self._queue.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    # ---- Queue draining --------------------------------------------------

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "result":
                    self._render_result(payload)
                elif kind == "error":
                    self._render_error(payload)
        except queue.Empty:
            pass
        self._parent.after(200, self._poll)

    def _finish(self) -> None:
        self._busy = False
        self._analyze_btn.configure(state=tk.NORMAL, text="Analyze sentiment")

    def _render_result(self, result: SentimentResult) -> None:
        self._finish()
        self._status_var.set("Done.")
        color = _LABEL_COLORS.get(result.label, GRAY)
        self._verdict_lbl.configure(text=f" {result.label} ", bg=color)
        self._metrics_var.set(
            f"score {result.score:+.2f}   confidence {result.confidence * 100:.0f}%"
        )
        self._set_text(self._rationale, result.rationale)
        self._set_text(self._raw, result.raw)

    def _render_error(self, message: str) -> None:
        self._finish()
        self._status_var.set(f"Error: {message}")
        self._verdict_lbl.configure(text=" ERROR ", bg=ORANGE)
        self._metrics_var.set("")
        self._set_text(self._rationale, message)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text or "")
        widget.configure(state=tk.DISABLED)
