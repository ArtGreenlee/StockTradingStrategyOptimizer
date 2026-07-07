"""Entry point for the paper-trading volatility bot.

Usage:
    1. pip install -r requirements.txt
    2. copy .env.example to .env and fill in your Alpaca PAPER API keys
    3. python main.py
"""

from __future__ import annotations

import multiprocessing
import sys
import tkinter as tk
from tkinter import messagebox

from trading_bot.alpaca_client import AlpacaClient
from trading_bot.config import load_settings
from trading_bot.gui import TradingBotGUI


def main() -> int:
    try:
        settings = load_settings()
    except RuntimeError as exc:
        # Show the config error in a dialog so double-click users see it too.
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Configuration error", str(exc))
        root.destroy()
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        client = AlpacaClient(settings)
    except Exception as exc:  # SDK construction / credential issues
        print(f"Failed to initialize Alpaca client: {exc}", file=sys.stderr)
        return 1

    gui = TradingBotGUI(settings, client)
    gui.run()
    return 0


if __name__ == "__main__":
    # Required for the hyperparameter-search subprocess under the "spawn" start
    # method (Windows/macOS), and harmless elsewhere.
    multiprocessing.freeze_support()
    raise SystemExit(main())
