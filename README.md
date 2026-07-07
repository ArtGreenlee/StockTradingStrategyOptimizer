# Stock Trading Strategy Optimizer (Portfolio Project)

This is a Python desktop app for testing and monitoring a paper-trading strategy.
It helps me explore trading ideas with live market data, backtesting tools, and a
simple visual dashboard.

## Screenshot

![App screenshot](Screenshot%202026-07-05%20210502.png)

## What This Project Does

- Connects to Alpaca paper trading APIs
- Shows strategy signals in a live GUI
- Runs backtests on historical price data
- Lets me tune strategy parameters
- Includes options-flow and sentiment inputs
- Keeps trading in paper mode for safety

## Tech Stack

- Python
- Tkinter GUI
- Alpaca API
- Pandas / NumPy style data processing

## Why I Built It

I wanted one place to:

- analyze short-term trading setups,
- compare strategy changes quickly,
- and practice building production-style Python tooling.

## Quick Start

```powershell
pip install -r requirements.txt
copy .env.example .env
# add your Alpaca paper keys to .env
python main.py
```

## Notes

- This project is for learning and portfolio demonstration.
- It is not financial advice.
- Do not commit real credentials. Keep secrets only in `.env`.
