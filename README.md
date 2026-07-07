# Volatility Mean-Reversion Trading Bot (Paper Only)

A single-ticker stock trading bot with a live GUI. It pulls market data from
the **Alpaca** API, visualizes fundamental snapshot data and volatility, and
makes **small** trades. It runs in **paper mode only** — the live, real-money
endpoint is never used.

## What it shows

The GUI exposes every step of the decision process in real time:

- **Volatility regime** — realized volatility, its percentile, ATR, Bollinger
  band width, classified LOW / NORMAL / HIGH.
- **RSI (14)** — oversold / overbought momentum extremes.
- **Bollinger %B** — where price sits inside the volatility envelope.
- **Trend (SMA 10/30)** — light directional bias.
- **Fundamentals (snapshot)** — gap vs previous close, intraday range
  position, VWAP side (the real data points Alpaca's free feed provides).
- **Risk / position** — enforces the small-size cap and can veto a signal.
- **Verdict** — aggregated BUY / SELL / HOLD with a score and confidence.

Charts (price + Bollinger/SMA + trade markers, volatility, RSI, volume)
redraw on every poll so you watch the bot reason with fresh information.

A second **Backtest / Simulation** tab lets you download historical data, tune
every parameter, and replay the strategy (including an artificial trade
latency) — see [Using the GUI](#using-the-gui).

## Strategy

In one line: **fade short-term extremes in the direction of the longer-term
trend, with defined risk.** The bot buys statistically stretched dips (RSI,
Bollinger %B, and a z-score) but only when price is above a long-term moving
average (trend filter), sizes each trade by volatility, and exits on
ATR-based stops/targets.

### Researched techniques used

These are established, documented methods (not curve-fitting):

- **ATR-based exits** (Wilder, *New Concepts in Technical Trading Systems*):
  every position has an ATR stop-loss, ATR take-profit, optional ATR trailing
  stop, and a time stop. This is the biggest single performance lever versus
  only ever closing on an opposite signal.
- **Volatility-based position sizing** (ATR risk parity / fractional Kelly):
  size each trade to risk a fixed % of equity over the stop distance, so size
  self-adjusts to volatility. (The old fixed "1 share" sizing was the main
  reason the equity curve looked flat — trades barely moved a $100k account.)
- **Trend filter** (Connors/Alvarez, *Short Term Trading Strategies That
  Work*): only take mean-reversion longs above a long-term SMA; fighting a
  confirmed downtrend bleeds.
- **Z-score mean reversion** (Ornstein–Uhlenbeck / pairs-trading literature):
  a principled "stretch" measure, `z = (price − mean) / std`.
- **Risk-adjusted optimization with out-of-sample validation** (López de
  Prado, on backtest overfitting): the hyperparameter search can optimize
  Sharpe or Calmar (not just raw return) and holds out a tail of the data to
  flag overfit configurations.

> Reality check: no parameter set guarantees profit. These techniques improve
> *risk-adjusted* behavior and cap losses. On mean-reverting markets the bot
> wins often with small gains; in strong trends the ATR stop limits damage.
> Always compare the **out-of-sample** result, not just in-sample.

### Unusual Options Activity (UOA) flow

The bot also scans the underlying's options chain live and folds an
options-flow read into its decision. UOA is detected with standard,
documented methods:

- **Volume / Open Interest (Vol/OI)** — the flagship flag. Open interest is
  how many contracts exist; volume is how many traded today. When a contract's
  *volume exceeds its open interest* (Vol/OI > ~2×), far more traded today than
  existed at the open — i.e. large **new** positioning, not routine closing.
- **Put/Call ratio** — total put vs call volume (and open interest) gives a
  sentiment skew; extremes are themselves unusual.
- **Aggressor side** — comparing the last trade to the bid/ask: trades at the
  ask are buyer-initiated (urgent demand); at the bid, seller-initiated. Heavy
  at-ask call buying is bullish; at-ask put buying is bearish.
- **Net premium (dollar volume)** — `volume × price × 100` shows where real
  capital is committed.
- **Implied volatility** and **near-dated, near-the-money concentration** —
  where informed/speculative flow tends to sit.

These roll up into a net sentiment in `[-1, 1]` that nudges the decision score
(weight `options_weight`), and the flagged contracts are listed in the
**Options flow (UOA)** panel and logged. Data comes from Alpaca's options API
(`OPTIONS_FEED=indicative` is free/delayed; `opra` is real-time and needs a
subscription). If your account has no options entitlement, the bot degrades
gracefully and shows "Options flow unavailable".

> Note: options flow is a **live-only** enrichment — Alpaca has no easily
> replayable historical chain, so the **backtest ignores it** (the simulated
> strategy is unchanged). It affects live decisions only.

## Setup

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your PAPER API keys
copy .env.example .env
#    then edit .env and paste your Alpaca PAPER keys

# 3. Run
python main.py
```

Get **paper** API keys at <https://app.alpaca.markets/> →
*Paper Trading* → *API Keys*.

## Configuration (`.env`)

| Variable                | Meaning                                          | Default |
| ----------------------- | ------------------------------------------------ | ------- |
| `ALPACA_API_KEY`        | Alpaca paper API key                             | —       |
| `ALPACA_SECRET_KEY`     | Alpaca paper secret key                          | —       |
| `TICKER`                | Symbol to trade                                  | `AAPL`  |
| `MAX_POSITION_SHARES`   | Max shares held at once (keeps trades small)     | `10`    |
| `ORDER_QTY`             | Shares per order                                 | `1`     |
| `POLL_INTERVAL_SECONDS` | How often to refresh data + re-evaluate          | `15`    |
| `LOOKBACK_MINUTES`      | Minutes of 1-min history for indicators/plotting | `390`   |
| `PROXY_URL`             | Forward proxy for HTTP+HTTPS (corporate network) | —       |
| `HTTP_PROXY` / `HTTPS_PROXY` | Per-scheme proxy overrides                  | —       |
| `NO_PROXY`              | Comma-separated hosts to bypass the proxy        | —       |
| `PROXY_CA_BUNDLE`       | Path to corporate root CA (PEM) for TLS intercept| —       |
| `PROXY_VERIFY_SSL`      | TLS verification (set `false` only as last resort)| `true`  |

## Corporate proxy / firewall

Behind a corporate firewall the bot can route all Alpaca API traffic through a
forward proxy. Set `PROXY_URL` in `.env` (credentials may be embedded and are
masked in the GUI/logs):

```dotenv
PROXY_URL=http://user:pass@proxy.corp.example:8080
NO_PROXY=localhost,127.0.0.1
```

The proxy applies to both the market-data and trading clients. The top bar
shows a **PROXY** / **DIRECT** badge, and the **Test connection** button makes
a single lightweight API call so proxy issues surface immediately.

**TLS-intercepting proxies:** if your proxy re-signs HTTPS with a corporate
root CA, point `PROXY_CA_BUNDLE` at that CA's PEM file — the secure fix.
Only as a last resort set `PROXY_VERIFY_SSL=false`; this disables certificate
verification and the GUI flags it loudly.

## Using the GUI

The window has three tabs (modes):

### Live Trading

- The bot starts **observing** immediately and explains its reasoning.
- **Change the ticker** any time using the symbol box in the top bar (type a
  symbol and press Enter or click **Set**); the polling loop restarts on the
  new symbol.
- **Auto-trade (paper)** is **OFF** by default. Tick it to let the bot place
  paper orders when its verdict is BUY or SELL.
- **Stop / Start** pauses and resumes the polling loop.
- The **Strategy parameters** panel shows the active tuning. Click **Import
  from Backtest** to load the parameters currently in the Backtest editor into
  live trading (takes effect on the next poll).
- The trade log records every order attempt and result.

### Backtest / Simulation

Tune the strategy and replay it over historical data — no real or paper orders
are placed.

1. **Download data** — pick a timeframe (1Min … 1Day) and date range, then
   download historical bars for the configured ticker. Tick **Download + use
   news sentiment** to also pull Alpaca news for the range (see below).
2. **Tune parameters** — every strategy and simulation knob is editable
   (indicator periods, regime cutoffs, signal weights, decision thresholds,
   sizing, news). **Reset** restores defaults.
3. **Run simulation** (Single run tab) — replays the bars through the *same*
   analysis + decision engine the live bot uses, then shows the equity curve
   vs buy & hold, trade markers, drawdown, and stats (return, win rate, max
   drawdown, Sharpe, exposure).

**News sentiment (backtest + live).** Unlike options/UOA, Alpaca's News API is
*historical with timestamps*, so news can be backtested. Articles are scored
with a deterministic finance lexicon and applied **strictly causally** — a
headline only affects bars at/after its `publish time + news_publish_lag_seconds`,
so there is no look-ahead. Per-bar sentiment is decay-weighted
(`news_half_life_minutes`) over a `news_lookback_minutes` window, and nudges the
decision score by `news_weight`. The same scorer runs live (Live tab → **News
sentiment** panel), so backtest and live behavior match. It also flows through
the hyperparameter search (news is split at the same OOS boundary).

#### Hyperparameter search (optimize for profit)

On the **Hyperparameter search** sub-tab:

- Enable the parameters to search and set their min/max bounds (a curated
  default space is pre-filled, including the ATR stop/target and risk-sizing
  knobs).
- Choose **trials**, **method** (`guided` hill-climb or `random`), a **seed**,
  an **objective** (`return`, `sharpe`, `calmar`, or `sharpe_calmar`), and an
  **OOS test %** (fraction of the tail held out), then **Start search**.
- **Anti-degeneracy guard** — set **Min trades** and **Min expo %** to penalize
  configs that "win" by trading almost never (the classic Sharpe pathology: a
  near-flat strategy with a handful of lucky trades scores an absurd Sharpe).
  Any objective below its minimum activity is scaled down proportionally, so a
  config must actually *participate* to score well. This is also how you get a
  **penalized Sharpe**: pick `sharpe` and set Min trades / Min expo %.
- **`sharpe_calmar` dual objective** — a balanced blend of (squashed) Sharpe
  and Calmar that requires **both** to be good (gaming one metric won't win).
  **Calmar wt** sets the blend (0 = pure Sharpe, 1 = pure Calmar, 0.5 = equal).
- A **live line graph shows how each (best-so-far) parameter evolves** over the
  trials (normalized to its range), with the objective curve below — so you
  watch the search converge in real time. **Stop** ends it early.
- On completion the status line reports both the **in-sample** and
  **out-of-sample** result. If OOS is far worse than in-sample, the
  configuration is overfit — prefer one that holds up out-of-sample.
- **Apply best to editor** loads the most profitable configuration into the
  parameter editor; from there you can run it or **Import from Backtest** on the
  Live tab.

**Artificial trade latency:** set `latency_seconds` in the Simulation group to
model execution delay. A decision on one bar fills on the bar that lands after
the delay (at its opening price), plus optional `slippage_bps` and
`commission_per_trade`, so you can see how latency erodes a fast strategy.

The simulation is **causal** (no look-ahead): indicators are backward-looking,
the volatility-regime percentile uses a trailing window, and intraday
"fundamentals" are reconstructed from bars already seen.

### Orchestrator (LLM / Copilot sentiment)

Send free text (a headline, earnings note, social post, etc.) to an
OpenAI-compatible LLM/Copilot endpoint and get a market **sentiment** read.

1. Configure the endpoint in `.env` (`LLM_BASE_URL`, `LLM_API_KEY`,
   `LLM_MODEL`; set `LLM_API_VERSION` for Azure OpenAI). Requests reuse the
   proxy settings. If left blank, the tab is disabled and says so.
2. Paste text and click **Analyze sentiment**. The query runs on a background
   thread, so the UI stays responsive.
3. The result shows a colour-coded **BULLISH / BEARISH / NEUTRAL** verdict, a
   score (−1…+1), a confidence, a one-line rationale, and the raw model reply.
4. **Edit sentiment prompt (advanced)** lets you override the system
   instruction sent to the model.

The model is asked for strict JSON; the parser tolerates code fences / stray
prose and falls back to a keyword scan, so the panel always shows a result.

## Safety notes

- `TradingClient(..., paper=True)` is hard-coded; there is no setting to point
  the bot at the live endpoint.
- Order size is capped by `ORDER_QTY` and total exposure by
  `MAX_POSITION_SHARES`.
- This project is for education / experimentation. It is **not** financial
  advice and makes no guarantee of profit.

## Notes on data

Alpaca's free feed uses **IEX** data and does not expose ratio-style
fundamentals (P/E, market cap). The bot therefore treats the real available
snapshot data (daily OHLCV, VWAP, previous close) as its "fundamental"
context and derives gap / range / VWAP signals from it.
