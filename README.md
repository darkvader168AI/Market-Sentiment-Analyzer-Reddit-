This project is a **Streamlit web app** that turns **Reddit discussion into a sentiment signal**, then uses that signal to **build and backtest a portfolio** against benchmarks—complete with charts, basic risk metrics, and a “does sentiment predict returns?” stats panel.

---

## 1) What the app is

A 3-tab dashboard:

1. **Overview** – shows sentiment summary + how many posts per day + sentiment time series per ticker
2. **Top Posts** – surfaces the highest-impact posts (by upvotes/comments) and lets you export them
3. **Backtest** – converts sentiment into portfolio weights, simulates performance (with transaction costs), plots vs benchmarks, and computes metrics + correlation / Granger tests

It’s designed as an **interactive research tool** (not a fully automated trading system): you choose tickers, subreddits, date window, filters, rebalance frequency, and cost assumptions, and it runs the analysis on demand.

---

## 2) Data sources and access (Config / Secrets)

**Two external data feeds:**

* **Reddit posts** (via `praw`)

  * Credentials can come from `config.py`, environment variables, or Streamlit secrets (`st.secrets`)
  * If no credentials exist, it can run in **demo mode** using `demo_posts.csv`

* **Market prices** (via `yfinance`)

  * Pulls “Adj Close” (preferred) or “Close” for the tickers and benchmarks

This makes the app portable: it can run locally with env vars or on Streamlit Cloud with `st.secrets`.

---

## 3) Reddit ingestion + filtering (fetch_posts)

When you click **Analyze**, it:

* Reads your inputs:

  * **Portfolio tickers** (e.g., `AAPL,MSFT,NVDA`)
  * **Subreddits** (e.g., `wallstreetbets,stocks,investing`)
  * Optional **query filter** (e.g., “NVDA”) to focus the overview/top-posts
  * Time window (default last 90 days)
  * Quality filters: minimum upvotes / minimum comments

Then `fetch_posts()`:

* Pulls recent posts from each subreddit using `subreddit.new()`
* Filters locally by:

  * creation timestamp within the chosen window
  * query string match (if provided)
* Returns a DataFrame with post metadata: title, body, score, comments, timestamp, permalink

Caching:

* `fetch_posts` is cached for 15 minutes (`ttl=900`) to avoid hammering Reddit and to keep the app fast.

---

## 4) Sentiment scoring (VADER)

For each post, it runs **VADER sentiment** on:

> `title + "\n" + selftext`

Outputs:

* `compound` score in [-1, +1]
* label buckets:

  * Positive (compound ≥ 0.05)
  * Neutral (between)
  * Negative (≤ -0.05)

The Overview tab shows:

* average compound score
* distribution of labels
* posts per day line chart
* sentiment time series per portfolio ticker

---

## 5) Building a per-ticker daily sentiment signal

For each portfolio ticker `t`:

* finds posts that mention the ticker in title or body (`contains(t)`)
* resamples to daily mean sentiment
* aligns to **business days** and forward-fills gaps

Result:

* `sentiment_daily`: index = business days, columns = tickers, values ≈ “how positive Reddit was about each ticker that day”

---

## 6) Backtest: sentiment → weights → portfolio curve

This is the “investment engine.”

### A) Price retrieval

`get_prices()` downloads prices for:

* portfolio tickers + selected benchmarks (SPY/DIA/QQQ/ACWI)

It includes a lot of defensive logic:

* handles single ticker vs multi-ticker yfinance structures
* falls back if “Adj Close” isn’t present
* forward-fills missing values
* warns if tickers have no data

### B) Sentiment to weights (long-only)

Each sentiment value is mapped to a raw weight:

* sentiment is [-1, +1]
* mapped to [0, 1] via:
  [
  w = \text{clip}\left(\frac{x+1}{2}, 0, 1\right)
  ]
  Then weights are **normalized across tickers** each day so they sum to 1.

So:

* strongly negative sentiment → near 0 weight
* strongly positive sentiment → near 1 (before normalization)
* always **long-only**, no negative weights

### C) Rebalancing

You choose rebalance frequency:

* Daily / Weekly / Monthly / Semi-annually / Annually

The weights are:

* computed daily from sentiment
* then **resampled** to your rebalance frequency and forward-filled

### D) Transaction costs

Cost is based on **turnover**:

* turnover = sum(|w_t − w_{t-1}|)
* cost = turnover * (tx_cost_bps / 10,000)

Portfolio return each day:

* [
  r_{p,t} = \sum_i w_{i,t} r_{i,t} - \text{cost}_t
  ]

Then builds an equity curve from `initial_cap`.

### E) Benchmark curves

Also builds $-value benchmark curves for selected benchmarks.

---

## 7) Performance + stats diagnostics

The Backtest tab reports:

* **CAPM alpha/beta/R²** vs primary benchmark (SPY by default)
* **Sharpe** and **Sortino**
* **Max drawdown** and drawdown chart

And a separate panel:

### “Does sentiment predict returns?”

For each ticker, it computes:

* Pearson & Spearman correlation between sentiment and:

  * **next-day returns** (predictive mode) or
  * same-day returns
* p-values (when SciPy available)
* **Granger causality** test (low-lag) to see if sentiment leads returns statistically

It formats and displays results nicely with significance markers.

---

## 8) What this project produces (practical deliverables)

* A curated, filterable set of “important” Reddit posts + export CSV
* A per-ticker daily sentiment series
* A backtested equity curve of a sentiment-weighted portfolio vs benchmarks + export curves CSV
* A quick statistical view of whether sentiment has predictive value (correlation + Granger)

---
