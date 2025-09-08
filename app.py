# app.py
# Streamlit app: Reddit Sentiment → Portfolio Backtester (refactored + upgrades)

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================
# Config / Secrets
# =========================
try:
    import config
except Exception:
    class _SecretCfg:
        REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
        REDDIT_SECRET    = os.getenv("REDDIT_SECRET")
        REDDIT_USERAGENT = os.getenv("REDDIT_USERAGENT", "sentiment-app/1.0")
    config = _SecretCfg()

st.set_page_config(page_title="Market Sentiment Analyzer | Reddit", layout="wide")
st.title("Market Sentiment Analyzer | Reddit")

def has_reddit_keys():
    cid = getattr(config, "REDDIT_CLIENT_ID", None) or st.secrets.get("REDDIT_CLIENT_ID", None)
    sec = getattr(config, "REDDIT_SECRET", None) or st.secrets.get("REDDIT_SECRET", None)
    ua  = getattr(config, "REDDIT_USERAGENT", None) or st.secrets.get("REDDIT_USERAGENT", None)
    return bool(cid and sec and ua)

@st.cache_resource(show_spinner=False)
def init_praw():
    import praw
    cid = getattr(config, "REDDIT_CLIENT_ID", None) or st.secrets.get("REDDIT_CLIENT_ID", None)
    sec = getattr(config, "REDDIT_SECRET", None) or st.secrets.get("REDDIT_SECRET", None)
    ua  = getattr(config, "REDDIT_USERAGENT", None) or st.secrets.get("REDDIT_USERAGENT", "sentiment-app/1.0")
    if not (cid and sec and ua):
        return None
    reddit = praw.Reddit(
        client_id=cid,
        client_secret=sec,
        user_agent=ua,
        check_for_async=False
    )
    return reddit

def reddit_url_from_submission(sub):
    base = "https://www.reddit.com"
    try:
        return base + str(sub.permalink).rstrip("/")
    except Exception:
        return getattr(sub, "url", base)

# =========================
# Data Fetching (cached)
# =========================
@st.cache_data(show_spinner=False, ttl=900)
def fetch_posts(subreddits, query, start_ts, end_ts, limit_per_sub=200, demo=False):
    """
    Returns DataFrame with: subreddit, id, title, selftext, score, num_comments, created_utc, permalink
    If demo=True or Reddit is not configured, loads from demo CSV if present.
    """
    cols = ["subreddit","id","title","selftext","score","num_comments","created_utc","permalink"]
    if demo or not has_reddit_keys():
        demo_path = os.path.join(os.path.dirname(__file__), "demo_posts.csv")
        if os.path.exists(demo_path):
            df = pd.read_csv(demo_path)
            if "created_utc" in df.columns:
                df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
            return df[cols].dropna(subset=["created_utc"]).copy()
        return pd.DataFrame(columns=cols)

    reddit = init_praw()
    if reddit is None:
        return pd.DataFrame(columns=cols)

    rows = []
    sub_list = [s.strip() for s in subreddits.split(",") if s.strip()]
    for s in sub_list:
        try:
            subreddit = reddit.subreddit(s)
            # Pull recent posts; filter locally by time & query
            for submission in subreddit.new(limit=limit_per_sub):
                cts = float(getattr(submission, "created_utc", 0.0) or 0.0)
                if cts < start_ts or cts > end_ts:
                    continue
                text_blob = f"{submission.title}\n{getattr(submission,'selftext','')}"
                if query and (query.lower() not in text_blob.lower()):
                    continue
                rows.append({
                    "subreddit": s,
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": getattr(submission, "selftext", ""),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": cts,
                    "permalink": getattr(submission, "permalink", ""),
                })
        except Exception as e:
            st.warning(f"Failed to fetch from r/{s}: {e}")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df[cols].copy()

# =========================
# Sentiment
# =========================
analyzer = SentimentIntensityAnalyzer()

def score_text(t):
    t = t or ""
    s = analyzer.polarity_scores(t)
    if s["compound"] >= 0.05:
        label = "Positive"
    elif s["compound"] <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return s["compound"], label

def label_distribution(labels: pd.Series):
    counts = labels.value_counts(dropna=False)
    order = ["Positive","Neutral","Negative"]
    return pd.Series({k: int(counts.get(k, 0)) for k in order})

# =========================
# Backtest helpers
# =========================
def get_prices(tickers, start, end):
    try:
        # Convert to list if single ticker
        ticker_list = [tickers] if isinstance(tickers, str) else tickers
        
        # Download data
        data = yf.download(ticker_list, start=start, end=end, progress=False)
        
        if data.empty:
            st.warning(f"No data downloaded for tickers: {ticker_list}")
            return pd.DataFrame(columns=ticker_list)
        
        # Debug information
        st.write(f"Downloaded data shape: {data.shape}")
        st.write(f"Downloaded data columns: {data.columns.tolist()}")
        
        # Handle different data structures from yfinance
        if len(ticker_list) == 1:
            # Single ticker - data is a DataFrame with columns like 'Adj Close'
            if 'Adj Close' in data.columns:
                px = data['Adj Close']
            elif 'Close' in data.columns:
                px = data['Close']
            else:
                # If neither exists, try to get the first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    px = data[numeric_cols[0]]
                else:
                    st.warning(f"No suitable price column found for {ticker_list[0]}")
                    return pd.DataFrame(columns=ticker_list)
            
            # Convert to DataFrame with ticker as column name
            px = px.to_frame()
            px.columns = ticker_list
            
        else:
            # Multiple tickers - data is a MultiIndex DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                # MultiIndex columns: (price_type, ticker)
                # Try to get 'Adj Close' first, then 'Close'
                if 'Adj Close' in data.columns.get_level_values(0):
                    px = data['Adj Close']
                elif 'Close' in data.columns.get_level_values(0):
                    px = data['Close']
                else:
                    # Try to get the first numeric column
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        px = data[numeric_cols[0]]
                    else:
                        st.warning(f"No suitable price column found for {ticker_list}")
                        return pd.DataFrame(columns=ticker_list)
            else:
                # Single level columns - this shouldn't happen with multiple tickers
                st.warning(f"Unexpected data structure for multiple tickers: {ticker_list}")
                return pd.DataFrame(columns=ticker_list)
        
        # Ensure we have the right column names
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(1)  # Get ticker names from second level
        
        # Forward fill and drop rows with all NaN
        result = px.ffill().dropna(how="all")
        
        # Ensure all requested tickers are present
        for ticker in ticker_list:
            if ticker not in result.columns:
                result[ticker] = np.nan
        
        # Reorder columns to match requested ticker order
        result = result.reindex(columns=ticker_list)
        
        st.write(f"Final price data shape: {result.shape}")
        st.write(f"Final price data columns: {result.columns.tolist()}")
        return result
        
    except Exception as e:
        st.error(f"Error downloading price data: {str(e)}")
        # Return empty DataFrame with proper structure
        return pd.DataFrame(columns=ticker_list if isinstance(tickers, list) else [tickers])

def map_sentiment_to_weight(x):
    if pd.isna(x):
        return 0.0
    return float(np.clip((x + 1) / 2.0, 0.0, 1.0))

def make_weights(sentiment_daily, tickers):
    # Ensure all tickers are present in sentiment_daily
    for ticker in tickers:
        if ticker not in sentiment_daily.columns:
            sentiment_daily[ticker] = 0.0
    
    # Only use tickers that exist in sentiment_daily
    available_tickers = [t for t in tickers if t in sentiment_daily.columns]
    if not available_tickers:
        # If no tickers available, return zeros
        return pd.DataFrame(0.0, index=sentiment_daily.index, columns=tickers)
    
    w = sentiment_daily[available_tickers].map(map_sentiment_to_weight)
    w_sum = w.sum(axis=1).replace(0, np.nan)
    w = w.div(w_sum, axis=0).fillna(0.0)
    
    # Ensure all requested tickers are in the result
    for ticker in tickers:
        if ticker not in w.columns:
            w[ticker] = 0.0
    
    return w[tickers].clip(0, 1)

# ---- Extra Performance Utilities ----
def max_drawdown(equity_curve: pd.Series):
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return float(dd.min()) if len(dd) else np.nan, dd

def sharpe_sortino(daily_ret: pd.Series, rf_daily: float = 0.0):
    if daily_ret.empty:
        return np.nan, np.nan
    excess = daily_ret - rf_daily
    vol = excess.std(ddof=0)
    sharpe = np.sqrt(252) * excess.mean() / vol if vol and vol > 0 else np.nan
    downside = excess[excess < 0]
    dvol = downside.std(ddof=0)
    sortino = np.sqrt(252) * excess.mean() / dvol if dvol and dvol > 0 else np.nan
    return float(sharpe), float(sortino)

def _freq_code(label: str) -> str:
    return {"Daily": "D", "Weekly": "W-FRI", "Monthly": "M", "Semi-annually": "6M", "Annually": "Y"}[label]

# ---- Backtest (supports extra benchmarks & rebalance) ----
def backtest(
    sentiment_daily: pd.DataFrame,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    tx_cost_bps: int = 10,
    initial_cap: int = 100_000,
    primary_bench: str = "SPY",
    extra_benches: list[str] = None,
    rebalance_label: str = "Daily",
):
    extra_benches = extra_benches or []
    benches = [primary_bench] + [b for b in extra_benches if b != primary_bench]

    # Prices & returns - get all data at once
    all_tickers = tickers + benches
    px = get_prices(all_tickers, start, end)
    
    # Debug: Check what we got from get_prices
    if px.empty:
        st.error("No price data retrieved. Please check if the tickers exist and have valid data.")
        return pd.Series(), pd.DataFrame(), pd.Series(), pd.Series(), pd.DataFrame()
    
    # Check which tickers we actually have data for
    available_price_tickers = [t for t in all_tickers if t in px.columns and not px[t].isna().all()]
    if not available_price_tickers:
        st.error(f"No price data found for any of the requested tickers: {all_tickers}")
        return pd.Series(), pd.DataFrame(), pd.Series(), pd.Series(), pd.DataFrame()
    
    # Filter to only use tickers we have data for
    px = px[available_price_tickers]
    rets = px.pct_change().fillna(0.0)
    
    # Show which tickers we successfully retrieved
    st.success(f"Successfully retrieved price data for: {available_price_tickers}")

    # Sentiment → daily weights → rebalance to chosen freq
    freq = _freq_code(rebalance_label)
    w_raw = sentiment_daily.reindex(rets.index, method="ffill").fillna(0.0)
    w_daily = make_weights(w_raw, tickers)
    w_rb = w_daily.resample(freq).last().reindex(rets.index, method="ffill")

    # Turnover & costs (simple daily application of rebalance changes)
    w_prev = w_rb.shift(1).fillna(0.0)
    turnover = (w_rb - w_prev).abs().sum(axis=1)
    cost = turnover * (tx_cost_bps / 10_000.0)

    # Portfolio returns, curve
    # Ensure we only use tickers that exist in both w_rb and rets
    available_tickers = [t for t in tickers if t in w_rb.columns and t in rets.columns]
    if not available_tickers:
        st.error(f"No valid tickers found for backtesting. Available tickers in sentiment: {list(w_rb.columns)}, Available tickers in prices: {list(rets.columns)}")
        return pd.Series(), pd.DataFrame(), pd.Series(), pd.Series(), pd.DataFrame()
    
    # Ensure all data is numeric
    w_rb_clean = w_rb[available_tickers].astype(float)
    rets_clean = rets[available_tickers].astype(float)
    cost_clean = cost.astype(float)
    
    port_ret = (w_rb_clean * rets_clean).sum(axis=1) - cost_clean
    port_curve = (1.0 + port_ret).cumprod() * initial_cap

    # Benchmarks
    bench_curves = {}
    available_benches = [b for b in benches if b in rets.columns]
    for b in available_benches:
        bench_ret = rets[b].astype(float)
        bench_curves[b] = (1.0 + bench_ret).cumprod() * initial_cap
    
    if bench_curves:
        bench_curves = pd.DataFrame(bench_curves)
    else:
        bench_curves = pd.DataFrame()
    
    # Return market returns for primary benchmark if available
    mkt_ret = rets[primary_bench].astype(float) if primary_bench in rets.columns else pd.Series()
    bench_rets = rets[available_benches].astype(float) if available_benches else pd.DataFrame()

    return port_curve, bench_curves, port_ret, mkt_ret, bench_rets, rets

def capm_metrics(port_ret, mkt_ret, rf_daily=0.0):
    try:
        import statsmodels.api as sm
    except Exception:
        return None
    y = port_ret - rf_daily
    X = pd.DataFrame({"mkt": mkt_ret - rf_daily})
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    alpha_daily = model.params.get("const", np.nan)
    beta = model.params.get("mkt", np.nan)
    r2 = model.rsquared
    alpha_annual = alpha_daily * 252 if pd.notna(alpha_daily) else np.nan
    return {"alpha_annual": alpha_annual, "beta": beta, "r2": r2, "alpha_daily": alpha_daily}

# === Sentiment vs Returns Correlation Analysis (robust) ===
def sentiment_return_correlation(sentiment_daily: pd.DataFrame, rets: pd.DataFrame, tickers: list[str], use_next_day: bool = True) -> pd.DataFrame:
    """Aligns sentiment to trading days, forward-fills gaps, supports next-day returns."""
    results = []
    if rets.empty or sentiment_daily.empty:
        return pd.DataFrame(results)

    for t in tickers:
        if t not in sentiment_daily.columns or t not in rets.columns:
            continue

        # Align sentiment to trading-day index and carry forward gaps
        s = sentiment_daily[t].reindex(rets.index).ffill()
        r = rets[t].shift(-1) if use_next_day else rets[t]

        aligned = pd.concat([
            pd.to_numeric(s, errors="coerce"),
            pd.to_numeric(r, errors="coerce")
        ], axis=1).dropna()

        if len(aligned) < 5:  # Need more data for Granger test
            continue

        try:
            pearson = aligned.corr().iloc[0, 1]
            spearman = aligned.corr(method="spearman").iloc[0, 1]
            try:
                from scipy.stats import pearsonr, spearmanr
                pearson_p = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])[1]
                spearman_p = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])[1]
            except Exception:
                pearson_p = np.nan
                spearman_p = np.nan

            # Granger causality test
            granger_p = np.nan
            granger_f = np.nan
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                # Prepare data for Granger test: [returns, sentiment] format
                granger_data = aligned.iloc[:, [1, 0]].values  # [returns, sentiment]
                
                # Test if sentiment (column 1) Granger-causes returns (column 0)
                # Use maxlag=2 to avoid overfitting with small samples
                maxlag = min(2, len(granger_data) // 3)
                if maxlag >= 1:
                    gc_result = grangercausalitytests(granger_data, maxlag=maxlag, verbose=False)
                    # Use the most recent lag result
                    granger_f = gc_result[maxlag][0]['ssr_ftest'][0]
                    granger_p = gc_result[maxlag][0]['ssr_ftest'][1]
            except Exception:
                granger_p = np.nan
                granger_f = np.nan

            results.append({
                "Ticker": t,
                "Pearson": float(pearson),
                "Pearson_P": float(pearson_p) if not pd.isna(pearson_p) else np.nan,
                "Spearman": float(spearman),
                "Spearman_P": float(spearman_p) if not pd.isna(spearman_p) else np.nan,
                "Granger_F": float(granger_f) if not pd.isna(granger_f) else np.nan,
                "Granger_P": float(granger_p) if not pd.isna(granger_p) else np.nan,
                "N": int(len(aligned)),
                "Horizon": "Next-day" if use_next_day else "Same-day",
            })
        except Exception:
            continue

    return pd.DataFrame(results)

# =========================
# Sidebar / UI Controls (UPGRADED)
# =========================
with st.sidebar:
    tickers_input = st.text_input("Portfolio holdings (comma-separated, e.g., AAPL,MSFT,NVDA)", "AAPL,MSFT,NVDA")
    subs = st.text_input("Subreddits (comma-separated)", "wallstreetbets,stocks,investing")
    query = st.text_input("Ticker search (optional e.g., NVDA)", "")

    # New filters
    min_upvotes  = st.number_input("Min upvotes (filter posts)", min_value=0, max_value=100000, value=0, step=10)
    min_comments = st.number_input("Min comments (filter posts)", min_value=0, max_value=100000, value=0, step=5)

    date_range = st.date_input(
        "Date range",
        value=(dt.date.today() - dt.timedelta(days=90), dt.date.today()),
        help="Posts will be filtered by creation time within this range."
    )

    # Benchmarks & rebalance
    primary_bench = st.selectbox("Primary benchmark (for CAPM)", ["SPY","DIA","QQQ","ACWI"], index=0)
    extra_benches = st.multiselect("Extra benchmarks to plot", ["DIA","QQQ","ACWI"], default=[])
    rebalance = st.selectbox("Rebalance frequency", ["Daily","Weekly","Monthly","Semi-annually","Annually"], index=0)

    tx_cost_bps = st.number_input("Transaction cost (bps)", min_value=0, max_value=100, value=10, step=1)
    initial_cap = st.number_input("Initial capital ($)", min_value=1000, max_value=10_000_000, value=100_000, step=1_000)
    demo_mode = st.toggle("Use demo data (no Reddit access)", value=(not has_reddit_keys()))
    run = st.button("Analyze")

tab_overview, tab_posts, tab_backtest = st.tabs(["Overview","Top Posts","Backtest"])

# =========================
# Main Flow
# =========================
if run:
    # Parse tickers from input
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if not tickers:
        st.warning("Please enter at least one ticker.")
        st.stop()

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # inclusive
    start_ts = start_date.timestamp()
    end_ts = end_date.timestamp()

    with st.spinner("Fetching posts..."):
        posts_df = fetch_posts(subs, query, start_ts, end_ts, limit_per_sub=300, demo=demo_mode)

    # Quality filters (NEW)
    if not posts_df.empty:
        posts_df = posts_df[
            (posts_df["score"].fillna(0) >= int(min_upvotes)) &
            (posts_df["num_comments"].fillna(0) >= int(min_comments))
        ]

    if posts_df.empty:
        st.info("No posts matched your filters in this window.")
        st.stop()

    # Sentiment per post
    posts_df["compound"], posts_df["label"] = zip(*posts_df.apply(
        lambda r: score_text((r["title"] or "") + "\n" + (r["selftext"] or "")), axis=1
    ))
    posts_df["created"] = pd.to_datetime(posts_df["created_utc"], unit="s")
    posts_df["url"] = posts_df["permalink"].apply(
        lambda p: ("https://www.reddit.com" + str(p).rstrip("/")) if isinstance(p, str) else ""
    )

    # Build per-ticker daily sentiment series
    daily = []
    # Use business days to better match market calendar
    date_index = pd.bdate_range(start=start_date, end=end_date, inclusive="left")
    
    # If query (ticker search) is provided, Overview/Top Posts focus on that stock's posts
    overview_posts_df = posts_df
    if query.strip():
        query_upper = query.strip().upper()
        overview_posts_df = posts_df[
            posts_df["title"].str.contains(query_upper, case=False, na=False) | 
            posts_df["selftext"].str.contains(query_upper, case=False, na=False)
        ].copy()
        st.info(f"Overview/Top Posts filtered to '{query_upper}'")
    
    # Build sentiment time series only for portfolio holdings (tickers)
    for t in tickers:
        mask = posts_df["title"].str.contains(t, case=False, na=False) | posts_df["selftext"].str.contains(t, case=False, na=False)
        df_t = posts_df[mask].copy()
        if df_t.empty:
            s = pd.Series(index=date_index, dtype=float, name=t)
        else:
            s = df_t.set_index("created").resample("D")["compound"].mean().reindex(date_index)
            s.name = t
        daily.append(s)
    sentiment_daily = pd.concat(daily, axis=1)
    # Forward-fill gaps so sentiment aligns with trading days
    sentiment_daily = sentiment_daily.reindex(date_index).ffill()

    # ---- Overview tab ----
    with tab_overview:
        # Show which stock is being analyzed
        if query.strip():
            st.subheader(f"Sentiment Overview for {query.strip().upper()}")
        else:
            st.subheader(f"Sentiment Overview for {', '.join(tickers)}")
        
        # Overview focuses on ticker search (if provided) else all
        dist = label_distribution(overview_posts_df["label"]) if not overview_posts_df.empty else pd.Series()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Compound Sentiment", f"{posts_df['compound'].mean():.3f}")
            st.bar_chart(dist.to_frame("Count"))
        with col2:
            st.write("Posts by Day (all)")
            by_day_all = overview_posts_df.set_index("created").resample("D")["id"].count().rename("Posts") if not overview_posts_df.empty else pd.Series()
            st.line_chart(by_day_all)

        st.write("Per-Ticker Average Daily Sentiment (Portfolio holdings)")
        st.line_chart(sentiment_daily)

    # ---- Top posts tab ----
    with tab_posts:
        st.subheader("Top Posts")
        # Top posts focus on ticker search if provided
        base_df = overview_posts_df if not overview_posts_df.empty else posts_df
        top_df = base_df.sort_values(["score","num_comments"], ascending=False).head(200)
        display_cols = ["subreddit","title","score","num_comments","created","url"]
        st.dataframe(top_df[display_cols].rename(columns={"url":"Link"}), use_container_width=True)

        # Download posts CSV (NEW)
        st.download_button(
            "Download posts (CSV)",
            top_df.to_csv(index=False).encode("utf-8"),
            file_name="posts.csv",
            mime="text/csv"
        )

    # ---- Backtest tab ----
    with tab_backtest:
        st.subheader("Backtest vs Benchmarks")

        try:
            port_curve, bench_curves, port_ret, mkt_ret, bench_rets, rets = backtest(
                sentiment_daily=sentiment_daily,
                tickers=tickers,
                start=start_date,
                end=end_date,
                tx_cost_bps=int(tx_cost_bps),
                initial_cap=int(initial_cap),
                primary_bench=primary_bench,
                extra_benches=extra_benches,
                rebalance_label=rebalance,
            )

            # Plot curves
            if not port_curve.empty and not bench_curves.empty:
                # Ensure both are DataFrames with numeric data
                port_df = port_curve.to_frame("Portfolio") if isinstance(port_curve, pd.Series) else port_curve
                port_df = port_df.astype(float)  # Ensure numeric data type
                bench_curves = bench_curves.astype(float)  # Ensure numeric data type
                
                curves = pd.concat([port_df, bench_curves], axis=1)
                st.line_chart(curves)
            elif not port_curve.empty:
                # Ensure portfolio curve is numeric
                port_df = port_curve.to_frame("Portfolio") if isinstance(port_curve, pd.Series) else port_curve
                port_df = port_df.astype(float)  # Ensure numeric data type
                st.line_chart(port_df)
            else:
                st.warning("No portfolio data available for plotting.")

            # Metrics: CAPM (vs primary benchmark), Sharpe, Sortino, Max DD
            metrics = capm_metrics(port_ret, mkt_ret, rf_daily=0.0) or {}
            sharpe, sortino = sharpe_sortino(port_ret, rf_daily=0.0)
            mdd, dd_series = max_drawdown(port_curve)

            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            if metrics:
                mcol1.metric("Alpha (annualized)", f"{metrics.get('alpha_annual', np.nan)*100:.2f}%")
                mcol2.metric(f"Beta (vs {primary_bench})", f"{metrics.get('beta', np.nan):.2f}")
                mcol3.metric("R²", f"{metrics.get('r2', np.nan):.3f}")
            mcol4.metric("Sharpe (ann.)", f"{sharpe:.2f}")
            mcol5.metric("Max Drawdown", f"{mdd*100:.1f}%")

            with st.expander("Show drawdown chart"):
                st.line_chart(dd_series.rename("Drawdown"))

            # === Sentiment vs Returns Correlation Analysis (Portfolio holdings) ===
            st.subheader("📊 Sentiment vs Returns Correlation (Portfolio holdings)")
            st.write("Analyzing whether Reddit sentiment predicts daily stock returns...")

            if not rets.empty:
                # Next-day (predictive) and Same-day correlations
                corr_next = sentiment_return_correlation(sentiment_daily, rets, tickers, use_next_day=True)
                corr_same = sentiment_return_correlation(sentiment_daily, rets, tickers, use_next_day=False)

                def _show_corr(df: pd.DataFrame, title: str):
                    if df.empty:
                        st.info(f"{title}: Not enough overlapping data.")
                        return
                    show = df.copy()
                    for c in ["Pearson","Spearman","Pearson_P","Spearman_P","Granger_F","Granger_P"]:
                        if c in show.columns:
                            show[c] = pd.to_numeric(show[c], errors="coerce")
                    
                    # Create a more attractive display
                    st.markdown(f"### 📊 {title}")
                    
                    # Summary statistics
                    if not show.empty:
                        avg_pearson = show["Pearson"].mean()
                        avg_spearman = show["Spearman"].mean()
                        significant_corr = len(show[show["Pearson_P"] < 0.05])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Pearson", f"{avg_pearson:.3f}")
                        with col2:
                            st.metric("Avg Spearman", f"{avg_spearman:.3f}")
                        with col3:
                            st.metric("Significant Correlations", f"{significant_corr}/{len(show)}")
                    
                    # Correlation table with better formatting
                    corr_display = show[["Ticker","Horizon","Pearson","Spearman","Pearson_P","Spearman_P","N"]].copy()
                    
                    # Add significance indicators
                    corr_display["Pearson_Sig"] = corr_display["Pearson_P"].apply(
                        lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else ""
                    )
                    corr_display["Spearman_Sig"] = corr_display["Spearman_P"].apply(
                        lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else ""
                    )
                    
                    # Rename columns for better display
                    corr_display = corr_display.rename(columns={
                        "Pearson": "Pearson r",
                        "Spearman": "Spearman ρ", 
                        "Pearson_P": "P-value",
                        "Spearman_P": "P-value",
                        "N": "Samples"
                    })
                    
                    st.dataframe(
                        corr_display[["Ticker","Horizon","Pearson r","Pearson_Sig","Spearman ρ","Spearman_Sig","Samples"]]
                            .sort_values(["Horizon","Ticker"]).style.format({
                                "Pearson r":"{:.3f}",
                                "Spearman ρ":"{:.3f}",
                            }),
                        use_container_width=True
                    )
                    
                    # Show Granger causality table
                    if "Granger_F" in show.columns and "Granger_P" in show.columns:
                        granger_df = show[["Ticker","Horizon","Granger_F","Granger_P","N"]].copy()
                        granger_df = granger_df[granger_df["Granger_F"].notna()].sort_values(["Horizon","Ticker"])
                        
                        if not granger_df.empty:
                            st.markdown("### 🔮 Granger Causality Test")
                            st.write("*Does sentiment lead returns?*")
                            
                            # Add significance indicators for Granger
                            granger_df["Significance"] = granger_df["Granger_P"].apply(
                                lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else ""
                            )
                            
                            granger_display = granger_df.rename(columns={
                                "Granger_F": "F-statistic",
                                "Granger_P": "P-value",
                                "N": "Samples"
                            })
                            
                            st.dataframe(
                                granger_display[["Ticker","Horizon","F-statistic","P-value","Significance","Samples"]]
                                    .style.format({
                                        "F-statistic":"{:.3f}",
                                        "P-value":"{:.3f}",
                                    }),
                                use_container_width=True
                            )
                        else:
                            st.info("Not enough data for Granger causality tests.")
                
                # Show interpretation guide once for all tests
                with st.expander("📚 Statistical Test Interpretation Guide"):
                    st.markdown("""
                    ### **Correlation Tests**
                    
                    **Pearson Correlation (r):**
                    - Measures linear relationship strength (-1 to +1)
                    - **r > 0.3**: Moderate positive correlation
                    - **r > 0.5**: Strong positive correlation
                    - **r < -0.3**: Moderate negative correlation
                    - **r < -0.5**: Strong negative correlation
                    
                    **Spearman Correlation (ρ):**
                    - Measures monotonic relationship (less sensitive to outliers)
                    - Same interpretation as Pearson but for non-linear relationships
                    
                    **Significance Levels:**
                    - **\***: P < 0.05 (significant)
                    - **\***: P < 0.01 (highly significant)  
                    - **\****: P < 0.001 (very highly significant)
                    
                    ### **Granger Causality Test**
                    
                    **What it tests:** Does past sentiment help predict current returns beyond what past returns alone can predict?
                    
                    **F-statistic:** Higher values = stronger evidence of causality
                    **P-value interpretation:**
                    - **P < 0.05**: Sentiment significantly Granger-causes returns (predictive power)
                    - **P < 0.01**: Strong evidence of predictive relationship
                    - **P < 0.001**: Very strong evidence of predictive relationship
                    - **P ≥ 0.05**: No significant predictive relationship found
                    """)

                _show_corr(corr_next, "Sentiment vs Returns (Next-day)")
                _show_corr(corr_same, "Sentiment vs Returns (Same-day)")
            else:
                st.info("Not enough price data to compute correlations.")

            # Download curves CSV (NEW)
            if 'curves' in locals() and not curves.empty:
                st.download_button(
                    "Download backtest curves (CSV)",
                    curves.to_csv(index=True).encode("utf-8"),
                    file_name="curves.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Backtest failed: {e}")

else:
    st.info("Set your inputs in the sidebar and click **Analyze** to run.")
