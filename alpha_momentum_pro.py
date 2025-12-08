""" 
Alpha Momentum Pro - FIXED VERSION
Enterprise Trading Dashboard
Comprehensive screener + backtester + metrics engine for Indian indices
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  # ✅ FIXED: Added missing import
from functools import lru_cache
import io

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Alpha Momentum Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .metric-card { background-color: #1f2937; padding: 20px; border-radius: 8px; }
        .green-text { color: #10b981; font-weight: bold; }
        .red-text { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════
BENCHMARKS: Dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "Nifty 100": "^CNX100",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 150": "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250": "^NIFTYSMLCAP250.NS",
}

GITHUB_BASE = "https://raw.githubusercontent.com/anki1007/alphamomentum/main/"
CSV_FILES: Dict[str, str] = {
    "Nifty 50": GITHUB_BASE + "nifty50.csv",
    "Nifty 100": GITHUB_BASE + "nifty100.csv",
    "Nifty 200": GITHUB_BASE + "nifty200.csv",
    "Nifty 500": GITHUB_BASE + "nifty500.csv",
    "Nifty Midcap 150": GITHUB_BASE + "niftymidcap150.csv",
    "Nifty Mid Small 400": GITHUB_BASE + "niftymidsmallcap400.csv",
    "Nifty Smallcap 250": GITHUB_BASE + "niftysmallcap250.csv",
    "Nifty Total Market": GITHUB_BASE + "niftytotalmarket.csv",
}

RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21
RISK_FREE_RATE = 6.5  # India 10-year G-sec

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def tv_symbol_from_yf(symbol: str) -> str:
    """Convert Yahoo Finance symbol to TradingView format."""
    s = symbol.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_chart_url(symbol: str) -> str:
    """Generate TradingView chart URL."""
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol_from_yf(symbol)}"

def _pick_close(df: pd.DataFrame | pd.Series, symbol: str) -> pd.Series:
    """Extract close price from yfinance data."""
    if isinstance(df, pd.Series):
        return pd.to_numeric(df, errors="coerce").dropna()
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in ("Close", "Adj Close"):
            col = (symbol, lvl)
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").dropna()
        return pd.Series(dtype=float)
    else:
        for col in ("Close", "Adj Close"):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").dropna()
        return pd.Series(dtype=float)

def jdk_components(price: pd.Series, bench: pd.Series, win: int = JDK_WINDOW) -> Tuple[pd.Series, pd.Series]:
    """Calculate RS-Ratio and RS-Momentum (Jdki Dynamic Kelly components)."""
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    rs = 100 * (df["p"] / df["b"])
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs - m) / s).dropna()
    
    rroc = rs_ratio.pct_change().mul(100)
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom = (101 + (rroc - m2) / s2).dropna()
    
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def perf_quadrant(x: float, y: float) -> str:
    """Classify stock into performance quadrant."""
    if x >= 100 and y >= 100:
        return "🟢 Leading"
    if x < 100 and y >= 100:
        return "🔵 Improving"
    if x < 100 and y < 100:
        return "🔴 Lagging"
    return "🟠 Weakening"

def row_bg_for_serial(sno: int) -> str:
    """Return background color for ranking tier."""
    if sno <= 30:
        return "rgba(46, 204, 113, 0.12)"
    if sno <= 60:
        return "rgba(255, 204, 0, 0.12)"
    if sno <= 90:
        return "rgba(52, 152, 219, 0.12)"
    return "rgba(231, 76, 60, 0.12)"

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """✅ FIXED: Convert DataFrame to markdown without tabulate dependency."""
    if df.empty:
        return "No data available"
    
    # Get column names
    cols = df.columns.tolist()
    
    # Header row
    md = "| " + " | ".join(str(c) for c in cols) + " |\n"
    md += "|" + "|".join(["---"] * len(cols)) + "|\n"
    
    # Data rows
    for _, row in df.iterrows():
        values = []
        for val in row:
            if pd.isna(val):
                values.append("")
            elif isinstance(val, float):
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        md += "| " + " | ".join(values) + " |\n"
    
    return md

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_universe_from_csv(url: str) -> pd.DataFrame:
    """Load stock universe from GitHub CSV."""
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()
    
    cols = {c.strip().lower(): c for c in df.columns}
    required = ["symbol", "company name", "industry"]
    
    for n in required:
        if n not in cols:
            raise ValueError(f"CSV must include: {required}. Missing: {n}")
    
    df = df[[cols["symbol"], cols["company name"], cols["industry"]]].copy()
    df.columns = ["Symbol", "Name", "Industry"]
    df = df.dropna(subset=["Symbol"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    
    return df

def _period_years_to_dates(period: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Convert period string to start/end dates."""
    years_map = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}
    years = years_map.get(period, 2)
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    end = today_ist + pd.Timedelta(days=1)
    start = today_ist - pd.DateOffset(years=years)
    return start, end

@st.cache_data(show_spinner=True)
def fetch_prices(tickers: List[str], benchmark: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    start, end = _period_years_to_dates(period)
    
    try:
        data = yf.download(
            tickers + [benchmark],
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as e:
        msg = str(e)
        if "Rate limited" in msg or "Too Many Requests" in msg:
            st.warning("Yahoo Finance rate limited. Please try again shortly.")
        else:
            st.error(f"Data fetch failed: {e}")
        return pd.DataFrame()
    
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            pass
    
    return data

# ═══════════════════════════════════════════════════════════════════════════
# SCREENER ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def analyze_momentum(adj: pd.Series) -> Optional[dict]:
    """Check if stock passes momentum filter."""
    if adj is None or adj.empty or len(adj) < 252:
        return None
    
    ema100 = adj.ewm(span=100, adjust=False).mean()
    
    try:
        one_year_return = (adj.iloc[-1] / adj.iloc[-252] - 1.0) * 100.0
    except Exception:
        return None
    
    high_52w = adj.iloc[-252:].max()
    within_20pct_high = adj.iloc[-1] >= high_52w * 0.8
    
    if len(adj) < 126:
        return None
    
    six_month = adj.iloc[-126:]
    up_days_pct = (six_month.pct_change() > 0).sum() / len(six_month) * 100.0
    
    if (adj.iloc[-1] >= ema100.iloc[-1] and one_year_return >= 6.5 and 
        within_20pct_high and up_days_pct > 45.0):
        try:
            r6 = (adj.iloc[-1] / adj.iloc[-126] - 1.0) * 100.0
            r3 = (adj.iloc[-1] / adj.iloc[-63] - 1.0) * 100.0
            r1 = (adj.iloc[-1] / adj.iloc[-21] - 1.0) * 100.0
        except Exception:
            return None
        
        return {
            "Return_6M": r6,
            "Return_3M": r3,
            "Return_1M": r1
        }
    
    return None

def build_screener_table(raw: pd.DataFrame, benchmark: str, universe_df: pd.DataFrame) -> pd.DataFrame:
    """Build full screener results table."""
    bench = _pick_close(raw, benchmark).dropna()
    
    if bench.empty:
        raise RuntimeError(f"Benchmark {benchmark} data empty.")
    
    cutoff = bench.index.max() - pd.Timedelta(days=RS_LOOKBACK_DAYS + 5)
    bench_rs = bench.loc[bench.index >= cutoff].copy()
    
    rows = []
    
    for _, rec in universe_df.iterrows():
        sym, name, industry = rec.Symbol, rec.Name, rec.Industry
        s = _pick_close(raw, sym).dropna()
        
        if s.empty or analyze_momentum(s) is None:
            continue
        
        s_rs = s.loc[s.index >= cutoff].copy()
        rr, mm = jdk_components(s_rs, bench_rs)
        
        if rr.empty or mm.empty:
            continue
        
        ix = rr.index.intersection(mm.index)
        
        rows.append({
            "Name": name,
            "Industry": industry,
            "Return_6M": float((s.iloc[-1] / s.iloc[-126] - 1) * 100) if len(s) >= 126 else np.nan,
            "Rank_6M": np.nan,
            "Return_3M": float((s.iloc[-1] / s.iloc[-63] - 1) * 100) if len(s) >= 63 else np.nan,
            "Rank_3M": np.nan,
            "Return_1M": float((s.iloc[-1] / s.iloc[-21] - 1) * 100) if len(s) >= 21 else np.nan,
            "Rank_1M": np.nan,
            "RS-Ratio": float(rr.loc[ix].iloc[-1]),
            "RS-Momentum": float(mm.loc[ix].iloc[-1]),
            "Performance": perf_quadrant(float(rr.loc[ix].iloc[-1]), float(mm.loc[ix].iloc[-1])),
            "Symbol": sym,
            "Chart": tradingview_chart_url(sym),
        })
    
    if not rows:
        raise RuntimeError("No tickers passed filters. Try longer period or adjust thresholds.")
    
    df = pd.DataFrame(rows)
    
    for c in ("Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min").astype("Int64")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min").astype("Int64")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min").astype("Int64")
    df["Final_Rank"] = (df["Rank_6M"].fillna(0) + df["Rank_3M"].fillna(0) + df["Rank_1M"].fillna(0)).astype("Int64")
    
    df = df.sort_values("Final_Rank", kind="mergesort").reset_index(drop=True)
    df.insert(0, "S.No", np.arange(1, len(df) + 1, dtype=int))
    
    order = ["S.No", "Name", "Industry", "Return_6M", "Rank_6M", "Return_3M", "Rank_3M", 
             "Return_1M", "Rank_1M", "RS-Ratio", "RS-Momentum", "Performance", "Final_Rank", "Chart", "Symbol"]
    
    return df[order]

# ═══════════════════════════════════════════════════════════════════════════
# BACKTESTER ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def backtest_strategy(prices: pd.Series, positions: pd.Series, initial_capital: float = 100000, 
                     transaction_cost: float = 0.001) -> pd.DataFrame:
    """Backtest momentum strategy with position sizing."""
    if prices.empty or positions.empty:
        return pd.DataFrame()
    
    # Align data
    common_idx = prices.index.intersection(positions.index)
    prices = prices.loc[common_idx]
    positions = positions.loc[common_idx]
    
    returns = prices.pct_change()
    position_returns = positions.shift(1) * returns
    position_returns.iloc[0] = 0
    
    # Apply transaction costs
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * transaction_cost
    net_returns = position_returns - transaction_costs
    
    # Equity curve
    cumulative_returns = (1 + net_returns).cumprod()
    equity = initial_capital * cumulative_returns
    
    # Drawdown calculation
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    
    return pd.DataFrame({
        'Date': prices.index,
        'Price': prices.values,
        'Position': positions.values,
        'Daily_Return': returns.values,
        'Strategy_Return': net_returns.values,
        'Equity': equity.values,
        'Drawdown': drawdown.values,
    }).set_index('Date')

def calculate_backtest_metrics(equity_curve: pd.Series, positions: pd.Series, 
                              prices: pd.Series, period_days: int = 252) -> Dict[str, float]:
    """Calculate comprehensive backtest metrics."""
    if equity_curve.empty:
        return {}
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    days = len(equity_curve) / 365
    cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(days, 1)) - 1) * 100
    
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else daily_returns.std()
    sortino = np.sqrt(252) * daily_returns.mean() / downside_std if downside_std > 0 else 0
    
    drawdown_series = (equity_curve / equity_curve.expanding().max() - 1) * 100
    max_dd = drawdown_series.min()
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    
    win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100 if len(daily_returns) > 0 else 0
    
    # Rolling returns
    rolling_1m = (equity_curve.iloc[-21] / equity_curve.iloc[-42] - 1) * 100 if len(equity_curve) >= 42 else 0
    rolling_3m = (equity_curve.iloc[-1] / equity_curve.iloc[-63] - 1) * 100 if len(equity_curve) >= 63 else 0
    rolling_6m = (equity_curve.iloc[-1] / equity_curve.iloc[-126] - 1) * 100 if len(equity_curve) >= 126 else 0
    
    return {
        'Total_Return': total_return,
        'CAGR': cagr,
        'Sharpe_Ratio': sharpe,
        'Sortino_Ratio': sortino,
        'Calmar_Ratio': calmar,
        'Max_Drawdown': max_dd,
        'Win_Rate': win_rate,
        'Rolling_1M': rolling_1m,
        'Rolling_3M': rolling_3m,
        'Rolling_6M': rolling_6m,
    }

def generate_rolling_metrics(equity_curve: pd.Series, window: int = 63) -> pd.DataFrame:
    """Generate rolling performance metrics (Sharpe, returns, etc.)."""
    rolling_returns = equity_curve.pct_change()
    rolling_sharpe = rolling_returns.rolling(window).mean() / rolling_returns.rolling(window).std() * np.sqrt(252)
    rolling_ret = (equity_curve.rolling(window).apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100, raw=False))
    rolling_dd = (equity_curve.rolling(window).apply(lambda x: ((x / x.expanding().max() - 1) * 100).min(), raw=False))
    
    return pd.DataFrame({
        'Rolling_Sharpe': rolling_sharpe,
        'Rolling_Return': rolling_ret,
        'Rolling_Drawdown': rolling_dd,
    })

# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
def plot_equity_curve(equity_curve: pd.Series, drawdown_series: pd.Series) -> go.Figure:
    """Plot equity curve with drawdown overlay."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown %")
    )
    
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Equity",
                  line=dict(color='#2bb0ff', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=drawdown_series.index, y=drawdown_series.values, name="Drawdown",
                  fill='tozeroy', line=dict(color='#ff5555', width=1)),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def plot_rolling_metrics(rolling_df: pd.DataFrame) -> go.Figure:
    """Plot rolling performance metrics."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Rolling Return (63d)", "Rolling Sharpe (63d)", "Rolling Drawdown (63d)")
    )
    
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['Rolling_Return'], name="Return",
                  line=dict(color='#2bb0ff', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['Rolling_Sharpe'], name="Sharpe",
                  line=dict(color='#27ae60', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['Rolling_Drawdown'], name="Drawdown",
                  fill='tozeroy', line=dict(color='#e74c3c', width=1)),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

st.title("📊 Alpha Momentum Pro")
st.markdown("Enterprise Screener + Backtester for Indian Equities")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    universe_choice = st.selectbox(
        "Select Universe",
        options=list(CSV_FILES.keys()),
        index=0
    )
    
    benchmark_choice = st.selectbox(
        "Select Benchmark",
        options=list(BENCHMARKS.keys()),
        index=0
    )
    
    period_choice = st.selectbox(
        "Historical Period",
        options=["1y", "2y", "3y", "5y"],
        index=1
    )
    
    portfolio_size = st.slider(
        "Portfolio Size (Top N stocks)",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    run_screener = st.button("🚀 Run Screener", use_container_width=True)

if run_screener:
    with st.spinner("Loading universe..."):
        universe_df = load_universe_from_csv(CSV_FILES[universe_choice])
        st.success(f"✅ Loaded {len(universe_df)} stocks")
    
    with st.spinner("Fetching price data..."):
        raw_data = fetch_prices(
            universe_df["Symbol"].tolist(),
            BENCHMARKS[benchmark_choice],
            period_choice
        )
        
        if raw_data.empty:
            st.error("Failed to fetch data")
            st.stop()
    
    with st.spinner("Building screener..."):
        screener_df = build_screener_table(raw_data, BENCHMARKS[benchmark_choice], universe_df)
        
        # Select top portfolio_size stocks
        display_df = screener_df.head(portfolio_size).copy()
    
    # Display results
    st.subheader(f"📈 Top {portfolio_size} Momentum Stocks")
    
    # ✅ FIXED: Use custom markdown converter instead of df.to_markdown()
    markdown_table = dataframe_to_markdown(display_df.drop(columns=["Symbol", "Chart"], errors="ignore"))
    st.markdown(markdown_table)
    
    # CSV Export ✅ FIXED: Use proper pandas method
    st.subheader("📥 Export Results")
    
    csv_buffer = io.StringIO()
    export_df = display_df.drop(columns=["Chart"], errors="ignore")
    export_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name=f"momentum_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Statistics
    st.subheader("📊 Portfolio Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_return_6m = display_df["Return_6M"].mean()
        st.metric("Avg 6M Return", f"{avg_return_6m:.2f}%")
    
    with col2:
        avg_rs_ratio = display_df["RS-Ratio"].mean()
        st.metric("Avg RS-Ratio", f"{avg_rs_ratio:.2f}")
    
    with col3:
        leading_count = (display_df["Performance"] == "🟢 Leading").sum()
        st.metric("Leading Stocks", leading_count)
    
    with col4:
        lagging_count = (display_df["Performance"] == "🔴 Lagging").sum()
        st.metric("Lagging Stocks", lagging_count)
    
    # Charts
    st.subheader("📉 Performance Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ret = px.bar(
            display_df.head(15),
            x="Name",
            y="Return_6M",
            title="Top 15 - 6M Returns",
            color="Return_6M",
            color_continuous_scale="RdYlGn"
        )
        fig_ret.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col2:
        fig_quad = px.scatter(
            display_df,
            x="RS-Ratio",
            y="RS-Momentum",
            color="Performance",
            size="Return_6M",
            hover_name="Name",
            title="RS-Ratio vs RS-Momentum Quadrant"
        )
        fig_quad.axhline(y=100, line_dash="dash", line_color="gray")
        fig_quad.axvline(x=100, line_dash="dash", line_color="gray")
        fig_quad.update_layout(height=400)
        st.plotly_chart(fig_quad, use_container_width=True)
    
    st.info("✅ Screener completed successfully!")
else:
    st.info("👈 Configure settings and click 'Run Screener' to get started")
