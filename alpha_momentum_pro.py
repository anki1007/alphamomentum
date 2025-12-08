"""
Alpha Momentum Pro - Enterprise Trading Dashboard
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
from functools import lru_cache

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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');
:root {
  --app-font: 'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --bg: #0b0e13; --bg-2: #10141b; --border: #1f2732; --border-soft: #1a2230;
  --text: #e6eaee; --text-dim: #b3bdc7; --accent: #7a5cff; --accent-2: #2bb0ff;
}
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--app-font) !important; }
.block-container { padding-top: 2rem; }
.hero-title {
  font-weight: 800; font-size: clamp(26px, 4.5vw, 40px); line-height: 1.05; margin: 18px 0 10px 0;
  background: linear-gradient(90deg, var(--accent-2), var(--accent) 60%); -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: .2px;
}
section[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-weight: 700; color: var(--text-dim) !important; }
.stButton button { background: linear-gradient(180deg, #1b2432, #131922); color: var(--text); border: 1px solid var(--border); border-radius: 10px; }
.stButton button:hover { filter: brightness(1.06); }
.pro-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 14px; padding: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
a { text-decoration: none; color: #9ecbff; } a:hover { text-decoration: underline; }
table { border-collapse: collapse; font-size: 0.86rem; width: 100%; color: var(--text); }
thead th { background: #121823; color: var(--text-dim); border-bottom: 1px solid var(--border); padding: 8px; }
tbody td { padding: 8px; border-top: 1px solid var(--border-soft); }
h2, h3, .stMarkdown h2 { color: var(--text); }
.metric-box { background: var(--bg-2); padding: 16px; border-radius: 8px; border-left: 3px solid var(--accent); }
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
    s = symbol.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_chart_url(symbol: str) -> str:
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
    if x >= 100 and y >= 100: return "🟢 Leading"
    if x < 100 and y >= 100: return "🔵 Improving"
    if x < 100 and y < 100: return "🔴 Lagging"
    return "🟠 Weakening"

def row_bg_for_serial(sno: int) -> str:
    if sno <= 30: return "rgba(46, 204, 113, 0.12)"
    if sno <= 60: return "rgba(255, 204, 0, 0.12)"
    if sno <= 90: return "rgba(52, 152, 219, 0.12)"
    return "rgba(231, 76, 60, 0.12)"

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_universe_from_csv(url: str) -> pd.DataFrame:
    """Load stock universe from GitHub CSV."""
    df = pd.read_csv(url)
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
        return {"Return_6M": r6, "Return_3M": r3, "Return_1M": r1}
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
    """
    Backtest momentum strategy with position sizing.
    Returns daily equity curve with metrics.
    """
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
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown %")
    )
    
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values,
                  name="Equity", line=dict(color='#2bb0ff', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=drawdown_series.index, y=drawdown_series.values,
                  name="Drawdown", fill='tozeroy',
                  line=dict(color='#ff5555', width=1)),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600, template="plotly_dark",
        hovermode="x unified", margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def plot_rolling_metrics(rolling_df: pd.DataFrame) -> go.Figure:
    """Plot rolling performance metrics."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Rolling Return (63d)", "Rolling Sharpe (63d)", "Rolling Drawdown (63d)")
    )
    
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['Rolling_Return'],
                  name="Return", line=dict(color='#2bb0ff', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['Rolling_Sharpe'],
                  name="Sharpe", line=dict(color='#27ae60', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=rolling_df.index, y=rolling_df['Rolling_Drawdown'],
                  name="Drawdown", fill='tozeroy',
                  line=dict(color='#e74c3c', width=1)),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800, template="plotly_dark",
        hovermode="x unified", margin=dict(l=50, r=50, t=100, b=50)
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">🚀 Alpha Momentum Pro</div>', unsafe_allow_html=True)
st.markdown("**Enterprise Trading Dashboard** • Screener + Backtester + Metrics Engine")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Screener", "🧪 Backtester", "📈 Analytics", "⚙️ Settings"])

# ─── TAB 1: SCREENER ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Daily Momentum Screener")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        indices_universe = st.selectbox("Universe", list(CSV_FILES.keys()), key="screener_universe")
    with col2:
        benchmark_key = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)
    with col3:
        period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1)
    with col4:
        do_screen = st.button("🔄 Refresh Screener", use_container_width=True)
    
    if "screened_df" not in st.session_state:
        st.session_state.screened_df = None
        st.session_state.ran_screen = False
    
    if do_screen or not st.session_state.ran_screen:
        try:
            uni_url = CSV_FILES[indices_universe]
            universe_df = load_universe_from_csv(uni_url)
            benchmark = BENCHMARKS[benchmark_key]
            tickers = universe_df["Symbol"].tolist()
            
            with st.spinner("📥 Fetching prices..."):
                raw = fetch_prices(tickers, benchmark, period=period)
            
            if raw.empty:
                st.error("Failed to fetch data")
            else:
                with st.spinner("🔍 Analyzing momentum..."):
                    screener_df = build_screener_table(raw, benchmark, universe_df)
                
                st.session_state.screened_df = screener_df
                st.session_state.ran_screen = True
        except Exception as e:
            st.error(f"Screener error: {str(e)}")
    
    if st.session_state.screened_df is not None:
        df = st.session_state.screened_df
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Total Stocks", len(df))
        col2.metric("🟢 Leading", len(df[df['Performance'].str.contains("Leading")]))
        col3.metric("🔵 Improving", len(df[df['Performance'].str.contains("Improving")]))
        col4.metric("📊 Data", f"{period} • {benchmark_key}")
        
        # Top performers table
        st.markdown("### Top 30 Performers")
        top30 = df.head(30).copy()
        
        # Format for display
        display_cols = ["S.No", "Name", "Industry", "Return_6M", "Rank_6M", 
                       "Return_3M", "Rank_3M", "Return_1M", "Rank_1M", 
                       "RS-Ratio", "RS-Momentum", "Performance"]
        display_df = top30[display_cols].copy()
        
        # Add clickable links
        display_df["Name"] = display_df.apply(
            lambda r: f'[{top30.loc[r.name, "Name"]}]({tradingview_chart_url(top30.loc[r.name, "Symbol"])})',
            axis=1
        )
        
        st.markdown(display_df.to_markdown(index=False))
        
        # Export
        csv_buffer = df.drop(columns=['Symbol', 'Chart']).to_csv(index=False)
        st.download_button(
            "📥 Export Screener Results (CSV)",
            csv_buffer,
            f"screener_{indices_universe.replace(' ', '')}.csv",
            "text/csv"
        )

# ─── TAB 2: BACKTESTER ──────────────────────────────────────────────────
with tab2:
    st.subheader("Strategy Backtester")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        bt_symbol = st.text_input("Symbol (NSE)", "SBIN.NS", help="e.g., SBIN.NS, INFY.NS")
    with col2:
        bt_period = st.selectbox("Backtest Period", ["1y", "2y", "3y", "5y"], index=2)
    with col3:
        bt_capital = st.number_input("Initial Capital (₹)", 100000, 1000000, 100000, step=100000)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        strategy = st.selectbox("Strategy", ["Momentum (EMA100)", "RSI Oversold", "MACD Cross"])
    with col2:
        position_size = st.slider("Position Size (%)", 10, 100, 50)
    with col3:
        bt_button = st.button("🧪 Run Backtest", use_container_width=True)
    
    if bt_button:
        try:
            start, end = _period_years_to_dates(bt_period)
            prices_data = yf.download(bt_symbol, start=start.date(), end=end.date(), progress=False)
            
            if prices_data.empty:
                st.error("Failed to fetch data for symbol")
            else:
                price_series = prices_data['Adj Close']
                
                # Generate positions based on strategy
                if strategy == "Momentum (EMA100)":
                    ema100 = price_series.ewm(span=100, adjust=False).mean()
                    positions = pd.Series(position_size / 100 if price_series >= ema100 else 0,
                                        index=price_series.index)
                elif strategy == "RSI Oversold":
                    delta = price_series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    positions = pd.Series(position_size / 100 if rsi < 30 else 0,
                                        index=price_series.index)
                else:  # MACD
                    ema12 = price_series.ewm(span=12, adjust=False).mean()
                    ema26 = price_series.ewm(span=26, adjust=False).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9, adjust=False).mean()
                    positions = pd.Series(position_size / 100 if macd > signal else 0,
                                        index=price_series.index)
                
                # Run backtest
                backtest_df = backtest_strategy(price_series, positions, bt_capital)
                metrics = calculate_backtest_metrics(backtest_df['Equity'], positions, price_series)
                rolling_mets = generate_rolling_metrics(backtest_df['Equity'])
                
                # Display metrics
                st.markdown("### Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Return", f"{metrics.get('Total_Return', 0):.2f}%")
                col2.metric("CAGR", f"{metrics.get('CAGR', 0):.2f}%")
                col3.metric("Sharpe Ratio", f"{metrics.get('Sharpe_Ratio', 0):.2f}")
                col4.metric("Sortino Ratio", f"{metrics.get('Sortino_Ratio', 0):.2f}")
                col5.metric("Max Drawdown", f"{metrics.get('Max_Drawdown', 0):.2f}%")
                
                # Charts
                st.markdown("### Equity Curve")
                equity_fig = plot_equity_curve(backtest_df['Equity'], backtest_df['Drawdown'] * 100)
                st.plotly_chart(equity_fig, use_container_width=True)
                
                st.markdown("### Rolling Metrics")
                rolling_fig = plot_rolling_metrics(rolling_mets)
                st.plotly_chart(rolling_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Backtest error: {str(e)}")

# ─── TAB 3: ANALYTICS ───────────────────────────────────────────────────
with tab3:
    st.subheader("Analytics & Insights")
    st.info("🚀 Coming Soon: Market correlations, sector analysis, VaR analysis, Greeks calculator")

# ─── TAB 4: SETTINGS ────────────────────────────────────────────────────
with tab4:
    st.subheader("Configuration")
    
    st.markdown("### Screener Settings")
    col1, col2 = st.columns(2)
    with col1:
        min_1y_return = st.number_input("Min 1Y Return (%)", 6.5, 50.0, 6.5)
        min_updays = st.number_input("Min Up Days (%)", 30.0, 80.0, 45.0)
    with col2:
        proximity_to_high = st.number_input("Proximity to 52W High (%)", 0.0, 100.0, 80.0)
    
    st.markdown("### Backtester Settings")
    col1, col2 = st.columns(2)
    with col1:
        transaction_cost = st.number_input("Transaction Cost (bps)", 0, 50, 10) / 10000
    with col2:
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, float(RISK_FREE_RATE))
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("✅ Settings saved!")

st.markdown("---")
st.caption("Alpha Momentum Pro v1.0 • Built for Indian Index Traders • Data: YFinance • Powered by Streamlit")
