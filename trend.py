import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaMomentum — NSE Scanner",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:       #080c10;
    --bg2:      #0d1520;
    --bg3:      #111d2e;
    --border:   #1e3a5f;
    --cyan:     #00d4ff;
    --gold:     #f5a623;
    --green:    #00e676;
    --red:      #ff4d6d;
    --muted:    #4a7090;
    --text:     #cde4f5;
}

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
}

/* Animated grid background */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(var(--border) 1px, transparent 1px),
        linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.18;
    pointer-events: none;
    z-index: 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Space Mono', monospace !important; }

/* All text */
p, label, span, div { color: var(--text) !important; font-family: 'Space Mono', monospace !important; }

/* Headers */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    letter-spacing: -1px !important;
    background: linear-gradient(135deg, var(--cyan) 0%, #7b8fff 60%, var(--gold) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0 !important;
}
h2 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--cyan) !important;
    letter-spacing: 1px !important;
    font-size: 1.2rem !important;
    text-transform: uppercase !important;
    border-left: 3px solid var(--gold) !important;
    padding-left: 12px !important;
    margin: 1.5rem 0 1rem 0 !important;
}
h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--gold) !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 2px !important;
    padding: 0.6rem 2.5rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--cyan);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.2s;
    z-index: -1;
}
.stButton > button:hover {
    color: var(--bg) !important;
    background: var(--cyan) !important;
    box-shadow: 0 0 24px #00d4ff60 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 16px !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), transparent);
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: var(--cyan) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricDelta"] { color: var(--green) !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}

/* Inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 2px !important;
}
.stSlider [data-baseweb="slider"] { padding: 0.4rem 0 !important; }
[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--gold)) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Alerts */
[data-testid="stSuccess"] { background: #001a0a !important; border-left: 3px solid var(--green) !important; }
[data-testid="stWarning"] { background: #1a1000 !important; border-left: 3px solid var(--gold) !important; }
[data-testid="stError"]   { background: #1a0008 !important; border-left: 3px solid var(--red) !important; }
[data-testid="stInfo"]    { background: #00101a !important; border-left: 3px solid var(--cyan) !important; }

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}
.stDownloadButton > button:hover { background: var(--gold) !important; color: var(--bg) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Checkbox */
.stCheckbox label span { color: var(--text) !important; font-size: 0.78rem !important; }

/* Spinner */
[data-testid="stSpinner"] p { color: var(--cyan) !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
NIFTY_URL = "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
CACHE_TTL = 3600
PLOTLY_DARK = dict(
    plot_bgcolor="#080c10",
    paper_bgcolor="#0d1520",
    font_color="#cde4f5",
    font_family="Space Mono",
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
)

# ── DATA FETCHERS ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_list():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(NIFTY_URL, headers=headers, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_daily_data(symbol: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(f"{symbol}.NS").history(start="2019-01-01", interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_close_matrix(symbols: list, start: str = "2022-01-01") -> pd.DataFrame:
    """Fetch close prices for multiple symbols for portfolio optimization."""
    frames = {}
    for sym in symbols:
        df = fetch_daily_data(sym)
        if not df.empty:
            frames[sym] = df["Close"]
    if not frames:
        return pd.DataFrame()
    combined = pd.DataFrame(frames)
    combined = combined[combined.index >= start]
    return combined.dropna(axis=1, thresh=int(0.8 * len(combined)))

# ── INDICATORS ────────────────────────────────────────────────────────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int) -> float:
    d = s.diff()
    ag = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    al = (-d.clip(upper=0)).ewm(com=n - 1, min_periods=n).mean()
    r = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    v = r.iloc[-1]
    return round(float(v), 1) if pd.notna(v) else np.nan

def macd(s: pd.Series):
    fast, slow, sig = ema(s, 12), ema(s, 26), None
    line = fast - slow
    sig = ema(line, 9)
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = s.rolling(n).mean()
    std = s.rolling(n).std()
    return mid, mid + k * std, mid - k * std

# ── SUPPORT DETECTION ─────────────────────────────────────────────────────────
def find_support_zones(df: pd.DataFrame, order: int = 10) -> list:
    """Use scipy local minima OR fallback rolling window."""
    lows = df["Low"].values
    if HAS_SCIPY:
        idx = argrelextrema(lows, np.less_equal, order=order)[0]
    else:
        idx = []
        for i in range(order, len(lows) - order):
            window = lows[i - order:i + order + 1]
            if lows[i] == window.min():
                idx.append(i)
    levels = sorted(set(round(lows[i], 2) for i in idx))
    ltp = df["Close"].iloc[-1]
    # Keep levels below LTP + 20%
    return [l for l in levels if l < ltp * 1.20]

def nearest_support(df: pd.DataFrame, threshold_pct: float):
    zones = find_support_zones(df)
    ltp = df["Close"].iloc[-1]
    if not zones:
        return False, None, zones
    dists = [(abs(ltp - z) / z, z) for z in zones if z > 0]
    dists.sort()
    nearest_dist, nearest_lvl = dists[0]
    return nearest_dist <= threshold_pct, round(nearest_lvl, 2), zones

# ── SCAN ONE STOCK ────────────────────────────────────────────────────────────
def scan_stock(symbol: str, sector: str, threshold_pct: float) -> dict | None:
    df = fetch_daily_data(symbol)
    if df.empty or len(df) < 260:
        return None

    c = df["Close"]
    ltp = round(c.iloc[-1], 2)

    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    e20v, e50v, e200v = round(e20.iloc[-1], 2), round(e50.iloc[-1], 2), round(e200.iloc[-1], 2)
    aligned = ltp > e20v > e50v > e200v

    r14  = rsi(c, 14)
    r21  = rsi(c, 21)
    r63  = rsi(c, 63)
    r126 = rsi(c, 126)
    r252 = rsi(c, 252)

    near_sup, sup_level, _ = nearest_support(df, threshold_pct)
    pct_200 = round((ltp / e200v - 1) * 100, 2) if e200v > 0 else 0

    # 52-week high/low
    yr = df[df.index >= df.index[-1] - pd.Timedelta(days=365)]
    wk52_h = round(yr["High"].max(), 2)
    wk52_l = round(yr["Low"].min(), 2)
    pct_from_52h = round((ltp / wk52_h - 1) * 100, 2)

    # Volume ratio (20d avg vs 50d avg)
    vol_ratio = round(df["Volume"].iloc[-20:].mean() / (df["Volume"].iloc[-50:].mean() + 1), 2)

    # ATR %
    atr_val = atr(df, 14).iloc[-1]
    atr_pct = round(atr_val / ltp * 100, 2)

    _, bb_upper, bb_lower = bollinger(c)
    bb_pct = round((ltp - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-9) * 100, 1)

    macd_line, macd_sig, _ = macd(c)
    macd_bull = bool(macd_line.iloc[-1] > macd_sig.iloc[-1])

    return {
        "Symbol"        : symbol,
        "Sector"        : sector,
        "LTP"           : ltp,
        "20 EMA"        : e20v,
        "50 EMA"        : e50v,
        "200 EMA"       : e200v,
        "% vs 200EMA"   : pct_200,
        "% from 52W High": pct_from_52h,
        "EMA Aligned"   : "✓" if aligned else "✗",
        "LTP vs 200EMA" : "ABOVE" if ltp > e200v else "BELOW",
        "Near Support"  : "✓" if near_sup else "–",
        "Support Level" : sup_level if sup_level else np.nan,
        "52W High"      : wk52_h,
        "52W Low"       : wk52_l,
        "Vol Ratio"     : vol_ratio,
        "ATR %"         : atr_pct,
        "BB %"          : bb_pct,
        "MACD Bull"     : "✓" if macd_bull else "✗",
        "RSI 14"        : r14,
        "RSI 21"        : r21,
        "RSI 63"        : r63,
        "RSI 126"       : r126,
        "RSI 252"       : r252,
    }

# ── TABLE STYLING ─────────────────────────────────────────────────────────────
def style_table(df: pd.DataFrame):
    RSI_COLS = [c for c in ["RSI 14", "RSI 21", "RSI 63", "RSI 126", "RSI 252"] if c in df.columns]

    def rsi_col(v):
        try:
            v = float(v)
            if v >= 70:   return "background:#003d1a;color:#00e676;font-weight:700"
            elif v >= 50: return "background:#001f0d;color:#4ddb8a"
            elif v >= 30: return "background:#1f0008;color:#ff8fa3"
            else:         return "background:#3d0010;color:#ff4d6d;font-weight:700"
        except: return ""

    def flag_col(v):
        s = str(v)
        if s in ("✓", "ABOVE"): return "color:#00e676;font-weight:700;text-align:center"
        if s in ("✗", "BELOW"): return "color:#ff4d6d;text-align:center"
        return "color:#4a7090;text-align:center"

    def pct_col(v):
        try:
            v = float(v)
            if v > 10:   return "color:#00e676;font-weight:700"
            elif v > 0:  return "color:#4ddb8a"
            elif v > -10: return "color:#f5a623"
            else:        return "color:#ff4d6d"
        except: return ""

    flag_cols = [c for c in ["EMA Aligned", "LTP vs 200EMA", "Near Support", "MACD Bull"] if c in df.columns]
    pct_cols  = [c for c in ["% vs 200EMA", "% from 52W High"] if c in df.columns]

    try:
        styler = df.style.map(rsi_col, subset=RSI_COLS)
        if flag_cols: styler = styler.map(flag_col, subset=flag_cols)
        if pct_cols:  styler = styler.map(pct_col, subset=pct_cols)
    except TypeError:
        styler = df.style.applymap(rsi_col, subset=RSI_COLS)

    styler = styler.set_properties(**{
        "background-color": "#0d1520",
        "color": "#cde4f5",
        "border-color": "#1e3a5f",
        "font-family": "Space Mono, monospace",
        "font-size": "0.75rem",
    }).set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#111d2e"),
            ("color", "#00d4ff"),
            ("font-size", "0.68rem"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "1px"),
            ("border-bottom", "2px solid #1e3a5f"),
            ("white-space", "nowrap"),
        ]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "#00d4ff0d")]},
    ]).format(precision=2, na_rep="–")
    return styler

# ── PLOTLY CHART ──────────────────────────────────────────────────────────────
def plot_stock(symbol: str, df: pd.DataFrame):
    if not HAS_PLOTLY or df.empty:
        st.warning("Plotly not available or no data.")
        return

    c = df["Close"]
    e20_s, e50_s, e200_s = ema(c, 20), ema(c, 50), ema(c, 200)
    macd_line, macd_sig, macd_hist = macd(c)
    rsi14 = 100 - 100 / (1 + c.diff().clip(lower=0).ewm(com=13, min_periods=14).mean() /
                          (-c.diff().clip(upper=0)).ewm(com=13, min_periods=14).mean().replace(0, np.nan))
    bb_mid, bb_up, bb_lo = bollinger(c)
    _, sup_level, sup_zones = nearest_support(df, 0.05)

    # Use last 365 trading days
    df_plot = df.iloc[-365:]
    idx = df_plot.index

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("", "MACD", "RSI 14"),
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=idx, open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        increasing_line_color="#00e676", decreasing_line_color="#ff4d6d",
        increasing_fillcolor="#00e67620", decreasing_fillcolor="#ff4d6d20",
        name="Price", showlegend=False,
    ), row=1, col=1)

    # Bollinger bands
    fig.add_trace(go.Scatter(x=idx, y=bb_up.reindex(idx), line=dict(color="#f5a62330", width=1),
                              fill=None, name="BB Upper", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=bb_lo.reindex(idx), line=dict(color="#f5a62330", width=1),
                              fill="tonexty", fillcolor="#f5a62308", name="BB Lower", showlegend=False), row=1, col=1)

    # EMAs
    for val, col, lbl in [(e20_s, "#00d4ff", "EMA 20"), (e50_s, "#f5a623", "EMA 50"), (e200_s, "#ff4d6d", "EMA 200")]:
        fig.add_trace(go.Scatter(x=idx, y=val.reindex(idx), line=dict(color=col, width=1.5),
                                  name=lbl, showlegend=True), row=1, col=1)

    # Support zones (horizontal lines)
    ltp = df["Close"].iloc[-1]
    for z in sup_zones[-6:]:
        if z < ltp * 1.1:
            fig.add_hline(y=z, line=dict(color="#00d4ff40", width=1, dash="dot"),
                          annotation_text=f"S {z:.0f}", annotation_font_size=9,
                          annotation_font_color="#00d4ff80", row=1, col=1)

    # Volume bars
    colors = ["#00e67640" if df_plot["Close"].iloc[i] >= df_plot["Open"].iloc[i]
              else "#ff4d6d40" for i in range(len(df_plot))]
    fig.add_trace(go.Bar(x=idx, y=df_plot["Volume"], marker_color=colors,
                          name="Volume", showlegend=False, yaxis="y2"), row=1, col=1)

    # ── MACD ──
    hist_colors = ["#00e676" if v >= 0 else "#ff4d6d" for v in macd_hist.reindex(idx).fillna(0)]
    fig.add_trace(go.Bar(x=idx, y=macd_hist.reindex(idx), marker_color=hist_colors,
                          name="MACD Hist", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=macd_line.reindex(idx), line=dict(color="#00d4ff", width=1.5),
                              name="MACD", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=macd_sig.reindex(idx), line=dict(color="#f5a623", width=1.5),
                              name="Signal", showlegend=False), row=2, col=1)

    # ── RSI ──
    fig.add_trace(go.Scatter(x=idx, y=rsi14.reindex(idx), line=dict(color="#7b8fff", width=1.5),
                              name="RSI 14", showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#ff4d6d40", dash="dash", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#00e67640", dash="dash", width=1), row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="#ffffff05", line_width=0, row=3, col=1)

    fig.update_layout(
        **PLOTLY_DARK,
        title=dict(text=f"<b>{symbol}.NS</b>  ·  {df_plot['Close'].iloc[-1]:.2f}",
                   font=dict(size=18, color="#00d4ff"), x=0.01),
        height=720,
        margin=dict(l=0, r=0, t=48, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    showticklabels=False, range=[0, df_plot["Volume"].max() * 6]),
    )
    fig.update_xaxes(showspikes=True, spikecolor="#1e3a5f", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor="#1e3a5f", spikethickness=1)

    st.plotly_chart(fig, use_container_width=True)

# ── PORTFOLIO OPTIMIZER ───────────────────────────────────────────────────────
def show_portfolio_tab(df_results: pd.DataFrame):
    st.markdown("## Portfolio Optimizer")
    if not HAS_PYPFOPT:
        st.error("pyportfolioopt not installed.")
        return
    if df_results.empty:
        st.info("Run a scan first to populate candidates.")
        return

    symbols = df_results["Symbol"].tolist()
    st.markdown(f"<p style='color:var(--muted);font-size:0.78rem;'>Select symbols from scan results ({len(symbols)} available)</p>", unsafe_allow_html=True)

    selected = st.multiselect("Symbols for portfolio", symbols, default=symbols[:min(12, len(symbols))])
    if len(selected) < 3:
        st.warning("Select at least 3 symbols.")
        return

    opt_method = st.radio("Optimization target", ["Max Sharpe Ratio", "Min Volatility", "Max Quadratic Utility"], horizontal=True)

    if st.button("◈ OPTIMIZE"):
        with st.spinner("Fetching price data..."):
            prices = fetch_close_matrix(selected)

        if prices.empty or len(prices.columns) < 3:
            st.error("Not enough price data. Try different symbols.")
            return

        try:
            mu = expected_returns.mean_historical_return(prices)
            S  = risk_models.sample_cov(prices)
            ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 0.40))

            if opt_method == "Max Sharpe Ratio":
                ef.max_sharpe(risk_free_rate=0.065)
            elif opt_method == "Min Volatility":
                ef.min_volatility()
            else:
                ef.max_quadratic_utility(risk_aversion=2)

            weights = ef.clean_weights()
            perf    = ef.portfolio_performance(risk_free_rate=0.065)

            # ── Metrics ──
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Return", f"{perf[0]*100:.1f}%")
            c2.metric("Volatility",      f"{perf[1]*100:.1f}%")
            c3.metric("Sharpe Ratio",    f"{perf[2]:.2f}")

            # ── Weights table ──
            wdf = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
            wdf = wdf[wdf["Weight"] > 0.001].sort_values("Weight", ascending=False)
            wdf["Weight %"] = (wdf["Weight"] * 100).round(2)
            wdf = wdf.drop(columns="Weight").reset_index().rename(columns={"index": "Symbol"})

            if HAS_PLOTLY:
                fig = go.Figure(go.Bar(
                    x=wdf["Symbol"], y=wdf["Weight %"],
                    marker=dict(color=wdf["Weight %"],
                                colorscale=[[0, "#1e3a5f"], [0.5, "#00d4ff"], [1, "#f5a623"]],
                                showscale=False),
                    text=wdf["Weight %"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside", textfont=dict(color="#cde4f5", size=10),
                ))
                fig.update_layout(**PLOTLY_DARK, height=320,
                                  margin=dict(l=0, r=0, t=16, b=0),
                                  yaxis_title="Weight %", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                wdf.style.format({"Weight %": "{:.2f}"})
                   .set_properties(**{"background-color": "#0d1520", "color": "#cde4f5"}),
                use_container_width=True, height=280
            )

            csv = wdf.to_csv(index=False).encode()
            st.download_button("↓ EXPORT WEIGHTS", csv,
                               f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

        except Exception as e:
            st.error(f"Optimization failed: {e}")

# ── CLUSTER ANALYSIS ──────────────────────────────────────────────────────────
def show_cluster_tab(df_results: pd.DataFrame):
    st.markdown("## RSI Momentum Clusters")
    if not HAS_SKLEARN:
        st.error("scikit-learn not installed.")
        return
    if df_results.empty or len(df_results) < 6:
        st.info("Run a scan with at least 6 stocks first.")
        return

    rsi_cols = [c for c in ["RSI 14", "RSI 21", "RSI 63", "RSI 126", "RSI 252"] if c in df_results.columns]
    feat_df = df_results[["Symbol"] + rsi_cols + ["% vs 200EMA", "ATR %"]].dropna()

    if len(feat_df) < 6:
        st.info("Not enough complete data for clustering.")
        return

    n_clusters = st.slider("Number of clusters", 2, 6, 4)

    X = feat_df[rsi_cols + ["% vs 200EMA", "ATR %"]].values
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat_df = feat_df.copy()
    feat_df["Cluster"] = km.fit_predict(X_scaled)

    # Label clusters by avg RSI14
    cluster_rsi = feat_df.groupby("Cluster")["RSI 14"].mean().sort_values(ascending=False)
    labels = {
        0: "🔥 Strong Momentum",
        1: "📈 Building Up",
        2: "⚖️ Neutral",
        3: "📉 Weakening",
        4: "💧 Oversold",
        5: "❄️ Deep Bear",
    }
    rank_map = {cid: i for i, cid in enumerate(cluster_rsi.index)}
    feat_df["Group"] = feat_df["Cluster"].map(lambda c: labels.get(rank_map[c], f"Cluster {rank_map[c]}"))

    if HAS_PLOTLY:
        fig = px.scatter(
            feat_df, x="RSI 14", y="% vs 200EMA",
            color="Group", hover_name="Symbol",
            size="ATR %", size_max=18,
            color_discrete_sequence=["#00e676", "#00d4ff", "#f5a623", "#ff4d6d", "#7b8fff", "#ff6ec7"],
        )
        fig.update_layout(
            **PLOTLY_DARK, height=480,
            margin=dict(l=0, r=0, t=16, b=0),
            xaxis_title="RSI 14", yaxis_title="% vs 200 EMA",
            legend=dict(orientation="v", x=1.01, y=0.5),
        )
        fig.add_vline(x=50, line=dict(color="#1e3a5f", dash="dash"))
        fig.add_hline(y=0,  line=dict(color="#1e3a5f", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    for grp_name, grp_df in feat_df.groupby("Group"):
        with st.expander(f"{grp_name}  ({len(grp_df)} stocks)"):
            show_cols = ["Symbol", "RSI 14", "RSI 63", "% vs 200EMA", "ATR %"]
            st.dataframe(
                grp_df[show_cols].sort_values("RSI 14", ascending=False)
                   .style.format(precision=1)
                   .set_properties(**{"background-color": "#0d1520", "color": "#cde4f5",
                                      "font-size": "0.75rem"}),
                use_container_width=True, hide_index=True,
            )

# ── BREADTH ───────────────────────────────────────────────────────────────────
def show_breadth(df: pd.DataFrame):
    st.markdown("## Market Breadth")
    n = len(df)
    if n == 0:
        return

    above200   = (df["LTP vs 200EMA"] == "ABOVE").sum()
    aligned    = (df["EMA Aligned"]   == "✓").sum()
    near_sup   = (df["Near Support"]  == "✓").sum()
    macd_bull  = (df.get("MACD Bull", pd.Series()) == "✓").sum()
    r14_bull   = (df["RSI 14"] > 50).sum()
    r14_ob     = (df["RSI 14"] > 70).sum()
    r14_os     = (df["RSI 14"] < 30).sum()

    def p(x): return f"{x}  ({int(100*x/n)}%)"

    cols = st.columns(4)
    cols[0].metric("Scanned",          str(n))
    cols[1].metric("Above 200 EMA",    p(above200))
    cols[2].metric("EMA Aligned",      p(aligned))
    cols[3].metric("Near Support",     str(near_sup))

    cols2 = st.columns(4)
    cols2[0].metric("RSI14 Bullish",   p(r14_bull))
    cols2[1].metric("RSI14 Overbought",p(r14_ob))
    cols2[2].metric("RSI14 Oversold",  p(r14_os))
    cols2[3].metric("MACD Bullish",    p(macd_bull) if macd_bull > 0 else "–")

    pct_above = int(100 * above200 / n)
    breadth   = "BULLISH" if pct_above > 60 else ("BEARISH" if pct_above < 40 else "NEUTRAL")
    color     = "#00e676" if breadth == "BULLISH" else ("#ff4d6d" if breadth == "BEARISH" else "#f5a623")
    avg14, avg63 = df["RSI 14"].mean(), df["RSI 63"].mean()

    st.markdown(f"""
    <div style='display:flex;gap:16px;flex-wrap:wrap;margin:12px 0;'>
        <div style='background:#0d1520;border:1px solid {color}40;border-left:3px solid {color};
                    padding:12px 20px;border-radius:2px;'>
            <div style='color:{color};font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;'>
                {breadth}
            </div>
            <div style='color:#4a7090;font-size:0.65rem;letter-spacing:1px;margin-top:2px;'>OVERALL BREADTH</div>
        </div>
        <div style='background:#0d1520;border:1px solid #1e3a5f;padding:12px 20px;border-radius:2px;'>
            <div style='color:#00d4ff;font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;'>{avg14:.1f}</div>
            <div style='color:#4a7090;font-size:0.65rem;letter-spacing:1px;margin-top:2px;'>AVG RSI 14</div>
        </div>
        <div style='background:#0d1520;border:1px solid #1e3a5f;padding:12px 20px;border-radius:2px;'>
            <div style='color:#f5a623;font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;'>{avg63:.1f}</div>
            <div style='color:#4a7090;font-size:0.65rem;letter-spacing:1px;margin-top:2px;'>AVG RSI 63</div>
        </div>
        <div style='background:#0d1520;border:1px solid #1e3a5f;padding:12px 20px;border-radius:2px;'>
            <div style='color:#cde4f5;font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;'>{pct_above}%</div>
            <div style='color:#4a7090;font-size:0.65rem;letter-spacing:1px;margin-top:2px;'>ABOVE 200 EMA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # RSI distribution bar chart
    if HAS_PLOTLY:
        bins  = [0, 20, 30, 40, 50, 60, 70, 80, 100]
        labs  = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", ">80"]
        cnts  = pd.cut(df["RSI 14"].dropna(), bins=bins, labels=labs).value_counts().reindex(labs, fill_value=0)
        cols_bar = ["#ff4d6d", "#ff4d6d", "#f5a623", "#f5a623", "#4ddb8a", "#4ddb8a", "#00e676", "#00e676"]
        fig = go.Figure(go.Bar(x=cnts.index, y=cnts.values, marker_color=cols_bar,
                                text=cnts.values, textposition="outside",
                                textfont=dict(color="#cde4f5", size=11)))
        fig.update_layout(**PLOTLY_DARK, height=260, margin=dict(l=0, r=0, t=8, b=0),
                          title=dict(text="RSI 14 Distribution", font=dict(size=13, color="#4a7090")),
                          xaxis_title="RSI Range", yaxis_title="# Stocks",
                          bargap=0.25)
        st.plotly_chart(fig, use_container_width=True)

# ── SECTOR BREAKDOWN ──────────────────────────────────────────────────────────
def show_sector_breakdown(df: pd.DataFrame):
    st.markdown("## Sector Breakdown")
    if "Sector" not in df.columns or df.empty:
        return

    sec = df.groupby("Sector").agg(
        Count=("Symbol", "count"),
        Avg_RSI14=("RSI 14", "mean"),
        Avg_RSI63=("RSI 63", "mean"),
        Above_200=("LTP vs 200EMA", lambda x: (x == "ABOVE").sum()),
        EMA_Aligned=("EMA Aligned", lambda x: (x == "✓").sum()),
        Near_Sup=("Near Support", lambda x: (x == "✓").sum()),
    ).round(1).reset_index()

    sec["% Above200"] = (sec["Above_200"] / sec["Count"] * 100).round(1)
    sec["% Aligned"]  = (sec["EMA_Aligned"] / sec["Count"] * 100).round(1)
    sec = sec.sort_values("Count", ascending=False)

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sec["Sector"], y=sec["% Above200"],
            name="% Above 200EMA",
            marker_color="#00d4ff",
            text=sec["% Above200"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside", textfont=dict(size=10, color="#cde4f5"),
        ))
        fig.add_trace(go.Scatter(
            x=sec["Sector"], y=sec["Avg_RSI14"],
            name="Avg RSI14", yaxis="y2",
            line=dict(color="#f5a623", width=2),
            mode="lines+markers", marker=dict(size=6),
        ))
        fig.add_hline(y=50, line=dict(color="#1e3a5f", dash="dash"), yref="y")
        fig.update_layout(
            **PLOTLY_DARK, height=360, margin=dict(l=0, r=0, t=16, b=0),
            yaxis=dict(title="% Above 200EMA", range=[0, 110]),
            yaxis2=dict(title="Avg RSI 14", overlaying="y", side="right", range=[20, 80]),
            legend=dict(orientation="h", y=1.05),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

    disp = sec[["Sector", "Count", "Avg_RSI14", "Avg_RSI63", "% Above200", "% Aligned", "Near_Sup"]]

    def pct_color(v):
        try:
            v = float(v)
            if v >= 60: return "color:#00e676;font-weight:700"
            elif v >= 40: return "color:#f5a623"
            else: return "color:#ff4d6d"
        except: return ""

    try:
        styled = disp.style.map(pct_color, subset=["% Above200", "% Aligned"])
    except TypeError:
        styled = disp.style.applymap(pct_color, subset=["% Above200", "% Aligned"])

    styled = styled.format(precision=1).set_properties(**{
        "background-color": "#0d1520", "color": "#cde4f5",
        "font-size": "0.75rem", "font-family": "Space Mono, monospace",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
def draw_header():
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    st.markdown(f"""
    <div style='padding:28px 0 8px 0;'>
        <h1>AlphaMomentum</h1>
        <div style='display:flex;align-items:center;gap:16px;margin-top:8px;flex-wrap:wrap;'>
            <span style='color:#4a7090;font-family:Space Mono,monospace;font-size:0.72rem;letter-spacing:2px;'>
                NSE TOTAL MARKET  ·  SCANNER + CHARTS + OPTIMIZER + CLUSTERS
            </span>
            <span style='margin-left:auto;color:#1e3a5f;font-size:0.68rem;font-family:Space Mono,monospace;'>{now} UTC</span>
        </div>
        <div style='display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;'>
            <span style='background:#0d1520;border:1px solid #1e3a5f;padding:3px 10px;color:#00d4ff;font-size:0.65rem;font-family:Space Mono,monospace;letter-spacing:1px;'>● LIVE</span>
            <span style='background:#0d1520;border:1px solid #1e3a5f;padding:3px 10px;color:#4a7090;font-size:0.65rem;font-family:Space Mono,monospace;letter-spacing:1px;'>yfinance</span>
            <span style='background:#0d1520;border:1px solid #1e3a5f;padding:3px 10px;color:#4a7090;font-size:0.65rem;font-family:Space Mono,monospace;letter-spacing:1px;'>scipy  |  sklearn  |  pypfopt</span>
        </div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,#00d4ff,#1e3a5f,transparent);margin:16px 0 24px 0;'></div>
    """, unsafe_allow_html=True)

# ── IDLE ──────────────────────────────────────────────────────────────────────
def draw_idle():
    st.markdown("""
    <div style='text-align:center;padding:60px 0;'>
        <div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;
                    color:#1e3a5f;letter-spacing:-2px;margin-bottom:16px;'>◈</div>
        <p style='color:#4a7090;font-family:Space Mono,monospace;font-size:0.78rem;
                  letter-spacing:3px;text-transform:uppercase;'>Awaiting scan command</p>
        <p style='color:#1e3a5f;font-size:0.72rem;font-family:Space Mono,monospace;margin-top:8px;'>
            Configure filters → click START SCANNING
        </p>
        <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
                    gap:12px;max-width:800px;margin:32px auto 0;'>
            {''.join(f"""
            <div style='background:#0d1520;border:1px solid #1e3a5f;border-top:2px solid {c};
                        padding:16px;border-radius:2px;text-align:left;'>
                <div style='color:{c};font-size:1.2rem;margin-bottom:8px;'>{icon}</div>
                <div style='color:#4a7090;font-size:0.68rem;font-family:Space Mono,monospace;
                            text-transform:uppercase;letter-spacing:1px;'>{lbl}</div>
            </div>""" for icon, lbl, c in [
                ("◈", "EMA 20/50/200 Alignment", "#00d4ff"),
                ("◎", "Multi-TF RSI 14–252", "#7b8fff"),
                ("◇", "scipy Support Zones", "#f5a623"),
                ("◉", "Portfolio Optimizer", "#00e676"),
                ("◐", "RSI Cluster Analysis", "#ff6ec7"),
                ("◑", "Sector Breadth Heatmap", "#ff4d6d"),
            ])}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    draw_header()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ◈ SCANNER CONFIG")
        st.markdown('<hr style="border-color:#1e3a5f;margin:8px 0 16px 0;">', unsafe_allow_html=True)

        max_stocks = st.slider("Max Stocks", 10, 200, 40, 10)
        support_pct = st.slider("Support Proximity %", 1, 15, 5, 1)
        threshold_pct = support_pct / 100

        st.markdown('<hr style="border-color:#1e3a5f;margin:16px 0;">', unsafe_allow_html=True)
        st.markdown("### ◎ FILTERS")
        f_support = st.checkbox("Near Support Only",   False)
        f_aligned = st.checkbox("EMA Aligned Only",    False)
        f_above   = st.checkbox("Above 200 EMA Only",  False)
        f_rsi     = st.checkbox("RSI14 > 50 Only",     False)
        f_macd    = st.checkbox("MACD Bullish Only",   False)

        st.markdown('<hr style="border-color:#1e3a5f;margin:16px 0;">', unsafe_allow_html=True)
        st.markdown("### ◇ SORT")
        sort_col = st.selectbox("By", ["RSI 14", "% vs 200EMA", "% from 52W High",
                                        "Vol Ratio", "ATR %", "BB %", "LTP", "Symbol"])
        sort_asc = st.checkbox("Ascending", False)

        st.markdown('<hr style="border-color:#1e3a5f;margin:16px 0;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='color:#1e3a5f;font-size:0.65rem;font-family:Space Mono,monospace;line-height:2;'>
        RSI ≥70  Overbought<br>RSI 50-70  Bullish<br>
        RSI 30-50  Bearish<br>RSI ≤30  Oversold<br>
        ──────────────────<br>
        Support = within {support_pct}% of<br>scipy local minima
        </div>
        """, unsafe_allow_html=True)

    # ── Load stock list ──
    with st.spinner("Loading Nifty Total Market..."):
        stocks_df = fetch_stock_list()

    if stocks_df.empty:
        st.warning("Using fallback stock list (niftyindices.com unreachable).")
        stocks_df = pd.DataFrame({
            "Symbol": ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
                       "BHARTIARTL","ITC","KOTAKBANK","WIPRO","AXISBANK","MARUTI","SUNPHARMA",
                       "TITAN","NESTLEIND","POWERGRID","NTPC","ULTRACEMCO","BAJFINANCE",
                       "ASIANPAINT","HCLTECH","TECHM","DIVISLAB","DRREDDY","CIPLA","ADANIENT",
                       "ADANIPORTS","TATAMOTORS","TATASTEEL"],
            "Industry": ["ENERGY","IT","FINANCIALS","IT","FINANCIALS","FMCG","FINANCIALS",
                         "TELECOM","FMCG","FINANCIALS","IT","FINANCIALS","AUTO","PHARMA",
                         "CONSUMER","FMCG","UTILITIES","UTILITIES","CEMENT","FINANCIALS",
                         "PAINTS","IT","IT","PHARMA","PHARMA","PHARMA","INFRA",
                         "INFRA","AUTO","METALS"],
        })
    else:
        st.success(f"✓  {len(stocks_df)} stocks loaded", icon="◈")

    cols_lower = {c.lower(): c for c in stocks_df.columns}
    sym_col = next((cols_lower[k] for k in ["symbol","sym"] if k in cols_lower), stocks_df.columns[0])
    sec_col = next((cols_lower[k] for k in ["industry","sector","ind"] if k in cols_lower), None)

    if sec_col:
        all_sectors = sorted(stocks_df[sec_col].dropna().astype(str).unique())
        with st.sidebar:
            st.markdown('<hr style="border-color:#1e3a5f;margin:16px 0;">', unsafe_allow_html=True)
            st.markdown("### ◉ SECTOR")
            sector_sel = st.multiselect("Filter sectors (blank = all)", all_sectors)
        if sector_sel:
            stocks_df = stocks_df[stocks_df[sec_col].astype(str).isin(sector_sel)]

    stocks_df = stocks_df.head(max_stocks)

    # ── Scan button ──
    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c2:
        scan_btn = st.button("◈  START SCANNING", use_container_width=True)

    # ── Session state for results ──
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = pd.DataFrame()

    if not scan_btn and st.session_state.scan_results.empty:
        draw_idle()
        return

    # ── Execute scan ──
    if scan_btn:
        st.markdown(f"<p style='color:#4a7090;font-size:0.72rem;text-align:center;letter-spacing:1px;'>"
                    f"Scanning {len(stocks_df)} stocks  ·  Support proximity {support_pct}%</p>",
                    unsafe_allow_html=True)

        prog  = st.progress(0)
        status = st.empty()
        results, errors = [], 0
        total = len(stocks_df)

        for i, (_, row) in enumerate(stocks_df.iterrows()):
            sym = str(row[sym_col]).strip().upper()
            sec = str(row[sec_col]).strip() if sec_col else "N/A"
            status.markdown(
                f"<p style='color:#1e3a5f;font-family:Space Mono,monospace;font-size:0.72rem;"
                f"text-align:center;'>[{i+1}/{total}]  {sym}  —  {sec}</p>",
                unsafe_allow_html=True)
            try:
                res = scan_stock(sym, sec, threshold_pct)
                if res: results.append(res)
                else:   errors += 1
            except Exception:
                errors += 1
            prog.progress((i + 1) / total)

        prog.empty()
        status.empty()

        if not results:
            st.error("No results. Check internet connection.")
            return

        st.session_state.scan_results = pd.DataFrame(results)
        if errors:
            st.info(f"ℹ  {errors} symbols skipped (no data / delisted / insufficient history)")

    df_res = st.session_state.scan_results

    # ── Filters ──
    df_f = df_res.copy()
    if f_support: df_f = df_f[df_f["Near Support"]  == "✓"]
    if f_aligned: df_f = df_f[df_f["EMA Aligned"]   == "✓"]
    if f_above:   df_f = df_f[df_f["LTP vs 200EMA"] == "ABOVE"]
    if f_rsi:     df_f = df_f[df_f["RSI 14"] > 50]
    if f_macd:    df_f = df_f[df_f.get("MACD Bull", pd.Series()) == "✓"]
    if sort_col in df_f.columns:
        df_f = df_f.sort_values(sort_col, ascending=sort_asc)

    # ── TABS ──
    t1, t2, t3, t4, t5 = st.tabs([
        "◈  SCANNER", "◎  CHART", "◉  PORTFOLIO", "◐  CLUSTERS", "◑  SECTORS"
    ])

    # ── TAB 1: Scanner ──
    with t1:
        show_breadth(df_res)
        st.markdown(
            f"## Results  "
            f"<span style='color:#4a7090;font-size:0.78rem;font-family:Space Mono;font-weight:400;'>"
            f"{len(df_f)} / {len(df_res)} stocks</span>",
            unsafe_allow_html=True,
        )

        # Legend
        st.markdown("""
        <div style='background:#0d1520;border:1px solid #1e3a5f;padding:8px 16px;
                    font-size:0.65rem;font-family:Space Mono,monospace;display:flex;
                    gap:20px;flex-wrap:wrap;margin-bottom:12px;border-radius:2px;'>
            <span><span style='color:#00e676'>■</span> RSI ≥70</span>
            <span><span style='color:#4ddb8a'>■</span> RSI 50-70</span>
            <span><span style='color:#f5a623'>■</span> RSI 30-50</span>
            <span><span style='color:#ff4d6d'>■</span> RSI ≤30</span>
            <span style='color:#4a7090;margin-left:8px;'>✓ = Yes  ✗ = No</span>
        </div>
        """, unsafe_allow_html=True)

        if df_f.empty:
            st.warning("No stocks match current filters. Relax filter conditions.")
        else:
            display_cols = ["Symbol","Sector","LTP","% vs 200EMA","% from 52W High",
                            "EMA Aligned","LTP vs 200EMA","Near Support","MACD Bull",
                            "Vol Ratio","ATR %","BB %","RSI 14","RSI 21","RSI 63","RSI 126","RSI 252"]
            display_cols = [c for c in display_cols if c in df_f.columns]
            try:
                st.dataframe(style_table(df_f[display_cols]),
                             use_container_width=True, height=540)
            except Exception:
                st.dataframe(df_f[display_cols], use_container_width=True, height=540)

            c1, c2, c3 = st.columns([2, 1, 2])
            with c2:
                st.download_button(
                    "↓ EXPORT CSV",
                    df_f.to_csv(index=False).encode(),
                    f"alphamomentum_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv", use_container_width=True,
                )

    # ── TAB 2: Chart ──
    with t2:
        st.markdown("## Stock Deep Dive")
        if df_res.empty:
            st.info("Run a scan first.")
        else:
            syms = df_res["Symbol"].tolist()
            chosen = st.selectbox("Select symbol", syms)
            if chosen:
                col_info = st.columns(6)
                row = df_res[df_res["Symbol"] == chosen].iloc[0]
                for col_w, (label, key) in zip(col_info, [
                    ("LTP", "LTP"), ("RSI 14", "RSI 14"), ("RSI 63", "RSI 63"),
                    ("% vs 200EMA", "% vs 200EMA"), ("EMA Aligned", "EMA Aligned"), ("Near Sup", "Near Support"),
                ]):
                    col_w.metric(label, row.get(key, "–"))

                with st.spinner(f"Loading chart for {chosen}..."):
                    df_chart = fetch_daily_data(chosen)
                plot_stock(chosen, df_chart)

    # ── TAB 3: Portfolio ──
    with t3:
        show_portfolio_tab(df_f if not df_f.empty else df_res)

    # ── TAB 4: Clusters ──
    with t4:
        show_cluster_tab(df_f if not df_f.empty else df_res)

    # ── TAB 5: Sectors ──
    with t5:
        show_sector_breakdown(df_f if not df_f.empty else df_res)

    # Footer
    st.markdown("""
    <div style='height:1px;background:linear-gradient(90deg,transparent,#1e3a5f,transparent);margin:32px 0 16px;'></div>
    <p style='text-align:center;color:#1e3a5f;font-family:Space Mono,monospace;font-size:0.62rem;'>
    FOR EDUCATIONAL & RESEARCH USE ONLY  ·  NOT INVESTMENT ADVICE  ·  DATA: YAHOO FINANCE / NIFTYINDICES
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
