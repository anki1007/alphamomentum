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

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Scanner Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NEON TEAL CSS THEME ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ── Base ── */
html, body, [data-testid="stApp"], .stApp {
    background: #010f14 !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, #00ffff08 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, #006d7518 0%, transparent 60%) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #011c22; }
::-webkit-scrollbar-thumb { background: #00ffff44; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00ffff99; }

/* ── Headers ── */
h1 {
    font-family: 'Orbitron', monospace !important;
    color: #00ffff !important;
    text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff60, 0 0 80px #00ffff30;
    letter-spacing: 3px;
}
h2 {
    font-family: 'Orbitron', monospace !important;
    color: #39ff14 !important;
    text-shadow: 0 0 15px #39ff1480;
    letter-spacing: 2px;
    border-bottom: 1px solid #39ff1430;
    padding-bottom: 8px;
}
h3 {
    color: #ff6ec7 !important;
    text-shadow: 0 0 10px #ff6ec780;
    font-family: 'Orbitron', monospace !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #010f14 0%, #011c22 100%) !important;
    border-right: 1px solid #00ffff30 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-size: 1rem !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .st-emotion-cache-1y4p8pa { color: #a0d8df !important; }

/* ── Widget labels ── */
label, .stSelectbox label, .stSlider label, .stMultiSelect label,
.stCheckbox label span, p { color: #a0d8df !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #003d4d 0%, #005566 100%) !important;
    color: #00ffff !important;
    border: 1px solid #00ffff !important;
    border-radius: 4px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    box-shadow: 0 0 15px #00ffff40, inset 0 0 15px #00ffff10 !important;
    transition: all 0.3s !important;
    padding: 0.5rem 2rem !important;
}
.stButton > button:hover {
    box-shadow: 0 0 30px #00ffff80, inset 0 0 20px #00ffff20 !important;
    transform: translateY(-1px);
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: #00ffff !important;
    font-size: 1.6rem !important;
    text-shadow: 0 0 10px #00ffff80;
}
[data-testid="stMetricLabel"] { color: #a0d8df !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="metric-container"] {
    background: #011c2280 !important;
    border: 1px solid #00ffff20 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #00ffff30 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
.dataframe thead th {
    background: #003d4d !important;
    color: #00ffff !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    border-bottom: 2px solid #00ffff50 !important;
}
.dataframe tbody tr { background: #010f14 !important; }
.dataframe tbody tr:nth-child(even) { background: #011c22 !important; }
.dataframe tbody tr:hover { background: #00ffff10 !important; }
.dataframe tbody td { color: #e0f7fa !important; font-size: 0.82rem !important; border-color: #00ffff15 !important; }

/* ── Divider ── */
hr { border-color: #00ffff25 !important; margin: 1rem 0 !important; }

/* ── Select / Input ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #011c22 !important;
    color: #00ffff !important;
    border: 1px solid #00ffff30 !important;
}
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

/* ── Progress ── */
[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #00ffff, #39ff14) !important;
}

/* ── Alerts ── */
[data-testid="stSuccess"] { background: #0a2a14 !important; border-left: 4px solid #39ff14 !important; color: #39ff14 !important; }
[data-testid="stWarning"] { background: #2a1a00 !important; border-left: 4px solid #ffaa00 !important; }
[data-testid="stError"] { background: #2a0a0a !important; border-left: 4px solid #ff4444 !important; }
[data-testid="stInfo"] { background: #00131a !important; border-left: 4px solid #00ffff !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: #001520 !important;
    color: #39ff14 !important;
    border: 1px solid #39ff14 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #00ffff !important; }

/* ── Checkbox ── */
.stCheckbox > label { color: #a0d8df !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
NIFTY_URL = "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
CACHE_TTL = 3600

# ── HELPERS ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_list():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(NIFTY_URL, headers=headers, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_daily_data(symbol: str):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start="2019-12-01", interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()

# ── INDICATOR CALCULATIONS ───────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    last = r.iloc[-1]
    return round(last, 1) if pd.notna(last) else np.nan

# ── SUPPORT LOGIC ────────────────────────────────────────────────────────────
def get_support_levels(df: pd.DataFrame, threshold_pct: float) -> tuple:
    """Returns (is_near_support, nearest_level, yearly_low, yearly_high, monthly_ohlc_df)"""
    if df.empty:
        return False, None, None, None, pd.DataFrame()

    hist = df[df.index.year.isin([2020, 2021, 2022])].copy()
    if hist.empty:
        return False, None, None, None, pd.DataFrame()

    monthly = hist.resample("ME").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    ).dropna()

    yearly_low  = round(hist["Low"].min(), 2)
    yearly_high = round(hist["High"].max(), 2)

    # Support candidates: monthly lows + yearly low
    levels = monthly["Low"].tolist() + [yearly_low]
    ltp = df["Close"].iloc[-1]

    nearest = None
    min_dist = float("inf")
    for lvl in levels:
        if lvl > 0:
            dist = abs(ltp - lvl) / lvl
            if dist < min_dist:
                min_dist = dist
                nearest = lvl

    near = min_dist <= threshold_pct

    return near, round(nearest, 2) if nearest else None, yearly_low, yearly_high, monthly

# ── SCAN ONE STOCK ───────────────────────────────────────────────────────────
def scan_stock(symbol: str, sector: str, threshold_pct: float) -> dict | None:
    df = fetch_daily_data(symbol)
    if df.empty or len(df) < 260:
        return None

    close = df["Close"]
    ltp   = round(close.iloc[-1], 2)

    e10  = round(ema(close, 10).iloc[-1], 2)
    e20  = round(ema(close, 20).iloc[-1], 2)
    e40  = round(ema(close, 40).iloc[-1], 2)
    e50  = round(ema(close, 50).iloc[-1], 2)
    e200 = round(ema(close, 200).iloc[-1], 2)

    aligned = bool(ltp > e10 > e20 > e50 > e200)

    r14  = rsi(close, 14)
    r21  = rsi(close, 21)
    r63  = rsi(close, 63)
    r126 = rsi(close, 126)
    r252 = rsi(close, 252)

    near_sup, sup_level, yr_low, yr_high, _ = get_support_levels(df, threshold_pct)

    pct_from_200 = round((ltp / e200 - 1) * 100, 2) if e200 > 0 else 0

    return {
        "Symbol"        : symbol,
        "Sector"        : sector,
        "LTP"           : ltp,
	"10 EMA"        : e10,
        "20 EMA"        : e20,
        "40 EMA"        : e40,
        "50 EMA"        : e50,
        "200 EMA"       : e200,
        "% vs 200EMA"   : pct_from_200,
        "EMA Aligned"   : "YES" if aligned else "NO",
        "LTP > 200EMA"  : "ABOVE" if ltp > e200 else "BELOW",
        "Near Support"  : "YES" if near_sup else "-",
        "Support Level" : sup_level if sup_level else np.nan,
        "2020-22 Low"   : yr_low if yr_low else np.nan,
        "2020-22 High"  : yr_high if yr_high else np.nan,
        "RSI 14"        : r14,
        "RSI 21"        : r21,
        "RSI 63"        : r63,
        "RSI 126"       : r126,
        "RSI 252"       : r252,
    }

# ── STYLING ──────────────────────────────────────────────────────────────────
RSI_COLS = ["RSI 14", "RSI 21", "RSI 63", "RSI 126", "RSI 252"]

def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def rsi_color(val):
        try:
            v = float(val)
            if v >= 70:
                return "background-color:#003d00;color:#39ff14;font-weight:bold"
            elif v >= 50:
                return "background-color:#001a00;color:#00e600;font-weight:bold"
            elif v >= 30:
                return "background-color:#2a0a0a;color:#ff6666;font-weight:bold"
            else:
                return "background-color:#3d0000;color:#ff1111;font-weight:bold"
        except:
            return ""

    def ema_align_color(val):
        if str(val) == "YES":
            return "background-color:#003d00;color:#39ff14;font-weight:bold;text-align:center"
        return "background-color:#2a0a0a;color:#ff4444;text-align:center"

    def ltp_200_color(val):
        if str(val) == "ABOVE":
            return "background-color:#001a00;color:#39ff14;font-weight:bold;text-align:center"
        return "background-color:#2a0a0a;color:#ff4444;font-weight:bold;text-align:center"

    def support_color(val):
        if str(val) == "YES":
            return "background-color:#001a3d;color:#00ffff;font-weight:bold;text-align:center"
        return "color:#445555;text-align:center"

    def pct_color(val):
        try:
            v = float(val)
            if v > 10:
                return "color:#39ff14;font-weight:bold"
            elif v > 0:
                return "color:#00cc00"
            elif v > -10:
                return "color:#ff8800"
            else:
                return "color:#ff4444"
        except:
            return ""

    try:
        # pandas >= 2.1 uses .map instead of .applymap
        styler = (
            df.style
            .map(rsi_color, subset=RSI_COLS)
            .map(ema_align_color, subset=["EMA Aligned"])
            .map(ltp_200_color, subset=["LTP > 200EMA"])
            .map(support_color, subset=["Near Support"])
            .map(pct_color, subset=["% vs 200EMA"])
            .set_properties(**{"background-color": "#010f14", "color": "#c8e6f0", "border-color": "#00ffff15"})
            .set_table_styles([
                {"selector": "thead th", "props": [
                    ("background-color", "#003d4d"),
                    ("color", "#00ffff"),
                    ("font-size", "0.72rem"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "0.5px"),
                    ("border-bottom", "2px solid #00ffff50"),
                ]},
                {"selector": "tbody tr:hover td", "props": [("background-color", "#00ffff10")]},
            ])
            .format(precision=2, na_rep="-")
        )
    except TypeError:
        # fallback for older pandas
        styler = (
            df.style
            .applymap(rsi_color, subset=RSI_COLS)
            .applymap(ema_align_color, subset=["EMA Aligned"])
            .applymap(ltp_200_color, subset=["LTP > 200EMA"])
            .applymap(support_color, subset=["Near Support"])
            .applymap(pct_color, subset=["% vs 200EMA"])
            .format(precision=2, na_rep="-")
        )
    return styler

# ── BREADTH DISPLAY ──────────────────────────────────────────────────────────
def show_breadth(df: pd.DataFrame):
    st.markdown("## 📊 Market Breadth")
    n = len(df)
    if n == 0:
        return

    above_200      = (df["LTP > 200EMA"] == "ABOVE").sum()
    ema_aligned    = (df["EMA Aligned"] == "YES").sum()
    near_support   = (df["Near Support"] == "YES").sum()
    rsi14_bull     = (df["RSI 14"] > 50).sum()
    rsi14_bear     = (df["RSI 14"] < 50).sum()
    rsi14_overbuy  = (df["RSI 14"] > 70).sum()
    rsi14_oversold = (df["RSI 14"] < 30).sum()
    all_rsi_bull   = ((df["RSI 14"] > 50) & (df["RSI 21"] > 50) & (df["RSI 63"] > 50)).sum()

    def pct(x): return f"{x} ({int(100*x/n)}%)"

    cols = st.columns(4)
    cols[0].metric("🔵 Total Scanned",    str(n))
    cols[1].metric("🟢 Above 200 EMA",    pct(above_200))
    cols[2].metric("⚡ EMA Aligned",      pct(ema_aligned))
    cols[3].metric("🎯 Near Support",     str(near_support))

    cols2 = st.columns(4)
    cols2[0].metric("📈 RSI14 Bullish",   pct(rsi14_bull))
    cols2[1].metric("📉 RSI14 Bearish",   pct(rsi14_bear))
    cols2[2].metric("🔥 RSI14 Overbought",pct(rsi14_overbuy))
    cols2[3].metric("💧 RSI14 Oversold",  pct(rsi14_oversold))

    cols3 = st.columns(4)
    cols3[0].metric("✅ All RSI>50 (14/21/63)", pct(all_rsi_bull))
    pct_above = int(100 * above_200 / n)
    breadth_status = "🟢 BULLISH" if pct_above > 60 else ("🔴 BEARISH" if pct_above < 40 else "🟡 NEUTRAL")
    cols3[1].metric("📊 Overall Breadth", breadth_status)
    avg_rsi14 = df["RSI 14"].mean()
    cols3[2].metric("📉 Avg RSI14", f"{avg_rsi14:.1f}")
    avg_rsi63 = df["RSI 63"].mean()
    cols3[3].metric("📉 Avg RSI63", f"{avg_rsi63:.1f}")

    st.divider()

# ── SECTOR BREAKDOWN ──────────────────────────────────────────────────────────
def show_sector_breakdown(df: pd.DataFrame):
    st.markdown("## 🏭 Sector Breakdown")
    if "Sector" not in df.columns:
        return

    sec = df.groupby("Sector").agg(
        Count          = ("Symbol", "count"),
        Avg_RSI14      = ("RSI 14", "mean"),
        Avg_RSI63      = ("RSI 63", "mean"),
        Above_200EMA   = ("LTP > 200EMA", lambda x: (x == "ABOVE").sum()),
        EMA_Aligned    = ("EMA Aligned", lambda x: (x == "YES").sum()),
        Near_Support   = ("Near Support", lambda x: (x == "YES").sum()),
        Avg_LTP        = ("LTP", "mean"),
    ).round(1).reset_index()

    sec["% Above 200EMA"] = (sec["Above_200EMA"] / sec["Count"] * 100).round(1)
    sec["% EMA Aligned"]  = (sec["EMA_Aligned"]  / sec["Count"] * 100).round(1)
    sec = sec.sort_values("Count", ascending=False)

    def color_pct(val):
        try:
            v = float(val)
            if v >= 60: return "color:#39ff14;font-weight:bold"
            elif v >= 40: return "color:#ffaa00"
            else: return "color:#ff4444"
        except: return ""

    try:
        styled_sec = (
            sec.style
            .map(color_pct, subset=["% Above 200EMA", "% EMA Aligned"])
            .format(precision=1, na_rep="-")
            .set_properties(**{"background-color": "#010f14", "color": "#c8e6f0"})
        )
    except TypeError:
        styled_sec = sec.style.applymap(color_pct, subset=["% Above 200EMA", "% EMA Aligned"]).format(precision=1, na_rep="-")

    st.dataframe(styled_sec, use_container_width=True)
    st.divider()

# ── HEADER ──────────────────────────────────────────────────────────────────
def draw_header():
    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px 0;">
        <h1 style="font-size:2.6rem;letter-spacing:5px;margin-bottom:4px;">
            ⚡ NSE STOCK SCANNER PRO ⚡
        </h1>
        <p style="color:#00ffff80;font-family:'Share Tech Mono',monospace;font-size:0.9rem;letter-spacing:3px;margin:0;">
            NIFTY TOTAL MARKET  ·  SUPPORT ZONES 2020–2022  ·  EMA  ·  RSI  ·  BREADTH
        </p>
        <div style="margin-top:12px;display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">
            <span style="background:#001c25;border:1px solid #00ffff30;border-radius:4px;padding:4px 12px;color:#00ffff;font-size:0.75rem;font-family:'Share Tech Mono'">📡 LIVE DATA</span>
            <span style="background:#001c25;border:1px solid #39ff1430;border-radius:4px;padding:4px 12px;color:#39ff14;font-size:0.75rem;font-family:'Share Tech Mono'">🇮🇳 NSE INDIA</span>
            <span style="background:#001c25;border:1px solid #ff6ec730;border-radius:4px;padding:4px 12px;color:#ff6ec7;font-size:0.75rem;font-family:'Share Tech Mono'">⚙️ POWERED BY YFINANCE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# ── IDLE SCREEN ───────────────────────────────────────────────────────────────
def draw_idle():
    st.markdown("""
    <div style="text-align:center;padding:80px 0;opacity:0.55;">
        <div style="font-size:4rem;margin-bottom:20px;">📡</div>
        <p style="color:#00ffff;font-family:'Orbitron',monospace;font-size:1.1rem;letter-spacing:3px;">
            AWAITING SCAN COMMAND
        </p>
        <p style="color:#a0d8df;font-family:'Share Tech Mono',monospace;font-size:0.85rem;">
            Configure filters in the sidebar → click <strong>START SCANNING</strong>
        </p>
        <div style="margin-top:30px;display:grid;grid-template-columns:repeat(3,1fr);gap:16px;max-width:700px;margin-left:auto;margin-right:auto;">
            <div style="background:#011c22;border:1px solid #00ffff20;border-radius:8px;padding:16px;">
                <div style="color:#00ffff;font-size:1.4rem;">🎯</div>
                <p style="color:#a0d8df;font-size:0.78rem;margin-top:6px;">Support Zones from<br>2020–2022 Monthly Data</p>
            </div>
            <div style="background:#011c22;border:1px solid #39ff1420;border-radius:8px;padding:16px;">
                <div style="color:#39ff14;font-size:1.4rem;">📈</div>
                <p style="color:#a0d8df;font-size:0.78rem;margin-top:6px;">EMA 20/50/200<br>Alignment Check</p>
            </div>
            <div style="background:#011c22;border:1px solid #ff6ec720;border-radius:8px;padding:16px;">
                <div style="color:#ff6ec7;font-size:1.4rem;">🌊</div>
                <p style="color:#a0d8df;font-size:0.78rem;margin-top:6px;">RSI 14/21/63/126/252<br>Multi-Timeframe</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    draw_header()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("### ⚙️ SCANNER CONFIG")
        st.markdown("---")

        max_stocks = st.slider("Max Stocks to Scan", 10, 751, 100, 10,
            help="Larger values take longer. Start with 30 to test.")
        support_threshold = st.slider("Support Proximity %", 1, 25, 2, 1,
            help="How close LTP must be to 2020–22 monthly lows to flag as 'Near Support'.")
        threshold_pct = support_threshold / 100

        st.markdown("---")
        st.markdown("### 🔍 FILTERS")
        filter_near_support = st.checkbox("Near Support Only",  False)
        filter_ema_aligned  = st.checkbox("EMA Aligned Only",   False)
        filter_above_200    = st.checkbox("Above 200 EMA Only", False)
        filter_rsi14_bull   = st.checkbox("RSI14 > 50 Only",    False)

        st.markdown("---")
        st.markdown("### 📊 SORT BY")
        sort_col = st.selectbox("Column", ["RSI 14", "LTP", "% vs 200EMA", "RSI 63", "Symbol"])
        sort_asc = st.checkbox("Ascending", False)

        st.markdown("---")
        st.markdown("""
        <div style='color:#445555;font-size:0.72rem;font-family:Share Tech Mono,monospace;line-height:1.8;'>
        🟢 RSI > 70  Overbought<br>
        🟡 RSI > 50  Bullish<br>
        🟠 RSI < 50  Bearish<br>
        🔴 RSI < 30  Oversold<br>
        ─────────────────────<br>
        ⚡ Near Support = within {threshold}% of<br>
        monthly lows (2020–22)
        </div>
        """.format(threshold=support_threshold), unsafe_allow_html=True)

    # ── LOAD STOCKS ──
    with st.spinner("🔄 Loading Nifty Total Market index..."):
        result = fetch_stock_list()
        if isinstance(result, tuple):
            stocks_df, err = result
        else:
            stocks_df = result
            err = None

    if stocks_df is None or (isinstance(stocks_df, pd.DataFrame) and stocks_df.empty):
        st.error(f"❌ Could not load stock list. Error: {err}")
        st.info("ℹ️ niftyindices.com may block direct requests. Using a fallback sample list.")
        # Minimal fallback
        stocks_df = pd.DataFrame({
            "Symbol": ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL","ITC","KOTAKBANK",
                       "WIPRO","LTIM","AXISBANK","MARUTI","SUNPHARMA","TITAN","NESTLEIND","POWERGRID","NTPC","ULTRACEMCO"],
            "Industry": ["ENERGY","IT","FINANCIALS","IT","FINANCIALS","FMCG","FINANCIALS","TELECOM","FMCG","FINANCIALS",
                         "IT","IT","FINANCIALS","AUTO","PHARMA","CONSUMER","FMCG","UTILITIES","UTILITIES","CEMENT"],
        })
        st.warning(f"Using fallback list of {len(stocks_df)} stocks.")
    else:
        st.success(f"✅ {len(stocks_df)} stocks loaded from Nifty Total Market")

    # Identify columns
    cols_lower = {c.lower(): c for c in stocks_df.columns}
    sym_col = next((cols_lower[k] for k in ["symbol", "sym"] if k in cols_lower), stocks_df.columns[0])
    sec_col = next((cols_lower[k] for k in ["industry", "sector", "ind"] if k in cols_lower), None)

    # Sector quick-filter (sidebar)
    if sec_col:
        all_sectors = sorted(stocks_df[sec_col].dropna().astype(str).unique().tolist())
        with st.sidebar:
            st.markdown("### 🏭 SECTOR FILTER")
            sector_sel = st.multiselect("Sectors (blank = all)", all_sectors)
        if sector_sel:
            stocks_df = stocks_df[stocks_df[sec_col].astype(str).isin(sector_sel)]
    else:
        sector_sel = []

    stocks_df = stocks_df.head(max_stocks)

    # ── SCAN BUTTON ──
    col_l, col_c, col_r = st.columns([1.5, 2, 1.5])
    with col_c:
        scan_btn = st.button("🚀  START SCANNING", use_container_width=True)

    if not scan_btn:
        draw_idle()
        return

    # ── SCANNING ──
    st.markdown(f"""
    <p style="color:#00ffff80;font-family:'Share Tech Mono',monospace;font-size:0.85rem;text-align:center;margin:8px 0;">
        Scanning {len(stocks_df)} stocks · Support proximity: {support_threshold}% · Data from yfinance (NSE)
    </p>
    """, unsafe_allow_html=True)

    prog_bar  = st.progress(0)
    status_ph = st.empty()
    results   = []
    errors    = 0
    total     = len(stocks_df)

    for i, (_, row) in enumerate(stocks_df.iterrows()):
        symbol = str(row[sym_col]).strip().upper()
        sector = str(row[sec_col]).strip() if sec_col else "N/A"

        status_ph.markdown(
            f"<p style='color:#00ffff80;font-family:Share Tech Mono,monospace;font-size:0.8rem;text-align:center;'>"
            f"[{i+1}/{total}] ⚡ Scanning <strong style='color:#00ffff'>{symbol}</strong> — {sector}</p>",
            unsafe_allow_html=True,
        )

        try:
            res = scan_stock(symbol, sector, threshold_pct)
            if res:
                results.append(res)
            else:
                errors += 1
        except Exception:
            errors += 1

        prog_bar.progress((i + 1) / total)
        time.sleep(0.05)

    prog_bar.empty()
    status_ph.empty()

    if not results:
        st.error("❌ No results. Check internet connection or try fewer stocks.")
        return

    df_res = pd.DataFrame(results)

    # ── APPLY FILTERS ──
    df_filt = df_res.copy()
    if filter_near_support: df_filt = df_filt[df_filt["Near Support"] == "YES"]
    if filter_ema_aligned:  df_filt = df_filt[df_filt["EMA Aligned"]  == "YES"]
    if filter_above_200:    df_filt = df_filt[df_filt["LTP > 200EMA"] == "ABOVE"]
    if filter_rsi14_bull:   df_filt = df_filt[df_filt["RSI 14"] > 50]

    if sort_col in df_filt.columns:
        df_filt = df_filt.sort_values(sort_col, ascending=sort_asc)

    # ── BREADTH ──
    show_breadth(df_res)

    # ── RESULTS TABLE ──
    st.markdown(f"## 📋 Scanner Results &nbsp; <span style='color:#00ffff80;font-size:1rem;font-family:Share Tech Mono'>{len(df_filt)} / {len(df_res)} stocks</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#011c22;border:1px solid #00ffff20;border-radius:6px;padding:10px 16px;margin-bottom:12px;font-family:Share Tech Mono,monospace;font-size:0.78rem;display:flex;gap:24px;flex-wrap:wrap;'>
        <span><span style='color:#39ff14'>■</span> RSI ≥ 70 Overbought</span>
        <span><span style='color:#00e600'>■</span> RSI 50-70 Bullish</span>
        <span><span style='color:#ff6666'>■</span> RSI 30-50 Bearish</span>
        <span><span style='color:#ff1111'>■</span> RSI ≤ 30 Oversold</span>
        <span><span style='color:#00ffff'>■</span> Near Support Zone</span>
        <span><span style='color:#39ff14'>■</span> EMA Aligned Bullish</span>
    </div>
    """, unsafe_allow_html=True)

    if df_filt.empty:
        st.warning("⚠️ No stocks match current filters. Try relaxing the filter conditions.")
    else:
        try:
            st.dataframe(style_table(df_filt), use_container_width=True, height=520)
        except Exception:
            st.dataframe(df_filt, use_container_width=True, height=520)

        # Download
        csv_data = df_filt.to_csv(index=False).encode()
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            st.download_button(
                "📥  EXPORT CSV",
                csv_data,
                f"nse_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True,
            )

    st.divider()

    # ── SECTOR BREAKDOWN ──
    show_sector_breakdown(df_filt if not df_filt.empty else df_res)

    # ── FOOTER ──
    if errors > 0:
        st.info(f"ℹ️ {errors} symbols returned no data (delisted / insufficient history / API limit).")
    st.markdown("""
    <p style='text-align:center;color:#334444;font-family:Share Tech Mono,monospace;font-size:0.72rem;margin-top:20px;'>
        ⚠️ FOR EDUCATIONAL & RESEARCH PURPOSES ONLY · NOT INVESTMENT ADVICE · DATA: YAHOO FINANCE / NIFTYINDICES
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
