import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Momentum Leadership Ranking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #e2e8f0;
  }
  .stApp { background-color: #0e1117; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #12161f;
    border-right: 1px solid #1e2433;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

  /* Header */
  .header-block {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
  }
  .header-block h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    color: #f8fafc;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
  }
  .header-block p {
    color: #64748b;
    font-size: 0.82rem;
    margin: 0;
    font-family: 'DM Mono', monospace;
  }

  /* Metric cards */
  .metric-card {
    background: #12161f;
    border: 1px solid #1e2433;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-card .label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'DM Mono', monospace;
  }
  .metric-card .value {
    font-size: 1.8rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    line-height: 1.2;
  }

  /* Zone badges */
  .badge-green { background: #052e16; color: #4ade80; border: 1px solid #166534;
                 padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
                 font-family: 'DM Mono', monospace; font-weight: 500; }
  .badge-blue  { background: #0c1a2e; color: #60a5fa; border: 1px solid #1d4ed8;
                 padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
                 font-family: 'DM Mono', monospace; font-weight: 500; }
  .badge-orange{ background: #2c1503; color: #fb923c; border: 1px solid #9a3412;
                 padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
                 font-family: 'DM Mono', monospace; font-weight: 500; }

  /* Regime pill */
  .regime-on   { background:#052e16; color:#4ade80; border:1px solid #166534;
                 padding:4px 14px; border-radius:20px; font-family:'DM Mono',monospace;
                 font-size:0.8rem; font-weight:500; display:inline-block; }
  .regime-neut { background:#1c1a05; color:#facc15; border:1px solid #854d0e;
                 padding:4px 14px; border-radius:20px; font-family:'DM Mono',monospace;
                 font-size:0.8rem; font-weight:500; display:inline-block; }
  .regime-off  { background:#2d0a0a; color:#f87171; border:1px solid #991b1b;
                 padding:4px 14px; border-radius:20px; font-family:'DM Mono',monospace;
                 font-size:0.8rem; font-weight:500; display:inline-block; }

  /* Dataframe tweaks */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
  div[data-testid="stDataFrame"] table { font-family: 'DM Mono', monospace !important; font-size: 0.78rem; }

  /* Section titles */
  .section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
    border-bottom: 1px solid #1e2433;
    padding-bottom: 6px;
  }

  /* Sidebar labels */
  .stSelectbox label, .stCheckbox label, .stSlider label, .stMultiSelect label {
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.5px;
  }

  /* Progress bar override */
  .stProgress > div > div { background: #3b82f6; }

  /* Sidebar divider */
  .sidebar-divider {
    border: none; border-top: 1px solid #1e2433;
    margin: 1rem 0;
  }

  /* Warning / info boxes */
  .info-box {
    background: #0c1a2e; border: 1px solid #1d4ed8; border-radius: 6px;
    padding: 0.75rem 1rem; font-size: 0.78rem; color: #93c5fd;
    font-family: 'DM Mono', monospace; margin-bottom: 1rem;
  }
  .warn-box {
    background: #1c1205; border: 1px solid #854d0e; border-radius: 6px;
    padding: 0.75rem 1rem; font-size: 0.78rem; color: #fde68a;
    font-family: 'DM Mono', monospace; margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
INDEX_URLS = {
    "Nifty 50":          "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty Next 50":     "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "Nifty 100":         "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 200":         "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "Nifty Midcap 150":  "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "Nifty Smallcap 250":"https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    "Nifty MidSmallcap 400":"https://www.niftyindices.com/IndexConstituent/ind_niftymidsmallcap400list.csv",
    "Nifty Microcap 250":"https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
    "Nifty 500":         "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}

BENCHMARK_TICKERS = {
    "Nifty 50 (NSEI)":  "^NSEI",
    "Nifty 200 (CNX200)":"^CNX200",
    "Nifty 500 (CNX500)":"^CRSLDX",
}

PERIOD_DAYS = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}

# yFinance rate-limits / delists NSE index tickers intermittently.
# Fallback chain tried in order until one returns sufficient data.
BENCHMARK_FALLBACKS = {
    "^NSEI":   ["^NSEI",   "NIFTYBEES.NS"],
    "^CNX200": ["^CNX200", "^NSEI",  "NIFTYBEES.NS"],
    "^CNX500": ["^CRSLDX",  "^CNX500", "^NSEI", "NIFTYBEES.NS"],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com",
}

# ─── DATA FUNCTIONS ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_constituents(index_name: str) -> pd.DataFrame:
    url = INDEX_URLS[index_name]
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        name_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)
        if sym_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["Symbol"] = df[sym_col].str.strip()
        if name_col:
            out["Company"] = df[name_col].str.strip()
        else:
            out["Company"] = out["Symbol"]
        out["YF_Ticker"] = out["Symbol"] + ".NS"
        return out.drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.error(f"Failed to fetch {index_name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_prices(tickers: list, period_days: int) -> pd.DataFrame:
    import datetime
    end = datetime.date.today()
    # Extra buffer so RSI(252) / 200-EMA have warmup even on the 1y view
    start = end - datetime.timedelta(days=period_days + 450)
    try:
        raw = yf.download(
            tickers, start=str(start), end=str(end),
            auto_adjust=True, progress=False, threads=True
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
        close = close.dropna(how="all")
        return close
    except Exception as e:
        st.error(f"Price download error: {e}")
        return pd.DataFrame()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_supertrend(high, low, close, period=7, multiplier=2.5):
    """Simple Supertrend implementation."""
    hl2 = (high + low) / 2
    atr = pd.Series(index=close.index, dtype=float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    for i in range(1, len(close)):
        idx = close.index[i]
        pidx = close.index[i - 1]
        ub = upper_band.iloc[i]
        lb = lower_band.iloc[i]

        # Adjust bands
        if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < supertrend.iloc[i-1] if not pd.isna(supertrend.iloc[i-1]) else False:
            lb = lower_band.iloc[i]
        else:
            lb = max(lower_band.iloc[i], supertrend.get(pidx, lower_band.iloc[i]))

        if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > supertrend.iloc[i-1] if not pd.isna(supertrend.iloc[i-1]) else False:
            ub = upper_band.iloc[i]
        else:
            ub = min(upper_band.iloc[i], supertrend.get(pidx, upper_band.iloc[i]))

        prev_dir = direction.iloc[i-1] if i > 1 else 1
        prev_st = supertrend.iloc[i-1] if not pd.isna(supertrend.iloc[i-1]) else lb

        if prev_dir == -1 and close.iloc[i] > prev_st:
            direction.iloc[i] = 1
            supertrend.iloc[i] = lb
        elif prev_dir == 1 and close.iloc[i] < prev_st:
            direction.iloc[i] = -1
            supertrend.iloc[i] = ub
        else:
            direction.iloc[i] = prev_dir
            supertrend.iloc[i] = lb if prev_dir == 1 else ub

    return supertrend, direction


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Relative Strength Index (RSI) over `period` bars."""
    delta = series.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # If avg_loss is 0 (only gains), RSI = 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def run_momentum_screen(
    prices: pd.DataFrame,
    bench_prices: pd.Series,
    bench_weekly: pd.DataFrame,
    rsi252_min: float,
    rsi88_min: float,
    st_period: int,
    st_multiplier: float,
    use_supertrend: bool,
    liquidity_threshold: float,
    volume_df: pd.DataFrame | None,
    period_days: int,
) -> tuple[pd.DataFrame, dict]:

    results = []
    tickers = [c for c in prices.columns]

    # ── Market Regime ──────────────────────────────────────────────────────────
    bench_close  = bench_prices.dropna() if bench_prices is not None else pd.Series(dtype=float)
    bench_data_ok = len(bench_close) >= 201    # need at least 200 bars for EMA

    if bench_data_ok:
        ema200_bench  = compute_ema(bench_close, 200)
        above_200_ema = bool(bench_close.iloc[-1] > ema200_bench.iloc[-1])
    else:
        ema200_bench  = bench_close  # empty / stub
        above_200_ema = True         # default to permissive when data missing

    # Weekly supertrend on benchmark (configurable period / ATR multiplier)
    bench_weekly_close = bench_weekly["Close"].dropna() if bench_weekly is not None and "Close" in bench_weekly.columns else pd.Series()
    above_supertrend = False
    supertrend_ok = False
    bench_supertrend = None
    if len(bench_weekly_close) > 30:
        try:
            w_h = bench_weekly["High"].dropna()
            w_l = bench_weekly["Low"].dropna()
            w_c = bench_weekly["Close"].dropna()
            st_vals, st_dir = compute_supertrend(w_h, w_l, w_c,
                                                 period=st_period, multiplier=st_multiplier)
            if len(st_vals.dropna()) > 0:
                last_close = w_c.iloc[-1]
                last_st = st_vals.dropna().iloc[-1]
                above_supertrend = bool(last_close > last_st)
                bench_supertrend = float(last_st)
                supertrend_ok = True
        except Exception:
            above_supertrend = False

    # If the Supertrend filter is switched off, treat it as satisfied
    if not use_supertrend:
        above_supertrend = True

    # ── Negative regime gate ─────────────────────────────────────────────────
    # When the Supertrend filter is ON and the benchmark closed BELOW its weekly
    # Supertrend, the index is in a negative regime → do not scan stocks.
    negative_regime = bool(
        use_supertrend and supertrend_ok and bench_data_ok and not above_supertrend
    )

    if not bench_data_ok:
        regime = "NEUTRAL"
    elif negative_regime:
        regime = "RISK OFF"
    elif above_200_ema and above_supertrend:
        regime = "RISK ON"
    elif above_200_ema and not above_supertrend:
        regime = "NEUTRAL"
    else:
        regime = "RISK OFF"

    regime_info = {
        "regime":           regime,
        "bench_close":      bench_close.iloc[-1] if bench_data_ok else None,
        "bench_ema200":     ema200_bench.iloc[-1] if bench_data_ok and len(ema200_bench) > 0 else None,
        "bench_supertrend": bench_supertrend,
        "above_200_ema":    above_200_ema,
        "above_supertrend": above_supertrend,
        "bench_data_ok":    bench_data_ok,
        "supertrend_ok":    supertrend_ok,
        "use_supertrend":   use_supertrend,
        "negative_regime":  negative_regime,
        "st_period":        st_period,
        "st_multiplier":    st_multiplier,
    }

    # Hard stop: negative regime → return no stocks, display layer shows the index
    if negative_regime:
        return pd.DataFrame(), regime_info

    # ── Per-stock calculations ─────────────────────────────────────────────────
    for ticker in tickers:
        try:
            s = prices[ticker].dropna()
            if len(s) < 252:
                continue

            close_last = s.iloc[-1]

            # EMAs
            ema50  = compute_ema(s, 50).iloc[-1]
            ema200 = compute_ema(s, 200).iloc[-1]

            above_ema200 = close_last > ema200
            ema50_gt_200 = ema50 > ema200

            if not (above_ema200 and ema50_gt_200):
                continue

            # ── RSI (Wilder) — 252-day and 88-day ────────────────────────────
            rsi252 = compute_rsi(s, 252).iloc[-1]
            rsi88  = compute_rsi(s, 88).iloc[-1]

            if pd.isna(rsi252) or pd.isna(rsi88):
                continue
            if rsi252 < rsi252_min or rsi88 < rsi88_min:
                continue

            # Returns (skip last month = 21 trading days)
            s_lagged = s.iloc[:-21]
            if len(s_lagged) < 252:
                continue

            ret_9m = (s_lagged.iloc[-1] / s_lagged.iloc[-189] - 1) * 100 if len(s_lagged) >= 189 else np.nan
            ret_6m = (s_lagged.iloc[-1] / s_lagged.iloc[-126] - 1) * 100 if len(s_lagged) >= 126 else np.nan
            ret_3m = (s_lagged.iloc[-1] / s_lagged.iloc[-63]  - 1) * 100 if len(s_lagged) >= 63  else np.nan

            if any(pd.isna(x) for x in [ret_9m, ret_6m, ret_3m]):
                continue

            weighted_mom = 0.40 * ret_9m + 0.30 * ret_6m + 0.30 * ret_3m

            # Volatility
            daily_ret = s.pct_change().dropna().tail(63)
            if len(daily_ret) < 20:
                continue
            vol_3m = daily_ret.std() * np.sqrt(252) * 100
            if vol_3m == 0:
                continue

            # Stock Composite Momentum Score = WeightedMomentum / 3M Volatility
            mom_score = weighted_mom / vol_3m

            # ── Sharpe Ratio (annualised, risk-free = 6.5% INR) × 100 ─────────
            ann_ret  = (s.iloc[-1] / s.iloc[-252] - 1) if len(s) >= 252 else np.nan
            excess   = ann_ret - 0.065 if not pd.isna(ann_ret) else np.nan
            sharpe   = round((excess / (vol_3m / 100)) * 100, 2) if vol_3m > 0 and not pd.isna(excess) else 0.0

            # ── UPI — Ulcer Performance Index ────────────────────────────────
            # UPI = Annualised Return / Ulcer Index ; Ulcer = sqrt(mean(drawdown%^2))
            s_252 = s.tail(252)
            rolling_max = s_252.cummax()
            drawdown_pct = ((s_252 - rolling_max) / rolling_max) * 100   # negative
            ulcer_index  = np.sqrt((drawdown_pct ** 2).mean())
            upi = round((ann_ret * 100) / ulcer_index, 4) if ulcer_index > 0 and not pd.isna(ann_ret) else 0.0

            # ── % Retracement from 52-Week High ──────────────────────────────
            high_52w = s.tail(252).max()
            retracement_pct = round((close_last - high_52w) / high_52w * 100, 2)  # negative

            results.append({
                "Ticker": ticker,
                "Close": round(close_last, 2),
                "EMA50":  round(ema50, 2),
                "EMA200": round(ema200, 2),
                "RSI252": round(rsi252, 1),
                "RSI88":  round(rsi88, 1),
                "Ret_9M%": round(ret_9m, 2),
                "Ret_6M%": round(ret_6m, 2),
                "Ret_3M%": round(ret_3m, 2),
                "WeightedMom": round(weighted_mom, 2),
                "Vol_3M%": round(vol_3m, 2),
                "MomScore": round(mom_score, 4),
                "Sharpe": sharpe,
                "UPI": upi,
                "52W_High": round(high_52w, 2),
                "Retracement%": retracement_pct,
            })

        except Exception:
            continue

    if not results:
        return pd.DataFrame(), regime_info

    df = pd.DataFrame(results)
    N = len(df)
    df = df.sort_values("MomScore", ascending=False).reset_index(drop=True)
    df["Position"] = df.index + 1
    df["FinalRank"] = (100 * (N - df["Position"] + 1) / N).round(1)

    def zone(r):
        if r >= 70: return "🟢 Green"
        elif r >= 40: return "🔵 Blue"
        else:        return "🟠 Orange"

    df["Zone"] = df["FinalRank"].apply(zone)
    return df, regime_info


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:1rem; color:#f8fafc;
                font-weight:600; margin-bottom:0.25rem;">⚡ MLR System</div>
    <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#475569;
                margin-bottom:1.5rem;">Momentum Leadership Ranking v1.0</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Universe</div>', unsafe_allow_html=True)
    indices_universe = st.selectbox(
        "Indices Universe",
        list(INDEX_URLS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    benchmark = st.selectbox(
        "Benchmark",
        list(BENCHMARK_TICKERS.keys()),
        index=2,
    )

    st.markdown('<div class="section-title">Data</div>', unsafe_allow_html=True)
    timeframe = st.selectbox("Timeframe", ["1d"], index=0, help="EOD data only")
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1)

    st.markdown('<div class="section-title">Sort By</div>', unsafe_allow_html=True)
    sort_by = st.selectbox(
        "Sort by",
        ["Final Rank", "UPI (Ulcer Performance)", "Sharpe Ratio", "MomScore", "Retracement% (Closest to ATH)"],
        label_visibility="collapsed",
    )
    sort_map = {
        "Final Rank": "FinalRank",
        "UPI (Ulcer Performance)": "UPI",
        "Sharpe Ratio": "Sharpe",
        "MomScore": "MomScore",
        "Retracement% (Closest to ATH)": "Retracement%",
    }

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">RSI Filters</div>', unsafe_allow_html=True)

    rsi252_min = st.slider("RSI (252-day) Minimum", 0, 100, 50, 1,
        help="Wilder RSI over 252 trading days. Stock must be ≥ this value.")
    rsi88_min  = st.slider("RSI (88-day) Minimum", 0, 100, 50, 1,
        help="Wilder RSI over 88 trading days. Stock must be ≥ this value.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Supertrend Filter</div>', unsafe_allow_html=True)

    use_supertrend = st.checkbox("Enable Supertrend Regime Filter", value=True,
        help="Weekly Supertrend on the selected benchmark gates the scan. "
             "Below Supertrend = negative regime, no stocks shown.")
    st_period = 1
    st_multiplier = 2.5
    if use_supertrend:
        st_period     = st.slider("Supertrend ATR Period", 1, 30, 1, 1)
        st_multiplier = st.slider("Supertrend ATR Multiplier", 0.5, 6.0, 2.5, 0.5)
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#475569;margin-top:-4px">'
            f'On benchmark: <span style="color:#94a3b8">{benchmark.split("(")[0].strip()}</span> (weekly)</div>',
            unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Optional Filters</div>', unsafe_allow_html=True)

    use_liquidity = st.checkbox("Liquidity Filter (ADTV)", value=False,
        help="Require minimum average daily traded value")
    liquidity_crore = 10.0
    if use_liquidity:
        liquidity_crore = st.slider("Min ADTV (₹ Cr)", 5, 50, 20, 5)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portfolio</div>', unsafe_allow_html=True)
    top_n = st.slider("Top N Stocks", 5, 50, 20, 5)
    show_all = st.checkbox("Show All Passing Stocks", value=False)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Screen", width="stretch", type="primary")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <h1>📊 Momentum Leadership Ranking</h1>
  <p>Systematic momentum framework · Nifty universe · Risk-adjusted ranking · EOD signals</p>
</div>
""", unsafe_allow_html=True)

# Info strip
col_i1, col_i2, col_i3, col_i4 = st.columns(4)
with col_i1:
    st.markdown(f'<div class="metric-card"><div class="label">Universe</div><div class="value" style="font-size:1rem;color:#60a5fa">{indices_universe}</div></div>', unsafe_allow_html=True)
with col_i2:
    st.markdown(f'<div class="metric-card"><div class="label">Benchmark</div><div class="value" style="font-size:1rem;color:#a78bfa">{benchmark.split("(")[0].strip()}</div></div>', unsafe_allow_html=True)
with col_i3:
    st.markdown(f'<div class="metric-card"><div class="label">Period</div><div class="value" style="font-size:1.4rem;color:#34d399">{period}</div></div>', unsafe_allow_html=True)
with col_i4:
    st.markdown(f'<div class="metric-card"><div class="label">Portfolio Size</div><div class="value" style="font-size:1.4rem;color:#f472b6">Top {top_n}</div></div>', unsafe_allow_html=True)

st.markdown("")

if not run_btn:
    st.markdown("""
    <div class="info-box">
    ℹ  Configure parameters in the sidebar and click <strong>▶ Run Screen</strong> to start analysis.
    The system fetches live NSE constituent lists and EOD prices via Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)

    # Show system rules summary
    st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**🔴 Market Regime**
- CNX500 > 200 EMA
- CNX500 Weekly > Supertrend (configurable)
- Risk ON / Neutral / Risk OFF

**📈 Stock Trend**
- Close > 200 EMA
- 50 EMA > 200 EMA
""")
    with c2:
        st.markdown("""
**⚡ RSI Filters**
- RSI(252) ≥ slider (default 50)
- RSI(88) ≥ slider (default 50)
- Wilder RSI, both adjustable

**📊 Momentum Score**
- Weighted: 40%×9M + 30%×6M + 30%×3M
- ÷ 3M Volatility (annualized)
- Skip last 21 days (mean reversion)
""")
    with c3:
        st.markdown("""
**🎯 Ranking & Zones**
- 🟢 Green: Rank 70–100 → Portfolio
- 🔵 Blue: Rank 40–69 → Watchlist
- 🟠 Orange: Rank 0–39 → Avoid

**🔄 Rebalance**
- Every 21 trading days (primary)
- Equal weight, 5% per position
""")
    st.stop()

# ─── RUN SCREEN ───────────────────────────────────────────────────────────────
with st.spinner("Fetching constituent list..."):
    constituents = fetch_constituents(indices_universe)

if constituents.empty:
    st.error("Could not fetch constituent list. NSE servers may be blocking the request. Try again.")
    st.stop()

st.success(f"✓ Loaded {len(constituents)} stocks from {indices_universe}")

tickers_yf = constituents["YF_Ticker"].tolist()
bench_yf   = BENCHMARK_TICKERS[benchmark]
period_days_val = PERIOD_DAYS[period]

# Batch download
with st.spinner(f"Downloading {len(tickers_yf)} stock prices ({period})..."):
    prices_df = fetch_prices(tickers_yf, period_days_val)

if prices_df.empty:
    st.error("Price download failed.")
    st.stop()

# Benchmark prices — with fallback chain
with st.spinner("Fetching benchmark data..."):
    import datetime
    end_d   = datetime.date.today()
    start_d = end_d - datetime.timedelta(days=period_days_val + 100)

    def _download_bench(ticker, start, end, interval="1d"):
        """Download a single ticker; return empty DataFrame on any failure."""
        try:
            df = yf.download(ticker, start=str(start), end=str(end),
                             interval=interval, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            return df
        except Exception:
            return pd.DataFrame()

    bench_raw     = pd.DataFrame()
    bench_weekly  = pd.DataFrame()
    bench_used    = bench_yf
    fallback_list = BENCHMARK_FALLBACKS.get(bench_yf, [bench_yf])

    for candidate in fallback_list:
        raw = _download_bench(candidate, start_d, end_d, "1d")
        if not raw.empty and "Close" in raw.columns and len(raw) >= 50:
            bench_raw    = raw
            bench_weekly = _download_bench(candidate, start_d, end_d, "1wk")
            bench_used   = candidate
            break

    if bench_raw.empty or "Close" not in bench_raw.columns:
        st.warning(
            f"⚠️ Benchmark data unavailable for **{benchmark}** (tried: {', '.join(fallback_list)}). "
            "Regime filter will default to NEUTRAL — stock-level screening continues."
        )
        bench_close_series = pd.Series(dtype=float)
        bench_weekly       = pd.DataFrame()
    else:
        bench_close_series = bench_raw["Close"].squeeze().dropna()
        if bench_used != bench_yf:
            st.info(f"ℹ️ Benchmark fallback: using **{bench_used}** as proxy for {benchmark}.")

st.success(f"✓ Prices loaded: {prices_df.shape[1]} stocks × {len(prices_df)} days")

# Run screen
with st.spinner("Computing momentum scores and rankings..."):
    results_df, regime_info = run_momentum_screen(
        prices=prices_df,
        bench_prices=bench_close_series,
        bench_weekly=bench_weekly,
        rsi252_min=rsi252_min,
        rsi88_min=rsi88_min,
        st_period=st_period,
        st_multiplier=st_multiplier,
        use_supertrend=use_supertrend,
        liquidity_threshold=liquidity_crore * 1e7,
        volume_df=None,
        period_days=period_days_val,
    )

# ─── REGIME DISPLAY ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Market Regime</div>', unsafe_allow_html=True)

regime = regime_info["regime"]
if regime == "RISK ON":
    badge = '<span class="regime-on">✅ RISK ON — New Positions Allowed</span>'
elif regime == "NEUTRAL":
    badge = '<span class="regime-neut">⚠️ NEUTRAL — Reduce Exposure</span>'
else:
    badge = '<span class="regime-off">🚫 RISK OFF — No New Positions</span>'

rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
with rc1:
    if not regime_info["bench_data_ok"]:
        st.markdown(badge + ' <span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#64748b">(no bench data — defaulted)</span>', unsafe_allow_html=True)
    else:
        st.markdown(badge, unsafe_allow_html=True)
with rc2:
    ema_color = "#4ade80" if regime_info["above_200_ema"] else "#f87171"
    ema_label = "ABOVE" if regime_info["above_200_ema"] else "BELOW"
    if not regime_info["bench_data_ok"]: ema_label = "N/A"; ema_color = "#475569"
    st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">200 EMA: <span style="color:{ema_color}">{ema_label}</span></div>', unsafe_allow_html=True)
with rc3:
    if not regime_info.get("use_supertrend", True):
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">Supertrend: <span style="color:#475569">OFF</span></div>', unsafe_allow_html=True)
    else:
        st_color = "#4ade80" if regime_info["above_supertrend"] else "#f87171"
        st_label = "ABOVE" if regime_info["above_supertrend"] else "BELOW"
        if not regime_info["bench_data_ok"] or not regime_info.get("supertrend_ok", False):
            st_label = "N/A"; st_color = "#475569"
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">ST({regime_info.get("st_period",1)},{regime_info.get("st_multiplier",2.5)}): <span style="color:{st_color}">{st_label}</span></div>', unsafe_allow_html=True)
with rc4:
    if regime_info["bench_close"] is not None:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">Bench: <span style="color:#f8fafc">{regime_info["bench_close"]:,.2f}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#475569">Bench: N/A</div>', unsafe_allow_html=True)

st.markdown("")

# ─── NEGATIVE REGIME GATE ─────────────────────────────────────────────────────
# Benchmark below its weekly Supertrend (filter on) → show the index, no scan.
if regime_info.get("negative_regime", False):
    bc  = regime_info.get("bench_close")
    bst = regime_info.get("bench_supertrend")
    p   = regime_info.get("st_period", 1)
    m   = regime_info.get("st_multiplier", 2.5)
    gap = ((bc - bst) / bst * 100) if (bc and bst) else None
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#2d0a0a 0%,#3d1010 100%);
                border:1px solid #991b1b;border-radius:10px;padding:1.75rem 2rem;
                margin-bottom:1rem;">
      <div style="font-family:'DM Mono',monospace;font-size:1.25rem;color:#fca5a5;
                  font-weight:600;margin-bottom:0.5rem;">🚫 NEGATIVE REGIME — Scan Halted</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#f87171;line-height:1.7;">
        <span style="color:#fecaca">{benchmark}</span> closed
        <span style="color:#fca5a5;font-weight:600">BELOW</span> its weekly
        Supertrend({p}, {m}). No new long positions — the index is in a downtrend.
      </div>
      <div style="display:flex;gap:2.5rem;margin-top:1rem;
                  font-family:'DM Mono',monospace;font-size:0.82rem;">
        <div><div style="color:#7f1d1d;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Benchmark Close</div>
             <div style="color:#fecaca;font-size:1.2rem;font-weight:600">{bc:,.2f}</div></div>
        <div><div style="color:#7f1d1d;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Weekly Supertrend</div>
             <div style="color:#fca5a5;font-size:1.2rem;font-weight:600">{bst:,.2f}</div></div>
        <div><div style="color:#7f1d1d;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Distance</div>
             <div style="color:#f87171;font-size:1.2rem;font-weight:600">{gap:+.2f}%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div class="warn-box">📉 Per the strategy rules, stock scanning is disabled while the '
        'benchmark is below its weekly Supertrend. Re-run once the index reclaims the Supertrend, '
        'or disable the Supertrend filter in the sidebar to scan regardless.</div>',
        unsafe_allow_html=True)
    st.stop()

if regime_info["regime"] == "RISK OFF":
    st.markdown('<div class="warn-box">⚠️ Market is in Risk OFF regime. System recommends no new positions.</div>', unsafe_allow_html=True)

if results_df.empty:
    st.warning("No stocks passed all filters. Try relaxing conditions or choose a different period.")
    st.stop()

# ─── SUMMARY METRICS ──────────────────────────────────────────────────────────
green = results_df[results_df["Zone"] == "🟢 Green"]
blue  = results_df[results_df["Zone"] == "🔵 Blue"]
orange= results_df[results_df["Zone"] == "🟠 Orange"]

st.markdown('<div class="section-title">Screening Summary</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="metric-card"><div class="label">Passed Filters</div><div class="value" style="color:#f8fafc">{len(results_df)}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="label">🟢 Green Zone</div><div class="value" style="color:#4ade80">{len(green)}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="label">🔵 Blue Zone</div><div class="value" style="color:#60a5fa">{len(blue)}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="label">🟠 Orange Zone</div><div class="value" style="color:#fb923c">{len(orange)}</div></div>', unsafe_allow_html=True)
with m5:
    avg_rsi = results_df["RSI252"].mean()
    st.markdown(f'<div class="metric-card"><div class="label">Avg RSI(252)</div><div class="value" style="color:#a78bfa">{avg_rsi:.1f}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ─── SORT & DISPLAY TABLE ─────────────────────────────────────────────────────
sort_col = sort_map[sort_by]
# Retracement% is negative; values closest to 0 are nearest ATH → sort ascending
sort_ascending = (sort_col == "Retracement%")
results_df = results_df.sort_values(sort_col, ascending=sort_ascending).reset_index(drop=True)

# Map ticker back to company name
ticker_to_name = dict(zip(constituents["YF_Ticker"], constituents["Company"]))
results_df["Company"] = results_df["Ticker"].map(ticker_to_name).fillna(results_df["Ticker"])
results_df["Symbol"] = results_df["Ticker"].str.replace(".NS", "", regex=False)

# TradingView URL embedded directly on the Symbol column (rendered as a link).
# Raw symbol kept after NSE%3A so the regex display_text shows the clean ticker.
from urllib.parse import quote as _quote
results_df["SymbolLink"] = results_df["Symbol"].apply(
    lambda s: f"https://www.tradingview.com/chart/?symbol=NSE%3A{s}"
)

display_df = results_df[["SymbolLink", "Company", "Zone", "FinalRank",
                          "MomScore", "UPI", "Sharpe",
                          "Retracement%", "52W_High",
                          "RSI252", "RSI88",
                          "Ret_9M%", "Ret_6M%", "Ret_3M%",
                          "Vol_3M%", "Close"]].copy()

display_df = display_df.rename(columns={
    "SymbolLink":   "Symbol",
    "FinalRank":    "Rank",
    "52W_High":     "52W High",
})

display_limit = len(results_df) if show_all else top_n
display_df_final = display_df.head(display_limit)

# ── Zone filter tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    f"🏆 Top {top_n} Portfolio",
    f"🟢 Green ({len(green)})",
    f"🔵 Blue ({len(blue)})",
    f"🟠 Orange ({len(orange)})",
])

def style_zone(val):
    if "Green" in str(val):
        return "color: #4ade80; font-weight: 600"
    elif "Blue" in str(val):
        return "color: #60a5fa; font-weight: 600"
    elif "Orange" in str(val):
        return "color: #fb923c; font-weight: 600"
    return ""

def render_table(df):
    fmt = {
        "Rank":         "{:.1f}",
        "MomScore":     "{:.4f}",
        "UPI":          "{:.3f}",
        "Sharpe":       "{:.1f}",
        "Retracement%": "{:+.2f}%",
        "52W High":     "₹{:.2f}",
        "RSI252":       "{:.1f}",
        "RSI88":        "{:.1f}",
        "Ret_9M%":      "{:+.2f}%",
        "Ret_6M%":      "{:+.2f}%",
        "Ret_3M%":      "{:+.2f}%",
        "Vol_3M%":      "{:.2f}%",
        "Close":        "₹{:.2f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in df.columns}

    def color_retracement(val):
        try:
            v = float(val)
            if v >= -5:    return "color: #4ade80; font-weight:600"
            elif v >= -15: return "color: #facc15"
            else:          return "color: #f87171"
        except: return ""

    def color_upi(val):
        try:
            v = float(val)
            if v >= 1.5:   return "color: #4ade80; font-weight:600"
            elif v >= 0.5: return "color: #facc15"
            else:          return "color: #fb923c"
        except: return ""

    def color_sharpe(val):   # Sharpe is ×100 scaled
        try:
            v = float(val)
            if v >= 100:   return "color: #4ade80; font-weight:600"
            elif v >= 0.0: return "color: #facc15"
            else:          return "color: #f87171"
        except: return ""

    def color_rsi(val):
        try:
            v = float(val)
            if v >= 70:   return "color: #4ade80; font-weight:600"
            elif v >= 50: return "color: #facc15"
            else:         return "color: #fb923c"
        except: return ""

    # Build the Styler on the FULL frame (Symbol URL included) so column
    # positions stay consistent through render. Styling ops only touch named
    # subsets; the Symbol URL column rides along and is read by column_config.
    styled = df.style.map(style_zone, subset=["Zone"])
    if "Retracement%" in df.columns:
        styled = styled.map(color_retracement, subset=["Retracement%"])
    if "UPI" in df.columns:
        styled = styled.map(color_upi, subset=["UPI"])
    if "Sharpe" in df.columns:
        styled = styled.map(color_sharpe, subset=["Sharpe"])
    for rcol in ("RSI252", "RSI88"):
        if rcol in df.columns:
            styled = styled.map(color_rsi, subset=[rcol])
    styled = (
        styled
        .format(fmt, na_rep="—")
        .background_gradient(subset=["Rank"], cmap="RdYlGn", vmin=0, vmax=100)
        .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "12px"})
    )

    col_cfg = {}
    if "Symbol" in df.columns:
        col_cfg["Symbol"] = st.column_config.LinkColumn(
            "Symbol",
            help="Open on TradingView",
            display_text=r"NSE%3A(.+)$",   # show clean ticker, link to TV
            width="small",
        )

    st.dataframe(styled, column_config=col_cfg, width="stretch", height=520)

with tab1:
    top_df = display_df.head(top_n)
    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;margin-bottom:0.5rem'>Top {top_n} stocks sorted by <strong style='color:#94a3b8'>{sort_by}</strong> · Equal weight 5% each</div>", unsafe_allow_html=True)
    render_table(top_df)

with tab2:
    g_df = display_df[display_df["Zone"] == "🟢 Green"]
    if g_df.empty:
        st.info("No stocks in Green zone with current filters.")
    else:
        render_table(g_df)

with tab3:
    b_df = display_df[display_df["Zone"] == "🔵 Blue"]
    if b_df.empty:
        st.info("No stocks in Blue zone.")
    else:
        render_table(b_df)

with tab4:
    o_df = display_df[display_df["Zone"] == "🟠 Orange"]
    if o_df.empty:
        st.info("No stocks in Orange zone.")
    else:
        render_table(o_df)

# ─── CHARTS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Zone Distribution**")
    zone_counts = pd.DataFrame({
        "Zone": ["🟢 Green", "🔵 Blue", "🟠 Orange"],
        "Count": [len(green), len(blue), len(orange)],
        "Color": ["#4ade80", "#60a5fa", "#fb923c"],
    })
    st.bar_chart(zone_counts.set_index("Zone")["Count"])

with chart_col2:
    st.markdown("**Top 20: Momentum Score Distribution**")
    top20 = results_df.head(20)[["Symbol", "MomScore"]].set_index("Symbol")
    st.bar_chart(top20)

# RSI / quality map (built from results_df so Symbol stays a plain ticker here)
st.markdown('<div class="section-title">RSI &amp; Quality Map — Top Candidates</div>', unsafe_allow_html=True)
rs_df = results_df.head(top_n)[["Symbol", "RSI252", "RSI88", "UPI", "Sharpe", "Retracement%", "Zone"]].copy()
st.dataframe(
    rs_df.style.background_gradient(subset=["RSI252", "RSI88"], cmap="RdYlGn", vmin=0, vmax=100)
              .map(style_zone, subset=["Zone"])
              .format({"RSI252":"{:.1f}", "RSI88":"{:.1f}",
                       "UPI":"{:.3f}", "Sharpe":"{:.1f}", "Retracement%":"{:+.2f}%"}, na_rep="—")
              .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "12px"}),
    width="stretch",
    height=300,
)

# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)
with dl1:
    csv_full = results_df.to_csv(index=False)
    st.download_button("⬇ Download Full Results (CSV)", csv_full,
                       file_name=f"momentum_screen_{indices_universe.replace(' ','_')}_{period}.csv",
                       mime="text/csv", width="stretch")
with dl2:
    csv_top = results_df.head(top_n).to_csv(index=False)
    st.download_button(f"⬇ Download Top {top_n} Portfolio (CSV)", csv_top,
                       file_name=f"portfolio_top{top_n}_{period}.csv",
                       mime="text/csv", width="stretch")

# ─── RULES FOOTER ─────────────────────────────────────────────────────────────
with st.expander("📋 Entry / Exit Rules Reference"):
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
**Entry Rules** *(all must be true)*
- Benchmark > 200 EMA  
- Benchmark Weekly > Supertrend (configurable)  
- Stock Close > 200 EMA  
- 50 EMA > 200 EMA  
- RSI(252) ≥ threshold  
- RSI(88) ≥ threshold  
- Rank ≥ 70  
""")
    with r2:
        st.markdown("""
**Exit Rules** *(any one triggers exit)*
- Close < 200 EMA  
- Rank < 40  
- CNX500 Weekly Close < Weekly Supertrend  
  → *Reduce by 50% or move to cash*
""")
