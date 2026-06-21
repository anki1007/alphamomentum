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
    page_title="US Momentum Leadership Ranking",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
  }
  .stApp { background-color: #080c14; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1a2035;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

  /* Header */
  .header-block {
    background: linear-gradient(135deg, #060b18 0%, #0f1a30 50%, #0d1525 100%);
    border: 1px solid #1a2d4a;
    border-top: 2px solid #3b82f6;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
  }
  .header-block::before {
    content: "★ ★ ★";
    position: absolute;
    top: 1.5rem;
    right: 2rem;
    font-size: 0.6rem;
    color: #1e3a5f;
    letter-spacing: 4px;
  }
  .header-block h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    color: #f8fafc;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
  }
  .header-block p {
    color: #4a6080;
    font-size: 0.82rem;
    margin: 0;
    font-family: 'DM Mono', monospace;
  }

  /* Metric cards */
  .metric-card {
    background: #0d1220;
    border: 1px solid #1a2035;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-card .label {
    font-size: 0.72rem;
    color: #4a6080;
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

  /* ADR badge */
  .adr-yes { background:#052e16; color:#4ade80; border:1px solid #166534;
             padding:1px 8px; border-radius:12px; font-size:0.72rem;
             font-family:'DM Mono',monospace; font-weight:500; }
  .adr-no  { background:#1a1a1a; color:#475569; border:1px solid #2a2a2a;
             padding:1px 8px; border-radius:12px; font-size:0.72rem;
             font-family:'DM Mono',monospace; }

  /* Dataframe tweaks */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
  div[data-testid="stDataFrame"] table { font-family: 'DM Mono', monospace !important; font-size: 0.78rem; }

  /* Section titles */
  .section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #2a3d5a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
    border-bottom: 1px solid #1a2035;
    padding-bottom: 6px;
  }

  /* Sidebar labels */
  .stSelectbox label, .stCheckbox label, .stSlider label, .stMultiSelect label {
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.5px;
  }

  /* Progress bar */
  .stProgress > div > div { background: #3b82f6; }

  /* Sidebar divider */
  .sidebar-divider { border: none; border-top: 1px solid #1a2035; margin: 1rem 0; }

  /* Info / warn */
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

# Hard-coded DJIA 30 constituents (stable list; update periodically)
DJIA_30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]

INDEX_UNIVERSES = {
    "S&P 500":   "sp500",
    "DJIA 30":   "djia30",
    "Nasdaq 100":"nasdaq100",
    "S&P 400 Midcap": "sp400",
    "Russell 1000": "russell1000",
}

BENCHMARK_TICKERS = {
    "S&P 500 (SPX)":  "^GSPC",
    "DJIA 30":        "^DJI",
    "Nasdaq 100":     "^NDX",
    "Russell 2000":   "^RUT",
}

PERIOD_DAYS = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}

BENCHMARK_FALLBACKS = {
    "^GSPC": ["^GSPC", "SPY"],
    "^DJI":  ["^DJI",  "DIA"],
    "^NDX":  ["^NDX",  "QQQ"],
    "^RUT":  ["^RUT",  "IWM"],
}

# ─── UNIVERSE FETCHERS ────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_constituents():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        out = pd.DataFrame()
        out["Symbol"]  = df["Symbol"].str.strip().str.replace(".", "-", regex=False)
        out["Company"] = df["Security"].str.strip()
        out["Sector"]  = df["GICS Sector"].str.strip()
        return out.drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.error(f"S&P 500 fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_nasdaq100_constituents():
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for t in tables:
            if "Ticker" in t.columns or "Symbol" in t.columns:
                sym_col = "Ticker" if "Ticker" in t.columns else "Symbol"
                name_col = next((c for c in t.columns if "company" in c.lower() or "name" in c.lower()), None)
                out = pd.DataFrame()
                out["Symbol"] = t[sym_col].str.strip()
                out["Company"] = t[name_col].str.strip() if name_col else out["Symbol"]
                out["Sector"] = ""
                return out.dropna(subset=["Symbol"]).drop_duplicates("Symbol").reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Nasdaq 100 fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_sp400_constituents():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        tables = pd.read_html(url)
        df = tables[0]
        sym_col = next((c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower()), None)
        name_col = next((c for c in df.columns if "company" in c.lower() or "security" in c.lower() or "name" in c.lower()), None)
        out = pd.DataFrame()
        out["Symbol"]  = df[sym_col].str.strip() if sym_col else pd.Series(dtype=str)
        out["Company"] = df[name_col].str.strip() if name_col else out["Symbol"]
        out["Sector"]  = ""
        return out.dropna(subset=["Symbol"]).drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.error(f"S&P 400 fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_russell1000_constituents():
    """Russell 1000 = top 1000 US stocks. Approximate via iShares IWB holdings."""
    try:
        url = "https://www.ishares.com/us/products/239707/ISHARES-RUSSELL-1000-ETF/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        raw = r.text
        # iShares CSV has metadata rows before the header
        lines = raw.split("\n")
        header_idx = next(i for i, l in enumerate(lines) if "Ticker" in l or "Name" in l)
        df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
        df.columns = [c.strip() for c in df.columns]
        sym_col = next((c for c in df.columns if "ticker" in c.lower()), None)
        name_col = next((c for c in df.columns if "name" in c.lower()), None)
        if sym_col is None:
            raise ValueError("No ticker column")
        out = pd.DataFrame()
        out["Symbol"]  = df[sym_col].str.strip()
        out["Company"] = df[name_col].str.strip() if name_col else out["Symbol"]
        out["Sector"]  = ""
        # Filter to equities only (drop cash, ETFs etc. — usually have '-' in symbol)
        out = out[out["Symbol"].str.match(r'^[A-Z]{1,5}$', na=False)]
        return out.drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.warning(f"Russell 1000 iShares download failed ({e}); falling back to S&P 500.")
        return fetch_sp500_constituents()

@st.cache_data(ttl=86400)
def fetch_djia30_constituents():
    out = pd.DataFrame({
        "Symbol": DJIA_30,
        "Company": DJIA_30,  # Will be enriched from yf info if needed
        "Sector": "",
    })
    return out

def fetch_constituents(index_name: str) -> pd.DataFrame:
    dispatch = {
        "S&P 500":         fetch_sp500_constituents,
        "DJIA 30":         fetch_djia30_constituents,
        "Nasdaq 100":      fetch_nasdaq100_constituents,
        "S&P 400 Midcap":  fetch_sp400_constituents,
        "Russell 1000":    fetch_russell1000_constituents,
    }
    fn = dispatch.get(index_name, fetch_sp500_constituents)
    df = fn()
    if not df.empty:
        df["YF_Ticker"] = df["Symbol"]   # US tickers need no suffix
    return df

# ─── PRICE FETCH ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices(tickers: list, period_days: int) -> pd.DataFrame:
    import datetime
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=period_days + 120)
    try:
        raw = yf.download(
            tickers, start=str(start), end=str(end),
            auto_adjust=True, progress=False, threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw
        close = close.dropna(how="all")
        return close
    except Exception as e:
        st.error(f"Price download error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_ohlcv(tickers: list, period_days: int) -> dict:
    """Returns dict with 'High', 'Low', 'Close', 'Volume' DataFrames for ADR calc."""
    import datetime
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=period_days + 120)
    try:
        raw = yf.download(
            tickers, start=str(start), end=str(end),
            auto_adjust=True, progress=False, threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            return {
                "High":   raw["High"],
                "Low":    raw["Low"],
                "Close":  raw["Close"],
                "Volume": raw["Volume"],
            }
        else:
            # single ticker fallback
            return {
                "High":   raw[["High"]]   if "High"   in raw.columns else pd.DataFrame(),
                "Low":    raw[["Low"]]    if "Low"    in raw.columns else pd.DataFrame(),
                "Close":  raw[["Close"]]  if "Close"  in raw.columns else pd.DataFrame(),
                "Volume": raw[["Volume"]] if "Volume" in raw.columns else pd.DataFrame(),
            }
    except Exception as e:
        st.error(f"OHLCV download error: {e}")
        return {"High": pd.DataFrame(), "Low": pd.DataFrame(), "Close": pd.DataFrame(), "Volume": pd.DataFrame()}

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_adr(high_df: pd.DataFrame, low_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average Daily Range % = mean((H-L)/((H+L)/2)) over window, expressed as %."""
    try:
        hl_range_pct = ((high_df - low_df) / ((high_df + low_df) / 2)) * 100
        return hl_range_pct.tail(window).mean()
    except Exception:
        return pd.Series(dtype=float)


def compute_supertrend(high, low, close, period=7, multiplier=2.5):
    hl2 = (high + low) / 2
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    direction  = pd.Series(index=close.index, dtype=int)

    for i in range(1, len(close)):
        ub = upper_band.iloc[i]
        lb = lower_band.iloc[i]

        prev_st = supertrend.iloc[i - 1] if not pd.isna(supertrend.iloc[i - 1]) else lb
        prev_dir = direction.iloc[i - 1] if i > 1 else 1

        if lower_band.iloc[i] > lower_band.iloc[i - 1] or close.iloc[i - 1] < prev_st:
            lb = lower_band.iloc[i]
        else:
            lb = max(lower_band.iloc[i], prev_st)

        if upper_band.iloc[i] < upper_band.iloc[i - 1] or close.iloc[i - 1] > prev_st:
            ub = upper_band.iloc[i]
        else:
            ub = min(upper_band.iloc[i], prev_st)

        if prev_dir == -1 and close.iloc[i] > prev_st:
            direction.iloc[i]  = 1
            supertrend.iloc[i] = lb
        elif prev_dir == 1 and close.iloc[i] < prev_st:
            direction.iloc[i]  = -1
            supertrend.iloc[i] = ub
        else:
            direction.iloc[i]  = prev_dir
            supertrend.iloc[i] = lb if prev_dir == 1 else ub

    return supertrend, direction


# ─── CORE SCREEN ─────────────────────────────────────────────────────────────
def run_momentum_screen(
    prices: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    bench_prices: pd.Series,
    bench_weekly: pd.DataFrame,
    use_composite_rs: bool,
    composite_rs_threshold: float,
    adr_window: int,
) -> tuple[pd.DataFrame, dict]:

    results = []
    tickers = list(prices.columns)

    ret_252 = prices.pct_change(252).iloc[-1]
    ret_88  = prices.pct_change(88).iloc[-1]
    universe_ret_252 = ret_252.dropna()
    universe_ret_88  = ret_88.dropna()

    # ── Market Regime ─────────────────────────────────────────────────────────
    bench_close  = bench_prices.dropna() if bench_prices is not None else pd.Series(dtype=float)
    bench_data_ok = len(bench_close) >= 201

    if bench_data_ok:
        ema200_bench  = compute_ema(bench_close, 200)
        above_200_ema = bool(bench_close.iloc[-1] > ema200_bench.iloc[-1])
    else:
        ema200_bench  = bench_close
        above_200_ema = True

    bench_weekly_close = bench_weekly["Close"].dropna() if (bench_weekly is not None and "Close" in bench_weekly.columns) else pd.Series()
    above_supertrend = False
    if len(bench_weekly_close) > 30:
        try:
            st_vals, st_dir = compute_supertrend(
                bench_weekly["High"].dropna(),
                bench_weekly["Low"].dropna(),
                bench_weekly["Close"].dropna(),
                period=7, multiplier=2.5,
            )
            if len(st_vals.dropna()) > 0:
                above_supertrend = bench_weekly_close.iloc[-1] > st_vals.dropna().iloc[-1]
        except Exception:
            above_supertrend = False

    if above_200_ema and above_supertrend:
        regime = "RISK ON"
    elif above_200_ema and not above_supertrend:
        regime = "NEUTRAL"
    else:
        regime = "RISK OFF"

    if not bench_data_ok:
        regime = "NEUTRAL"

    regime_info = {
        "regime":           regime,
        "bench_close":      bench_close.iloc[-1]  if bench_data_ok else None,
        "bench_ema200":     ema200_bench.iloc[-1] if bench_data_ok and len(ema200_bench) > 0 else None,
        "above_200_ema":    above_200_ema,
        "above_supertrend": above_supertrend,
        "bench_data_ok":    bench_data_ok,
    }

    # Pre-compute universe-wide ADR
    adr_series = pd.Series(dtype=float)
    if not high_df.empty and not low_df.empty:
        try:
            adr_series = compute_adr(high_df, low_df, window=adr_window)
        except Exception:
            pass

    # ── Per-stock ────────────────────────────────────────────────────────────
    for ticker in tickers:
        try:
            s = prices[ticker].dropna()
            if len(s) < 252:
                continue

            close_last = s.iloc[-1]
            ema50  = compute_ema(s, 50).iloc[-1]
            ema200 = compute_ema(s, 200).iloc[-1]

            if not (close_last > ema200 and ema50 > ema200):
                continue

            # RS
            rs252 = (universe_ret_252 < ret_252.get(ticker, np.nan)).sum() / len(universe_ret_252) * 100 if len(universe_ret_252) >= 5 else 50.0
            rs88  = (universe_ret_88  < ret_88.get(ticker,  np.nan)).sum() / len(universe_ret_88)  * 100 if len(universe_ret_88)  >= 5 else 50.0

            composite_rs = 0.60 * rs252 + 0.40 * rs88

            if use_composite_rs:
                if composite_rs <= composite_rs_threshold:
                    continue
            else:
                if rs252 <= 50 or rs88 <= 50:
                    continue

            # Momentum returns — skip last 21 days
            s_lag = s.iloc[:-21]
            if len(s_lag) < 252:
                continue

            ret_9m = (s_lag.iloc[-1] / s_lag.iloc[-189] - 1) * 100 if len(s_lag) >= 189 else np.nan
            ret_6m = (s_lag.iloc[-1] / s_lag.iloc[-126] - 1) * 100 if len(s_lag) >= 126 else np.nan
            ret_3m = (s_lag.iloc[-1] / s_lag.iloc[-63]  - 1) * 100 if len(s_lag) >= 63  else np.nan

            if any(pd.isna(x) for x in [ret_9m, ret_6m, ret_3m]):
                continue

            weighted_mom = 0.40 * ret_9m + 0.30 * ret_6m + 0.30 * ret_3m

            daily_ret = s.pct_change().dropna().tail(63)
            if len(daily_ret) < 20:
                continue
            vol_3m = daily_ret.std() * np.sqrt(252) * 100
            if vol_3m == 0:
                continue

            mom_score = weighted_mom / vol_3m

            # Sharpe (rf = 5.25% USD, approximate Fed funds)
            ann_ret = (s.iloc[-1] / s.iloc[-252] - 1) if len(s) >= 252 else np.nan
            excess  = ann_ret - 0.0525 if not pd.isna(ann_ret) else np.nan
            sharpe  = round(excess / (vol_3m / 100), 3) if vol_3m > 0 and not pd.isna(excess) else 0.0

            # UPI
            s_252 = s.tail(252)
            rolling_max  = s_252.cummax()
            drawdown_pct = ((s_252 - rolling_max) / rolling_max) * 100
            ulcer_index  = np.sqrt((drawdown_pct ** 2).mean())
            upi = round((ann_ret * 100) / ulcer_index, 4) if ulcer_index > 0 and not pd.isna(ann_ret) else 0.0

            # 52W retracement
            high_52w      = s.tail(252).max()
            retracement   = round((close_last - high_52w) / high_52w * 100, 2)

            # ADR%
            adr_val = adr_series.get(ticker, np.nan) if not adr_series.empty else np.nan
            adr_flag = bool(not pd.isna(adr_val) and adr_val > 3.0)

            results.append({
                "Ticker":      ticker,
                "Close":       round(close_last, 2),
                "EMA50":       round(ema50, 2),
                "EMA200":      round(ema200, 2),
                "RS252":       round(rs252, 1),
                "RS88":        round(rs88, 1),
                "CompositeRS": round(composite_rs, 1),
                "Ret_9M%":     round(ret_9m, 2),
                "Ret_6M%":     round(ret_6m, 2),
                "Ret_3M%":     round(ret_3m, 2),
                "WeightedMom": round(weighted_mom, 2),
                "Vol_3M%":     round(vol_3m, 2),
                "MomScore":    round(mom_score, 4),
                "Sharpe":      sharpe,
                "UPI":         upi,
                "52W_High":    round(high_52w, 2),
                "Retracement%": retracement,
                "ADR%":        round(adr_val, 2) if not pd.isna(adr_val) else 0.0,
                "ADR>3%":      "✅ Yes" if adr_flag else "—",
            })

        except Exception:
            continue

    if not results:
        return pd.DataFrame(), regime_info

    df = pd.DataFrame(results)
    N  = len(df)
    df = df.sort_values("MomScore", ascending=False).reset_index(drop=True)
    df["Position"]  = df.index + 1
    df["FinalRank"] = (100 * (N - df["Position"] + 1) / N).round(1)

    def zone(r):
        if r >= 70:   return "🟢 Green"
        elif r >= 40: return "🔵 Blue"
        else:         return "🟠 Orange"

    df["Zone"] = df["FinalRank"].apply(zone)
    return df, regime_info


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:1rem; color:#f8fafc;
                font-weight:600; margin-bottom:0.25rem;">🇺🇸 MLR · US Markets</div>
    <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#2a3d5a;
                margin-bottom:1.5rem;">Momentum Leadership Ranking v1.0</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Universe</div>', unsafe_allow_html=True)
    indices_universe = st.selectbox(
        "Stock Universe",
        list(INDEX_UNIVERSES.keys()),
        index=0,
        label_visibility="collapsed",
    )

    benchmark = st.selectbox("Benchmark", list(BENCHMARK_TICKERS.keys()), index=0)

    st.markdown('<div class="section-title">Data</div>', unsafe_allow_html=True)
    period = st.selectbox("Lookback Period", ["1y", "2y", "3y", "5y"], index=1)

    st.markdown('<div class="section-title">Sort By</div>', unsafe_allow_html=True)
    sort_by = st.selectbox(
        "Sort by",
        ["Final Rank", "UPI (Ulcer Performance)", "Sharpe Ratio",
         "MomScore", "Retracement% (Closest to ATH)", "ADR% (Highest Volatility)"],
        label_visibility="collapsed",
    )
    sort_map = {
        "Final Rank":                  ("FinalRank",   False),
        "UPI (Ulcer Performance)":     ("UPI",         False),
        "Sharpe Ratio":                ("Sharpe",      False),
        "MomScore":                    ("MomScore",    False),
        "Retracement% (Closest to ATH)":("Retracement%", True),
        "ADR% (Highest Volatility)":   ("ADR%",        False),
    }

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)

    use_composite = st.checkbox("Composite RS Filter", value=False,
        help="0.60×RS252 + 0.40×RS88 > threshold")
    composite_threshold = 60.0
    if use_composite:
        composite_threshold = st.slider("Composite RS Threshold", 50, 80, 60, 1)

    adr_filter = st.checkbox("Show only ADR > 3% stocks", value=False,
        help="Filter to stocks whose 20-day Average Daily Range exceeds 3%")

    adr_window = st.slider("ADR Lookback (days)", 5, 60, 20, 5,
        help="Window for computing Average Daily Range %")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portfolio</div>', unsafe_allow_html=True)
    top_n    = st.slider("Top N Stocks", 5, 50, 20, 5)
    show_all = st.checkbox("Show All Passing Stocks", value=False)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Screen", use_container_width=True, type="primary")


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <h1>🇺🇸 US Momentum Leadership Ranking</h1>
  <p>Systematic momentum · S&P 500 / DJIA / Nasdaq universe · Risk-adjusted ranking · EOD signals · ADR volatility filter</p>
</div>
""", unsafe_allow_html=True)

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
    Constituent lists are fetched from Wikipedia (S&P 500 / Nasdaq 100) and EOD prices from Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**🔴 Market Regime**
- Benchmark > 200-day EMA
- Benchmark Weekly > Supertrend(7,2.5)
- Risk ON / Neutral / Risk OFF

**📈 Stock Trend Filters**
- Close > 200 EMA
- 50 EMA > 200 EMA
""")
    with c2:
        st.markdown("""
**⚡ Relative Strength**
- RS252 > 50th percentile
- RS88 > 50th percentile
- Optional Composite RS > threshold

**📊 Momentum Score**
- Weighted: 40%×9M + 30%×6M + 30%×3M
- ÷ 3M Realized Volatility
- Skip last 21 days (mean reversion)
""")
    with c3:
        st.markdown("""
**📐 ADR% Column (New)**
- Avg Daily Range % = mean((H-L)/mid×100) over N days
- ✅ Yes → ADR > 3% (high volatility / active)
- Useful for options & momentum traders

**🎯 Zones**
- 🟢 Green: Rank 70–100 → Portfolio
- 🔵 Blue: Rank 40–69 → Watchlist
- 🟠 Orange: Rank 0–39 → Avoid
""")
    st.stop()

# ─── RUN SCREEN ───────────────────────────────────────────────────────────────
with st.spinner("Fetching constituent list..."):
    constituents = fetch_constituents(indices_universe)

if constituents.empty:
    st.error("Could not fetch constituent list. Check network access.")
    st.stop()

st.success(f"✓ Loaded {len(constituents)} stocks from {indices_universe}")

tickers_yf      = constituents["YF_Ticker"].tolist()
bench_yf        = BENCHMARK_TICKERS[benchmark]
period_days_val = PERIOD_DAYS[period]

with st.spinner(f"Downloading {len(tickers_yf)} stock prices (OHLCV) for {period}..."):
    ohlcv = fetch_ohlcv(tickers_yf, period_days_val)
    prices_df = ohlcv.get("Close", pd.DataFrame())
    high_df   = ohlcv.get("High",  pd.DataFrame())
    low_df    = ohlcv.get("Low",   pd.DataFrame())

if prices_df.empty:
    st.error("Price download failed.")
    st.stop()

# Benchmark
with st.spinner("Fetching benchmark data..."):
    import datetime
    end_d   = datetime.date.today()
    start_d = end_d - datetime.timedelta(days=period_days_val + 120)

    def _dl_bench(ticker, start, end, interval="1d"):
        try:
            df = yf.download(ticker, start=str(start), end=str(end),
                             interval=interval, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            return df
        except Exception:
            return pd.DataFrame()

    bench_raw    = pd.DataFrame()
    bench_weekly = pd.DataFrame()
    bench_used   = bench_yf
    fallbacks    = BENCHMARK_FALLBACKS.get(bench_yf, [bench_yf])

    for candidate in fallbacks:
        raw = _dl_bench(candidate, start_d, end_d, "1d")
        if not raw.empty and "Close" in raw.columns and len(raw) >= 50:
            bench_raw    = raw
            bench_weekly = _dl_bench(candidate, start_d, end_d, "1wk")
            bench_used   = candidate
            break

    if bench_raw.empty or "Close" not in bench_raw.columns:
        st.warning(f"⚠️ Benchmark data unavailable. Regime defaults to NEUTRAL.")
        bench_close_series = pd.Series(dtype=float)
        bench_weekly       = pd.DataFrame()
    else:
        bench_close_series = bench_raw["Close"].squeeze().dropna()
        if bench_used != bench_yf:
            st.info(f"ℹ️ Benchmark fallback: using **{bench_used}** as proxy for {benchmark}.")

st.success(f"✓ Prices: {prices_df.shape[1]} stocks × {len(prices_df)} days")

with st.spinner("Computing momentum scores, ADR, and rankings..."):
    results_df, regime_info = run_momentum_screen(
        prices       = prices_df,
        high_df      = high_df,
        low_df       = low_df,
        bench_prices = bench_close_series,
        bench_weekly = bench_weekly,
        use_composite_rs        = use_composite,
        composite_rs_threshold  = composite_threshold,
        adr_window   = adr_window,
    )

# ── ADR filter ────────────────────────────────────────────────────────────────
if adr_filter and not results_df.empty:
    results_df = results_df[results_df["ADR>3%"] == "✅ Yes"].reset_index(drop=True)
    N = len(results_df)
    if N > 0:
        results_df["FinalRank"] = (100 * (N - results_df.index) / N).round(1)
    st.info(f"ℹ️ ADR > 3% filter active: {N} stocks retained.")

# ─── REGIME ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Market Regime</div>', unsafe_allow_html=True)

regime = regime_info["regime"]
badge = {
    "RISK ON":  '<span class="regime-on">✅ RISK ON — New Positions Allowed</span>',
    "NEUTRAL":  '<span class="regime-neut">⚠️ NEUTRAL — Reduce Exposure</span>',
    "RISK OFF": '<span class="regime-off">🚫 RISK OFF — No New Positions</span>',
}.get(regime, "")

rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
with rc1:
    st.markdown(badge, unsafe_allow_html=True)
with rc2:
    ema_color = "#4ade80" if regime_info["above_200_ema"] else "#f87171"
    ema_label = ("ABOVE" if regime_info["above_200_ema"] else "BELOW") if regime_info["bench_data_ok"] else "N/A"
    st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">200 EMA: <span style="color:{ema_color}">{ema_label}</span></div>', unsafe_allow_html=True)
with rc3:
    st_color = "#4ade80" if regime_info["above_supertrend"] else "#f87171"
    st_label = ("ABOVE" if regime_info["above_supertrend"] else "BELOW") if regime_info["bench_data_ok"] else "N/A"
    st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">Supertrend: <span style="color:{st_color}">{st_label}</span></div>', unsafe_allow_html=True)
with rc4:
    if regime_info["bench_close"] is not None:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">Bench: <span style="color:#f8fafc">{regime_info["bench_close"]:,.2f}</span></div>', unsafe_allow_html=True)

if regime == "RISK OFF":
    st.markdown('<div class="warn-box">⚠️ Market is in Risk OFF regime. System recommends no new positions.</div>', unsafe_allow_html=True)

st.markdown("")

if results_df.empty:
    st.warning("No stocks passed all filters. Try relaxing conditions or choosing a different period.")
    st.stop()

# ─── SUMMARY METRICS ──────────────────────────────────────────────────────────
green  = results_df[results_df["Zone"] == "🟢 Green"]
blue   = results_df[results_df["Zone"] == "🔵 Blue"]
orange = results_df[results_df["Zone"] == "🟠 Orange"]
adr_high = results_df[results_df["ADR>3%"] == "✅ Yes"]

st.markdown('<div class="section-title">Screening Summary</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f'<div class="metric-card"><div class="label">Passed Filters</div><div class="value" style="color:#f8fafc">{len(results_df)}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="label">🟢 Green</div><div class="value" style="color:#4ade80">{len(green)}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="label">🔵 Blue</div><div class="value" style="color:#60a5fa">{len(blue)}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="label">🟠 Orange</div><div class="value" style="color:#fb923c">{len(orange)}</div></div>', unsafe_allow_html=True)
with m5:
    avg_rs = results_df["CompositeRS"].mean()
    st.markdown(f'<div class="metric-card"><div class="label">Avg Composite RS</div><div class="value" style="color:#a78bfa">{avg_rs:.1f}</div></div>', unsafe_allow_html=True)
with m6:
    st.markdown(f'<div class="metric-card"><div class="label">ADR &gt; 3%</div><div class="value" style="color:#4ade80">{len(adr_high)}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ─── SORT ─────────────────────────────────────────────────────────────────────
sort_col, sort_asc = sort_map[sort_by]
results_df = results_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

ticker_to_name   = dict(zip(constituents["YF_Ticker"], constituents["Company"]))
ticker_to_sector = dict(zip(constituents["YF_Ticker"], constituents.get("Sector", pd.Series(dtype=str))))
results_df["Company"] = results_df["Ticker"].map(ticker_to_name).fillna(results_df["Ticker"])
results_df["Sector"]  = results_df["Ticker"].map(ticker_to_sector).fillna("")
results_df["Symbol"]  = results_df["Ticker"]
results_df["TV"] = results_df["Symbol"].apply(
    lambda s: f"https://www.tradingview.com/chart/?symbol={s}"
)

display_df = results_df[[
    "Symbol", "Company", "Sector", "TV", "Zone", "FinalRank",
    "MomScore", "UPI", "Sharpe",
    "ADR%", "ADR>3%",
    "Retracement%", "52W_High",
    "RS252", "RS88", "CompositeRS",
    "Ret_9M%", "Ret_6M%", "Ret_3M%",
    "Vol_3M%", "Close",
]].rename(columns={
    "FinalRank":    "Rank",
    "52W_High":     "52W High",
    "TV":           "Chart",
})

display_limit    = len(results_df) if show_all else top_n
display_df_final = display_df.head(display_limit)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"🏆 Top {top_n} Portfolio",
    f"🟢 Green ({len(green)})",
    f"🔵 Blue ({len(blue)})",
    f"🟠 Orange ({len(orange)})",
    f"📐 ADR > 3% ({len(adr_high)})",
])

def style_zone(val):
    if "Green"  in str(val): return "color:#4ade80;font-weight:600"
    if "Blue"   in str(val): return "color:#60a5fa;font-weight:600"
    if "Orange" in str(val): return "color:#fb923c;font-weight:600"
    return ""

def style_adr_flag(val):
    if "✅" in str(val):   return "color:#4ade80;font-weight:600"
    return "color:#334155"

def render_table(df):
    fmt = {
        "Rank":         "{:.1f}",
        "MomScore":     "{:.4f}",
        "UPI":          "{:.3f}",
        "Sharpe":       "{:.3f}",
        "ADR%":         "{:.2f}%",
        "Retracement%": "{:+.2f}%",
        "52W High":     "${:.2f}",
        "RS252":        "{:.1f}",
        "RS88":         "{:.1f}",
        "CompositeRS":  "{:.1f}",
        "Ret_9M%":      "{:+.2f}%",
        "Ret_6M%":      "{:+.2f}%",
        "Ret_3M%":      "{:+.2f}%",
        "Vol_3M%":      "{:.2f}%",
        "Close":        "${:.2f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in df.columns}

    def color_retracement(val):
        try:
            v = float(val)
            if v >= -5:    return "color:#4ade80;font-weight:600"
            elif v >= -15: return "color:#facc15"
            else:          return "color:#f87171"
        except: return ""

    def color_upi(val):
        try:
            v = float(val)
            if v >= 1.5:   return "color:#4ade80;font-weight:600"
            elif v >= 0.5: return "color:#facc15"
            else:          return "color:#fb923c"
        except: return ""

    def color_sharpe(val):
        try:
            v = float(val)
            if v >= 1.0:   return "color:#4ade80;font-weight:600"
            elif v >= 0.0: return "color:#facc15"
            else:          return "color:#f87171"
        except: return ""

    def color_adr(val):
        try:
            v = float(str(val).replace("%", ""))
            if v > 5:      return "color:#f87171;font-weight:600"
            elif v > 3:    return "color:#facc15;font-weight:600"
            else:          return "color:#94a3b8"
        except: return ""

    has_chart = "Chart" in df.columns
    data_cols = [c for c in df.columns if c != "Chart"]
    df_data   = df[data_cols].copy()

    styled = (
        df_data.style
        .applymap(style_zone,        subset=["Zone"])
        .applymap(style_adr_flag,    subset=["ADR>3%"])
        .applymap(color_retracement, subset=["Retracement%"])
        .applymap(color_upi,         subset=["UPI"])
        .applymap(color_sharpe,      subset=["Sharpe"])
        .applymap(color_adr,         subset=["ADR%"])
        .format(fmt, na_rep="—")
        .background_gradient(subset=["Rank"], cmap="RdYlGn", vmin=0, vmax=100)
        .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "12px"})
    )

    if has_chart:
        styled.data["Chart"] = df["Chart"].values

    col_cfg = {}
    if has_chart:
        col_cfg["Chart"] = st.column_config.LinkColumn(
            "📈 Chart",
            help="Open on TradingView",
            display_text="TradingView ↗",
            width="small",
        )

    st.dataframe(styled, column_config=col_cfg, use_container_width=True, height=520)

with tab1:
    top_df = display_df.head(top_n)
    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;margin-bottom:0.5rem'>Top {top_n} sorted by <strong style='color:#94a3b8'>{sort_by}</strong> · Equal weight 5%</div>", unsafe_allow_html=True)
    render_table(top_df)

with tab2:
    g_df = display_df[display_df["Zone"] == "🟢 Green"]
    render_table(g_df) if not g_df.empty else st.info("No stocks in Green zone.")

with tab3:
    b_df = display_df[display_df["Zone"] == "🔵 Blue"]
    render_table(b_df) if not b_df.empty else st.info("No stocks in Blue zone.")

with tab4:
    o_df = display_df[display_df["Zone"] == "🟠 Orange"]
    render_table(o_df) if not o_df.empty else st.info("No stocks in Orange zone.")

with tab5:
    a_df = display_df[display_df["ADR>3%"] == "✅ Yes"]
    if a_df.empty:
        st.info(f"No stocks with ADR > 3% over the selected {adr_window}-day window.")
    else:
        st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;margin-bottom:0.5rem'>{len(a_df)} stocks with Average Daily Range &gt; 3% (window: {adr_window} days) — ideal for options &amp; active momentum traders</div>", unsafe_allow_html=True)
        render_table(a_df)

# ─── CHARTS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("**Zone Distribution**")
    zone_counts = pd.DataFrame({
        "Zone":  ["🟢 Green", "🔵 Blue", "🟠 Orange"],
        "Count": [len(green), len(blue), len(orange)],
    })
    st.bar_chart(zone_counts.set_index("Zone")["Count"])

with chart_col2:
    st.markdown(f"**Top {min(20, top_n)}: Momentum Score**")
    top20 = results_df.head(min(20, top_n))[["Symbol", "MomScore"]].set_index("Symbol")
    st.bar_chart(top20)

# ADR distribution chart
st.markdown('<div class="section-title">ADR% Distribution — All Passing Stocks</div>', unsafe_allow_html=True)
if "ADR%" in results_df.columns:
    adr_dist = results_df[["Symbol", "ADR%"]].head(min(40, len(results_df))).set_index("Symbol")
    st.bar_chart(adr_dist)

# RS Map
st.markdown('<div class="section-title">Relative Strength Map — Top Candidates</div>', unsafe_allow_html=True)
rs_df = display_df.head(top_n)[["Symbol", "Company", "RS252", "RS88", "CompositeRS", "ADR%", "ADR>3%", "UPI", "Sharpe", "Retracement%", "Zone"]].copy()
st.dataframe(
    rs_df.style
    .background_gradient(subset=["RS252","RS88","CompositeRS"], cmap="RdYlGn", vmin=0, vmax=100)
    .applymap(style_zone,     subset=["Zone"])
    .applymap(style_adr_flag, subset=["ADR>3%"])
    .format({
        "RS252":"{:.1f}", "RS88":"{:.1f}", "CompositeRS":"{:.1f}",
        "ADR%":"{:.2f}%", "UPI":"{:.3f}", "Sharpe":"{:.3f}", "Retracement%":"{:+.2f}%",
    }, na_rep="—")
    .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "12px"}),
    use_container_width=True, height=350,
)

# ─── SECTOR BREAKDOWN ─────────────────────────────────────────────────────────
if "Sector" in results_df.columns and results_df["Sector"].any():
    st.markdown('<div class="section-title">Sector Breakdown — Top Portfolio</div>', unsafe_allow_html=True)
    sector_df = results_df.head(top_n)["Sector"].value_counts().reset_index()
    sector_df.columns = ["Sector", "Count"]
    st.bar_chart(sector_df.set_index("Sector")["Count"])

# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇ Download Full Results (CSV)",
        results_df.to_csv(index=False),
        file_name=f"us_momentum_{indices_universe.replace(' ','_')}_{period}.csv",
        mime="text/csv", use_container_width=True,
    )
with dl2:
    st.download_button(
        f"⬇ Download Top {top_n} Portfolio (CSV)",
        results_df.head(top_n).to_csv(index=False),
        file_name=f"us_portfolio_top{top_n}_{period}.csv",
        mime="text/csv", use_container_width=True,
    )

# ─── RULES FOOTER ─────────────────────────────────────────────────────────────
with st.expander("📋 Entry / Exit Rules Reference"):
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
**Entry Rules** *(all must hold)*
- Benchmark > 200 EMA
- Benchmark Weekly > Supertrend(7,2.5)
- Stock Close > 200 EMA
- 50 EMA > 200 EMA
- RS252 > 50th percentile
- RS88 > 50th percentile
- Rank ≥ 70
""")
    with r2:
        st.markdown("""
**Exit Rules** *(any one triggers)*
- Close < 200 EMA
- Rank < 40
- Benchmark Weekly < Weekly Supertrend
  → Reduce 50% or move to cash

**Risk-Free Rate**: 5.25% (USD)
**Sharpe** = (AnnRet − 5.25%) / AnnVol
""")
    with r3:
        st.markdown("""
**ADR% Definition**
- ADR% = mean((H−L)/((H+L)/2) × 100) over N days
- > 3% → active/volatile stock  
- > 5% → very high intraday range  
- Useful for: options traders needing premium, momentum scalers

**Rebalance**: Every 21 trading days
**Sizing**: Equal weight 5% per position
""")
