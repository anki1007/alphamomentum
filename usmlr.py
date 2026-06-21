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
    page_title="US Momentum Leadership Ranking v2",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:#080c14; color:#e2e8f0; }
  .stApp { background-color:#080c14; }
  section[data-testid="stSidebar"] { background:#0d1220; border-right:1px solid #1a2035; }
  section[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }

  .header-block {
    background:linear-gradient(135deg,#060b18 0%,#0f1a30 50%,#0d1525 100%);
    border:1px solid #1a2d4a; border-top:2px solid #3b82f6;
    border-radius:10px; padding:1.5rem 2rem; margin-bottom:1.5rem; position:relative;
  }
  .header-block::before { content:"★ ★ ★"; position:absolute; top:1.4rem; right:2rem;
                          font-size:0.6rem; color:#1e3a5f; letter-spacing:4px; }
  .header-block h1 { font-family:'DM Mono',monospace; font-size:1.5rem; color:#f8fafc;
                     margin:0 0 0.25rem 0; letter-spacing:-0.5px; }
  .header-block p  { color:#4a6080; font-size:0.82rem; margin:0; font-family:'DM Mono',monospace; }

  .metric-card { background:#0d1220; border:1px solid #1a2035; border-radius:8px;
                 padding:0.9rem 1rem; text-align:center; }
  .metric-card .label { font-size:0.68rem; color:#4a6080; text-transform:uppercase;
                        letter-spacing:1px; font-family:'DM Mono',monospace; }
  .metric-card .value { font-size:1.6rem; font-weight:600; font-family:'DM Mono',monospace; line-height:1.2; }

  .regime-on   { background:#052e16; color:#4ade80; border:1px solid #166534; padding:4px 14px;
                 border-radius:20px; font-family:'DM Mono',monospace; font-size:0.8rem;
                 font-weight:500; display:inline-block; }
  .regime-neut { background:#1c1a05; color:#facc15; border:1px solid #854d0e; padding:4px 14px;
                 border-radius:20px; font-family:'DM Mono',monospace; font-size:0.8rem;
                 font-weight:500; display:inline-block; }
  .regime-off  { background:#2d0a0a; color:#f87171; border:1px solid #991b1b; padding:4px 14px;
                 border-radius:20px; font-family:'DM Mono',monospace; font-size:0.8rem;
                 font-weight:500; display:inline-block; }

  .stDataFrame { border-radius:8px; overflow:hidden; }
  div[data-testid="stDataFrame"] table { font-family:'DM Mono',monospace !important; font-size:0.78rem; }

  .section-title { font-family:'DM Mono',monospace; font-size:0.72rem; color:#2a3d5a;
                   text-transform:uppercase; letter-spacing:2px; margin-bottom:0.75rem;
                   margin-top:1.5rem; border-bottom:1px solid #1a2035; padding-bottom:6px; }

  .stSelectbox label, .stCheckbox label, .stSlider label, .stMultiSelect label {
    font-size:0.78rem !important; color:#94a3b8 !important;
    font-family:'DM Mono',monospace !important; letter-spacing:0.5px;
  }
  .stProgress > div > div { background:#3b82f6; }
  .sidebar-divider { border:none; border-top:1px solid #1a2035; margin:1rem 0; }

  .info-box  { background:#0c1a2e; border:1px solid #1d4ed8; border-radius:6px;
               padding:0.75rem 1rem; font-size:0.78rem; color:#93c5fd;
               font-family:'DM Mono',monospace; margin-bottom:1rem; }
  .warn-box  { background:#1c1205; border:1px solid #854d0e; border-radius:6px;
               padding:0.75rem 1rem; font-size:0.78rem; color:#fde68a;
               font-family:'DM Mono',monospace; margin-bottom:1rem; }
  .vix-card  { background:#1a0a2e; border:1px solid #7c3aed; border-radius:8px;
               padding:0.75rem 1.2rem; font-size:0.8rem; color:#c4b5fd;
               font-family:'DM Mono',monospace; display:inline-flex; align-items:center; gap:0.6rem; }
  .tag-new   { background:#0c2a4e; color:#38bdf8; border:1px solid #0369a1;
               padding:1px 7px; border-radius:4px; font-size:0.65rem;
               font-family:'DM Mono',monospace; vertical-align:middle; margin-left:4px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
INDEX_UNIVERSES = {
    "S&P 500":         "sp500",
    "DJIA 30":         "djia30",
    "Nasdaq 100":      "nasdaq100",
    "S&P 400 Midcap":  "sp400",
    "Russell 1000":    "russell1000",
}

# DJIA 30 — stable hardcoded list (update periodically)
DJIA_30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]

# SPX & DJIA listed first (primary requested benchmarks)
BENCHMARK_TICKERS = {
    "S&P 500 (SPX)":  "^GSPC",
    "DJIA 30 (DJI)":  "^DJI",
    "Nasdaq 100":     "^NDX",
    "Russell 2000":   "^RUT",
}

BENCHMARK_FALLBACKS = {
    "^GSPC": ["^GSPC", "SPY"],
    "^DJI":  ["^DJI",  "DIA"],
    "^NDX":  ["^NDX",  "QQQ"],
    "^RUT":  ["^RUT",  "IWM"],
}

PERIOD_DAYS = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ─── UNIVERSE FETCHERS ────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_constituents() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
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
def fetch_nasdaq100_constituents() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cols = [c for c in t.columns]
            sym_col = "Ticker" if "Ticker" in cols else ("Symbol" if "Symbol" in cols else None)
            if sym_col:
                name_col = next((c for c in cols if "company" in str(c).lower() or "name" in str(c).lower()), None)
                out = pd.DataFrame()
                out["Symbol"]  = t[sym_col].astype(str).str.strip()
                out["Company"] = t[name_col].astype(str).str.strip() if name_col else out["Symbol"]
                out["Sector"]  = ""
                return out.dropna(subset=["Symbol"]).drop_duplicates("Symbol").reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Nasdaq 100 fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_sp400_constituents() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
        df = tables[0]
        sym_col  = next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), None)
        name_col = next((c for c in df.columns if "company" in str(c).lower() or "security" in str(c).lower()), None)
        out = pd.DataFrame()
        out["Symbol"]  = df[sym_col].astype(str).str.strip()
        out["Company"] = df[name_col].astype(str).str.strip() if name_col else out["Symbol"]
        out["Sector"]  = ""
        return out.dropna(subset=["Symbol"]).drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.error(f"S&P 400 fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_russell1000_constituents() -> pd.DataFrame:
    """iShares IWB holdings; falls back to S&P 500 on failure."""
    try:
        url = ("https://www.ishares.com/us/products/239707/"
               "ISHARES-RUSSELL-1000-ETF/1467271812596.ajax"
               "?fileType=csv&fileName=IWB_holdings&dataType=fund")
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        lines = r.text.split("\n")
        hdr = next(i for i, l in enumerate(lines) if "Ticker" in l or "Name" in l)
        df = pd.read_csv(StringIO("\n".join(lines[hdr:])))
        df.columns = [c.strip() for c in df.columns]
        sym_col  = next((c for c in df.columns if "ticker" in c.lower()), None)
        name_col = next((c for c in df.columns if "name" in c.lower()), None)
        if sym_col is None:
            raise ValueError("no ticker column")
        out = pd.DataFrame()
        out["Symbol"]  = df[sym_col].astype(str).str.strip()
        out["Company"] = df[name_col].astype(str).str.strip() if name_col else out["Symbol"]
        out["Sector"]  = ""
        out = out[out["Symbol"].str.match(r'^[A-Z]{1,5}$', na=False)]
        return out.drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.warning(f"Russell 1000 download failed ({e}); using S&P 500.")
        return fetch_sp500_constituents()

@st.cache_data(ttl=86400)
def fetch_djia30_constituents() -> pd.DataFrame:
    return pd.DataFrame({"Symbol": DJIA_30, "Company": DJIA_30, "Sector": ""})

def fetch_constituents(index_name: str) -> pd.DataFrame:
    dispatch = {
        "S&P 500":        fetch_sp500_constituents,
        "DJIA 30":        fetch_djia30_constituents,
        "Nasdaq 100":     fetch_nasdaq100_constituents,
        "S&P 400 Midcap": fetch_sp400_constituents,
        "Russell 1000":   fetch_russell1000_constituents,
    }
    df = dispatch.get(index_name, fetch_sp500_constituents)()
    if not df.empty:
        df["YF_Ticker"] = df["Symbol"]   # US tickers need no suffix
    return df


@st.cache_data(ttl=3600)
def fetch_ohlcv(tickers: tuple, period_days: int) -> dict:
    """Download OHLCV; returns dict of High/Low/Close DataFrames (Close used for screen, H/L for ADR)."""
    import datetime
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=period_days + 450)  # warmup for long EMAs
    try:
        raw = yf.download(list(tickers), start=str(start), end=str(end),
                          auto_adjust=True, progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            return {"High": raw["High"], "Low": raw["Low"],
                    "Close": raw["Close"].dropna(how="all")}
        return {
            "High":  raw[["High"]]  if "High"  in raw.columns else pd.DataFrame(),
            "Low":   raw[["Low"]]   if "Low"   in raw.columns else pd.DataFrame(),
            "Close": (raw[["Close"]] if "Close" in raw.columns else raw).dropna(how="all"),
        }
    except Exception as e:
        st.error(f"Price download error: {e}")
        return {"High": pd.DataFrame(), "Low": pd.DataFrame(), "Close": pd.DataFrame()}


@st.cache_data(ttl=1800)
def fetch_vix() -> float | None:
    """Fetch current CBOE VIX. Returns None if unavailable."""
    for ticker in ["^VIX", "^VIX9D", "VIXY"]:
        try:
            df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            if "Close" in df.columns and len(df) > 0:
                val = float(df["Close"].dropna().iloc[-1])
                if val > 0:
                    return val
        except Exception:
            continue
    return None


# ─── COMPUTATION FUNCTIONS ────────────────────────────────────────────────────
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0, 100.0)


def compute_adr(high_df: pd.DataFrame, low_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average Daily Range % = mean((H-L)/((H+L)/2)*100) over window."""
    try:
        hl_pct = ((high_df - low_df) / ((high_df + low_df) / 2)) * 100
        return hl_pct.tail(window).mean()
    except Exception:
        return pd.Series(dtype=float)


def compute_supertrend_np(close, high, low, period=7, multiplier=2.5):
    """Vectorised numpy Supertrend. Returns (supertrend, direction); +1 bull / -1 bear."""
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.zeros(n, dtype=np.int8)

    prev_c = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])

    atr = np.zeros(n, dtype=np.float64)
    atr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha

    hl2      = (high + low) / 2.0
    ub_basic = hl2 + multiplier * atr
    lb_basic = hl2 - multiplier * atr

    ub = np.zeros(n); lb = np.zeros(n); st = np.zeros(n); di = np.zeros(n, dtype=np.int8)
    ub[0] = ub_basic[0]; lb[0] = lb_basic[0]; st[0] = ub_basic[0]; di[0] = -1

    for i in range(1, n):
        ub[i] = ub_basic[i] if (ub_basic[i] < ub[i-1] or close[i-1] > ub[i-1]) else ub[i-1]
        lb[i] = lb_basic[i] if (lb_basic[i] > lb[i-1] or close[i-1] < lb[i-1]) else lb[i-1]
        if di[i-1] == -1:
            di[i] = 1  if close[i] > ub[i-1] else -1
        else:
            di[i] = -1 if close[i] < lb[i-1] else  1
        st[i] = lb[i] if di[i] == 1 else ub[i]

    return st, di


def supertrend_last(weekly_df, period=7, multiplier=2.5):
    needed = ["High", "Low", "Close"]
    if weekly_df is None or weekly_df.empty or not all(c in weekly_df.columns for c in needed):
        return None, False
    w = weekly_df[needed].dropna()
    if len(w) < period + 5:
        return None, False
    st, di = compute_supertrend_np(w["Close"].values, w["High"].values, w["Low"].values,
                                    period=period, multiplier=multiplier)
    last_st  = float(st[-1])
    is_above = (int(di[-1]) == 1) and (float(w["Close"].iloc[-1]) > last_st)
    return last_st, is_above


# ─── CORE MOMENTUM SCREEN ────────────────────────────────────────────────────
def run_momentum_screen(
    prices, bench_prices, bench_weekly,
    rsi252_min, rsi88_min, st_period, st_multiplier, use_supertrend, period_days,
    bench_6m_ret=0.0, bench_3m_ret=0.0,
    require_positive_rs=True, require_acceleration=False, require_mom_consistency=False,
    max_retracement=-30.0, max_rolling_dd=-20.0,
    adr_series=None,
):
    results = []

    # ── Market regime ─────────────────────────────────────────────────────────
    bench_close   = bench_prices.dropna() if bench_prices is not None else pd.Series(dtype=float)
    bench_data_ok = len(bench_close) >= 201

    if bench_data_ok:
        ema200_bench  = compute_ema(bench_close, 200)
        above_200_ema = bool(bench_close.iloc[-1] > ema200_bench.iloc[-1])
    else:
        ema200_bench  = bench_close
        above_200_ema = True

    bench_supertrend, above_supertrend, supertrend_ok = None, False, False
    if bench_weekly is not None and len(bench_weekly) > 30:
        try:
            bench_supertrend, above_supertrend = supertrend_last(bench_weekly, st_period, st_multiplier)
            supertrend_ok = bench_supertrend is not None
        except Exception:
            pass

    if not use_supertrend:
        above_supertrend = True

    negative_regime = bool(use_supertrend and supertrend_ok and bench_data_ok and not above_supertrend)

    if not bench_data_ok:        regime = "NEUTRAL"
    elif negative_regime:        regime = "RISK OFF"
    elif above_200_ema and above_supertrend: regime = "RISK ON"
    elif above_200_ema:          regime = "NEUTRAL"
    else:                        regime = "RISK OFF"

    regime_info = dict(
        regime=regime,
        bench_close=float(bench_close.iloc[-1]) if bench_data_ok else None,
        bench_ema200=float(ema200_bench.iloc[-1]) if bench_data_ok and len(ema200_bench) > 0 else None,
        bench_supertrend=bench_supertrend,
        above_200_ema=above_200_ema, above_supertrend=above_supertrend,
        bench_data_ok=bench_data_ok, supertrend_ok=supertrend_ok,
        use_supertrend=use_supertrend, negative_regime=negative_regime,
        st_period=st_period, st_multiplier=st_multiplier,
    )

    if negative_regime:
        return pd.DataFrame(), regime_info

    has_adr = adr_series is not None and not adr_series.empty

    # ── Per-stock loop ──────────────────────────────────────────────────────────
    for ticker in prices.columns:
        try:
            s = prices[ticker].dropna()
            if len(s) < 252:
                continue

            close_last = float(s.iloc[-1])
            ema50  = float(compute_ema(s, 50).iloc[-1])
            ema200 = float(compute_ema(s, 200).iloc[-1])

            if not (close_last > ema200 and ema50 > ema200):
                continue

            rsi252 = float(compute_rsi(s, 252).iloc[-1])
            rsi88  = float(compute_rsi(s, 88).iloc[-1])
            if pd.isna(rsi252) or pd.isna(rsi88):
                continue
            if rsi252 < rsi252_min or rsi88 < rsi88_min:
                continue

            s_lag = s.iloc[:-21]
            if len(s_lag) < 252:
                continue

            ret_9m = (s_lag.iloc[-1] / s_lag.iloc[-189] - 1) * 100 if len(s_lag) >= 189 else np.nan
            ret_6m = (s_lag.iloc[-1] / s_lag.iloc[-126] - 1) * 100 if len(s_lag) >= 126 else np.nan
            ret_3m = (s_lag.iloc[-1] / s_lag.iloc[-63]  - 1) * 100 if len(s_lag) >= 63  else np.nan
            if any(pd.isna(x) for x in [ret_9m, ret_6m, ret_3m]):
                continue

            if require_mom_consistency and (ret_9m < 0 or ret_6m < 0 or ret_3m < 0):
                continue

            monthly_3m, monthly_6m = ret_3m / 3.0, ret_6m / 6.0
            accelerating = monthly_3m > monthly_6m
            if require_acceleration and not accelerating:
                continue

            rs_6m = ret_6m - bench_6m_ret
            rs_3m = ret_3m - bench_3m_ret
            if require_positive_rs and rs_6m < 0:
                continue

            weighted_mom = 0.40 * ret_9m + 0.30 * ret_6m + 0.30 * ret_3m

            daily_ret = s.pct_change().dropna().tail(63)
            if len(daily_ret) < 20:
                continue
            vol_3m = float(daily_ret.std() * np.sqrt(252) * 100)
            if vol_3m == 0:
                continue

            s_63        = s.tail(63).values
            roll_max_63 = np.maximum.accumulate(s_63)
            max_dd_63   = float(np.min((s_63 - roll_max_63) / roll_max_63) * 100)
            if max_dd_63 < max_rolling_dd:
                continue

            mom_score = weighted_mom / vol_3m

            ann_ret = float(s.iloc[-1] / s.iloc[-252] - 1) if len(s) >= 252 else np.nan
            excess  = ann_ret - 0.0525 if not pd.isna(ann_ret) else np.nan   # Rf = 5.25% USD
            sharpe  = round((excess / (vol_3m / 100)) * 100, 2) if not pd.isna(excess) else 0.0

            s_252 = s.tail(252)
            rolling_max  = s_252.cummax()
            drawdown_pct = ((s_252 - rolling_max) / rolling_max) * 100
            ulcer_index  = float(np.sqrt((drawdown_pct ** 2).mean()))
            upi = round((ann_ret * 100) / ulcer_index, 4) if ulcer_index > 0 and not pd.isna(ann_ret) else 0.0

            high_52w        = float(s.tail(252).max())
            retracement_pct = round((close_last - high_52w) / high_52w * 100, 2)
            if retracement_pct < max_retracement:
                continue

            adr_val  = float(adr_series.get(ticker, np.nan)) if has_adr else np.nan
            adr_flag = bool(not pd.isna(adr_val) and adr_val > 3.0)

            results.append({
                "Ticker":       ticker,
                "Close":        round(close_last, 2),
                "EMA50":        round(ema50, 2),
                "EMA200":       round(ema200, 2),
                "RSI252":       round(rsi252, 1),
                "RSI88":        round(rsi88, 1),
                "Ret_9M%":      round(ret_9m, 2),
                "Ret_6M%":      round(ret_6m, 2),
                "Ret_3M%":      round(ret_3m, 2),
                "RS_6M":        round(rs_6m, 2),
                "RS_3M":        round(rs_3m, 2),
                "Accel":        "✓" if accelerating else "–",
                "WeightedMom":  round(weighted_mom, 2),
                "Vol_3M%":      round(vol_3m, 2),
                "MaxDD_63d%":   round(max_dd_63, 2),
                "MomScore":     round(mom_score, 4),
                "Sharpe":       sharpe,
                "UPI":          upi,
                "52W_High":     round(high_52w, 2),
                "Retracement%": retracement_pct,
                "ADR%":         round(adr_val, 2) if not pd.isna(adr_val) else 0.0,
                "ADR>3%":       "✅ Yes" if adr_flag else "–",
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame(), regime_info

    df = pd.DataFrame(results)
    N  = len(df)

    # Composite Score — percentile blend of 4 alpha signals
    df["MomScore_pct"] = df["MomScore"].rank(pct=True, method="average") * 100
    df["UPI_pct"]      = df["UPI"].rank(pct=True,      method="average") * 100
    df["Sharpe_pct"]   = df["Sharpe"].rank(pct=True,   method="average") * 100
    df["RS_pct"]       = df["RS_6M"].rank(pct=True,    method="average") * 100
    df["CompositeScore"] = (
        0.35 * df["MomScore_pct"] + 0.25 * df["UPI_pct"] +
        0.20 * df["Sharpe_pct"]   + 0.20 * df["RS_pct"]
    ).round(2)

    df = df.sort_values("CompositeScore", ascending=False).reset_index(drop=True)
    df["Position"]  = df.index + 1
    df["FinalRank"] = (100 * (N - df["Position"] + 1) / N).round(1)

    def _zone(r):
        if r >= 70: return "🟢 Green"
        if r >= 40: return "🔵 Blue"
        return "🟠 Orange"
    df["Zone"] = df["FinalRank"].apply(_zone)
    return df, regime_info


# ─── PORTFOLIO CONSTRUCTION ───────────────────────────────────────────────────
def build_portfolio(results_df, prices_df, top_n,
                    use_vol_sizing=True, max_pos_pct=10.0,
                    use_corr_dedup=True, max_corr=0.80):
    if results_df.empty:
        return pd.DataFrame()

    candidates = results_df["Ticker"].tolist()

    if use_corr_dedup and len(candidates) > top_n:
        avail = [t for t in candidates if t in prices_df.columns]
        if len(avail) >= 2:
            ret_mat = prices_df[avail].pct_change().tail(63).dropna()
            if not ret_mat.empty and ret_mat.shape[1] >= 2:
                corr_mat = ret_mat.corr()
                selected = []
                for t in candidates:
                    if len(selected) >= top_n: break
                    if t not in corr_mat.columns: selected.append(t); continue
                    if not selected: selected.append(t); continue
                    in_corr = [s for s in selected if s in corr_mat.columns]
                    max_c = corr_mat.loc[t, in_corr].abs().max() if in_corr else 0.0
                    if max_c < max_corr: selected.append(t)
                if len(selected) < top_n:
                    for t in candidates:
                        if t not in selected: selected.append(t)
                        if len(selected) >= top_n: break
                candidates = selected

    portfolio = (results_df[results_df["Ticker"].isin(candidates[:top_n])]
                 .copy().head(top_n).reset_index(drop=True))

    if use_vol_sizing and "Vol_3M%" in portfolio.columns and len(portfolio) > 0:
        inv_vol = 1.0 / portfolio["Vol_3M%"].clip(lower=0.5)
        weights = inv_vol / inv_vol.sum()
        weights = weights.clip(upper=max_pos_pct / 100.0)
        weights = weights / weights.sum()
        portfolio["Weight%"] = (weights * 100).round(2)
    else:
        portfolio["Weight%"] = round(100.0 / max(len(portfolio), 1), 2)

    return portfolio


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:1rem;color:#f8fafc;
                font-weight:600;margin-bottom:0.25rem;">🇺🇸 MLR · US Markets</div>
    <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#2a3d5a;
                margin-bottom:1.5rem;">Momentum Leadership Ranking v2.0</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Universe</div>', unsafe_allow_html=True)
    indices_universe = st.selectbox("Stock Universe", list(INDEX_UNIVERSES.keys()), index=0, label_visibility="collapsed")
    benchmark = st.selectbox("Benchmark", list(BENCHMARK_TICKERS.keys()), index=0)

    st.markdown('<div class="section-title">Data</div>', unsafe_allow_html=True)
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1)

    st.markdown('<div class="section-title">Sort By</div>', unsafe_allow_html=True)
    sort_by = st.selectbox(
        "Sort by",
        ["Composite Score ★", "Final Rank", "UPI (Ulcer Performance)", "Sharpe Ratio",
         "MomScore", "RS vs Benchmark (6M)", "Retracement% (Closest to ATH)",
         "ADR% (Highest Volatility)"],
        label_visibility="collapsed",
    )
    sort_map = {
        "Composite Score ★":             "CompositeScore",
        "Final Rank":                     "FinalRank",
        "UPI (Ulcer Performance)":        "UPI",
        "Sharpe Ratio":                   "Sharpe",
        "MomScore":                       "MomScore",
        "RS vs Benchmark (6M)":           "RS_6M",
        "Retracement% (Closest to ATH)":  "Retracement%",
        "ADR% (Highest Volatility)":      "ADR%",
    }

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">RSI Filters</div>', unsafe_allow_html=True)
    rsi252_min = st.slider("RSI(252) Minimum", 0, 100, 50, 1, help="Wilder RSI over 252 days")
    rsi88_min  = st.slider("RSI(88) Minimum",  0, 100, 50, 1, help="Wilder RSI over 88 days")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Regime — Benchmark Supertrend</div>', unsafe_allow_html=True)
    use_supertrend = st.checkbox("Enable Supertrend Regime Filter", value=True)
    st_period, st_multiplier = 1, 2.5
    if use_supertrend:
        st_period     = st.slider("ST ATR Period",     1, 30,  1, 1)
        st_multiplier = st.slider("ST ATR Multiplier", 0.5, 6.0, 2.5, 0.5)
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#2a3d5a">'
            f'Applied weekly to: <span style="color:#94a3b8">{benchmark.split("(")[0].strip()}</span></div>',
            unsafe_allow_html=True)

    # VIX gate (CBOE)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">CBOE VIX Gate <span class="tag-new">NEW</span></div>', unsafe_allow_html=True)
    use_vix_gate = st.checkbox("Enable VIX Gate", value=True, help="Displays VIX level; warns when elevated.")
    vix_warn_level, vix_block_level = 20.0, 30.0
    if use_vix_gate:
        vix_warn_level  = st.slider("VIX Warning Level", 10, 40, 20, 1)
        vix_block_level = st.slider("VIX Block Level",   15, 60, 30, 1)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Stock Filters <span class="tag-new">IMPROVED</span></div>', unsafe_allow_html=True)
    require_positive_rs = st.checkbox("Require Positive RS vs Benchmark (6M)", value=True,
        help="Stock must have outperformed the index over 6M — removes pure-beta movers.")
    require_acceleration = st.checkbox("Require Momentum Acceleration", value=False,
        help="3-month monthly rate > 6-month monthly rate.")
    require_mom_consistency = st.checkbox("Require Momentum Consistency", value=False,
        help="All three return windows (3M/6M/9M) positive.")
    max_retracement = st.slider("Max Retracement from 52W High (%)", -60, -5, -30, 5,
        help="Excludes stocks more than this far below their 52-week high.")
    max_rolling_dd = st.slider("Max 63-Day Rolling Drawdown (%)", -50, -5, -20, 5,
        help="Excludes stocks with a peak-to-trough loss > |X|% in the last 63 days.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ADR Filter <span class="tag-new">NEW</span></div>', unsafe_allow_html=True)
    adr_window = st.slider("ADR Lookback (days)", 5, 60, 20, 5,
        help="Window for Average Daily Range %.")
    adr_only = st.checkbox("Show only ADR > 3% stocks", value=False,
        help="Restrict results to stocks whose ADR exceeds 3% (high-volatility / active names).")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portfolio Construction <span class="tag-new">NEW</span></div>', unsafe_allow_html=True)
    top_n    = st.slider("Top N Stocks", 5, 50, 20, 5)
    show_all = st.checkbox("Show All Passing Stocks in Tab 5", value=False)
    use_vol_sizing = st.checkbox("Inverse-Volatility Sizing", value=True,
        help="Weight = 1/Vol_3M, normalised. Lower-vol names get more weight.")
    max_pos_pct = 10.0
    if use_vol_sizing:
        max_pos_pct = st.slider("Max Single Position (%)", 5, 25, 10, 1)
    use_corr_dedup = st.checkbox("Correlation Deduplication", value=True,
        help="Greedily drops names with ρ > threshold vs already-selected positions.")
    max_corr = 0.80
    if use_corr_dedup:
        max_corr = st.slider("Max Allowed Correlation", 0.50, 0.95, 0.80, 0.05)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Screen", use_container_width=True, type="primary")


# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
  <h1>🇺🇸 US Momentum Leadership Ranking
    <span style="font-size:0.72rem;color:#38bdf8;background:#0c2a4e;padding:2px 9px;
                 border-radius:4px;border:1px solid #0369a1;vertical-align:middle">v2.0</span>
  </h1>
  <p>Composite ranking · Relative strength · VIX gate · Inverse-vol sizing · Correlation dedup · ADR&gt;3% column</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val, color in [
    (c1, "Universe",       indices_universe,                "#60a5fa"),
    (c2, "Benchmark",      benchmark.split("(")[0].strip(), "#a78bfa"),
    (c3, "Period",         period,                          "#34d399"),
    (c4, "Portfolio Size", f"Top {top_n}",                  "#f472b6"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                    f'<div class="value" style="font-size:1rem;color:{color}">{val}</div></div>',
                    unsafe_allow_html=True)

st.markdown("")

if not run_btn:
    st.markdown("""
    <div class="info-box">
    ℹ Configure parameters in the sidebar and click <strong>▶ Run Screen</strong>.
    v2.0 upgrades: Composite ranking · RS filter · CBOE VIX gate · 63-day DD filter ·
    inverse-vol sizing · correlation dedup · vectorised Supertrend · ADR&gt;3% column.
    Constituents from Wikipedia (S&P 500 / Nasdaq 100) & iShares; prices via Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)

    ia, ib, ic = st.columns(3)
    with ia:
        st.markdown("""
**📈 Better Stock Selection**

🔵 **Composite Score** *(default rank)*
35% MomScore + 25% UPI + 20% Sharpe + 20% RS.
Rewards quality of return, not just magnitude.

🔵 **Relative Strength filter**
Removes stocks that rose only on market beta.

🔵 **Momentum Acceleration** *(optional)*
3M monthly rate > 6M. Avoids fading trends.

🔵 **Consistency gate** *(optional)*
All windows (3M/6M/9M) positive.
""")
    with ib:
        st.markdown("""
**🛡 Drawdown Reduction**

🔴 **CBOE VIX gate**
Warns/blocks on elevated fear. High VIX precedes
sharp index drops — cutting exposure early trims
left-tail risk.

🔴 **63-day rolling drawdown filter**
Excludes names already down >|X|% in the quarter.
Avoids falling knives.

🔴 **52W High retracement cap**
Momentum works near highs.

🔴 **Vectorised Supertrend** *(numpy)*
Correct band-adjustment logic.
""")
    with ic:
        st.markdown("""
**⚖ Smarter Portfolio**

🟢 **Inverse-vol position sizing**
Weight = 1/σ. Same expected return, lower variance.

🟢 **Position cap**
No single name dominates the book.

🟢 **Correlation deduplication**
Greedy ρ-threshold drop → real diversification.

📐 **ADR>3% column** *(US build)*
Avg Daily Range flag for options / active traders.
""")
    st.stop()


# ─── EXECUTION ────────────────────────────────────────────────────────────────
with st.spinner("Fetching constituent list..."):
    constituents = fetch_constituents(indices_universe)

if constituents.empty:
    st.error("Could not fetch constituent list. Check network access.")
    st.stop()

st.success(f"✓ Loaded {len(constituents)} stocks from {indices_universe}")

tickers_yf      = tuple(constituents["YF_Ticker"].tolist())
bench_yf        = BENCHMARK_TICKERS[benchmark]
period_days_val = PERIOD_DAYS[period]

# ── VIX ─────────────────────────────────────────────────────────────────────
us_vix = None
if use_vix_gate:
    with st.spinner("Fetching CBOE VIX..."):
        us_vix = fetch_vix()
    if us_vix is not None:
        vix_c = "#4ade80" if us_vix < vix_warn_level else ("#facc15" if us_vix < vix_block_level else "#f87171")
        vix_s = "Normal ✓"  if us_vix < vix_warn_level else ("Elevated ⚠" if us_vix < vix_block_level else "EXTREME 🚫")
        st.markdown(
            f'<div class="vix-card">🌡 CBOE VIX: '
            f'<strong style="color:{vix_c}">{us_vix:.2f} — {vix_s}</strong>'
            f'&nbsp;|&nbsp; Warn:{vix_warn_level} &nbsp; Block:{vix_block_level}</div>',
            unsafe_allow_html=True)
        if us_vix >= vix_block_level:
            st.markdown(f'<div class="warn-box">⚠️ VIX ({us_vix:.1f}) ≥ block level ({vix_block_level}). '
                        f'High-fear mode. New longs carry extreme risk — consider cash or 50%+ size cut.</div>',
                        unsafe_allow_html=True)
        elif us_vix >= vix_warn_level:
            st.markdown(f'<div class="warn-box">⚡ VIX ({us_vix:.1f}) ≥ warning level ({vix_warn_level}). '
                        f'Volatility elevated — reduce sizes and tighten stops.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">ℹ VIX unavailable — gate shown as N/A.</div>', unsafe_allow_html=True)

# ── Stock OHLCV download ──────────────────────────────────────────────────────
with st.spinner(f"Downloading {len(tickers_yf)} stock prices ({period})..."):
    ohlcv     = fetch_ohlcv(tickers_yf, period_days_val)
    prices_df = ohlcv.get("Close", pd.DataFrame())
    high_df   = ohlcv.get("High",  pd.DataFrame())
    low_df    = ohlcv.get("Low",   pd.DataFrame())

if prices_df.empty:
    st.error("Price download failed.")
    st.stop()

# ── ADR series ────────────────────────────────────────────────────────────────
adr_series = pd.Series(dtype=float)
if not high_df.empty and not low_df.empty:
    try:
        adr_series = compute_adr(high_df, low_df, window=adr_window)
    except Exception:
        pass

# ── Benchmark download ────────────────────────────────────────────────────────
with st.spinner("Fetching benchmark data..."):
    import datetime
    end_d   = datetime.date.today()
    start_d = end_d - datetime.timedelta(days=period_days_val + 100)

    def _dl(ticker, start, end, interval="1d"):
        try:
            df = yf.download(ticker, start=str(start), end=str(end),
                             interval=interval, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            return df
        except Exception:
            return pd.DataFrame()

    bench_raw, bench_weekly, bench_used = pd.DataFrame(), pd.DataFrame(), bench_yf
    for candidate in BENCHMARK_FALLBACKS.get(bench_yf, [bench_yf]):
        raw = _dl(candidate, start_d, end_d, "1d")
        if not raw.empty and "Close" in raw.columns and len(raw) >= 50:
            bench_raw    = raw
            bench_weekly = _dl(candidate, start_d, end_d, "1wk")
            bench_used   = candidate
            break

    if bench_raw.empty or "Close" not in bench_raw.columns:
        st.warning("⚠️ Benchmark data unavailable — regime defaulted to NEUTRAL.")
        bench_close_series = pd.Series(dtype=float)
        bench_weekly       = pd.DataFrame()
    else:
        bench_close_series = bench_raw["Close"].squeeze().dropna()
        if bench_used != bench_yf:
            st.info(f"ℹ️ Benchmark fallback active: using **{bench_used}** as proxy for {benchmark}.")

# ── Benchmark 6M/3M return (lagged) ───────────────────────────────────────────
bench_6m_ret = bench_3m_ret = 0.0
if len(bench_close_series) >= 147:
    b_lag = bench_close_series.iloc[:-21]
    if len(b_lag) >= 126:
        bench_6m_ret = float((b_lag.iloc[-1] / b_lag.iloc[-127] - 1) * 100)
    if len(b_lag) >= 63:
        bench_3m_ret = float((b_lag.iloc[-1] / b_lag.iloc[-64]  - 1) * 100)

st.success(f"✓ {prices_df.shape[1]} stocks × {len(prices_df)} days loaded"
           f" | Benchmark 6M: {bench_6m_ret:+.1f}%  3M: {bench_3m_ret:+.1f}%")

# ── Run screen ────────────────────────────────────────────────────────────────
with st.spinner("Computing composite scores and rankings..."):
    results_df, regime_info = run_momentum_screen(
        prices=prices_df, bench_prices=bench_close_series, bench_weekly=bench_weekly,
        rsi252_min=rsi252_min, rsi88_min=rsi88_min,
        st_period=st_period, st_multiplier=st_multiplier,
        use_supertrend=use_supertrend, period_days=period_days_val,
        bench_6m_ret=bench_6m_ret, bench_3m_ret=bench_3m_ret,
        require_positive_rs=require_positive_rs,
        require_acceleration=require_acceleration,
        require_mom_consistency=require_mom_consistency,
        max_retracement=float(max_retracement), max_rolling_dd=float(max_rolling_dd),
        adr_series=adr_series,
    )

# ── ADR-only filter (re-rank after) ──────────────────────────────────────────
if adr_only and not results_df.empty:
    results_df = results_df[results_df["ADR>3%"] == "✅ Yes"].reset_index(drop=True)
    n = len(results_df)
    if n > 0:
        results_df["Position"]  = results_df.index + 1
        results_df["FinalRank"] = (100 * (n - results_df["Position"] + 1) / n).round(1)
        results_df["Zone"] = results_df["FinalRank"].apply(
            lambda r: "🟢 Green" if r >= 70 else ("🔵 Blue" if r >= 40 else "🟠 Orange"))
    st.info(f"ℹ️ ADR > 3% filter active: {n} stocks retained.")

# ── Regime display ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Market Regime</div>', unsafe_allow_html=True)
regime = regime_info["regime"]
badge  = ('<span class="regime-on">✅ RISK ON — New Positions Allowed</span>'  if regime == "RISK ON"  else
          '<span class="regime-neut">⚠️ NEUTRAL — Reduce Exposure</span>'      if regime == "NEUTRAL" else
          '<span class="regime-off">🚫 RISK OFF — No New Positions</span>')

rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
with rc1:
    note = ' <span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#64748b">(no bench data)</span>' \
           if not regime_info["bench_data_ok"] else ""
    st.markdown(badge + note, unsafe_allow_html=True)
with rc2:
    c = "#4ade80" if regime_info["above_200_ema"] else "#f87171"; l = "ABOVE" if regime_info["above_200_ema"] else "BELOW"
    if not regime_info["bench_data_ok"]: c="#475569"; l="N/A"
    st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">200 EMA: <span style="color:{c}">{l}</span></div>', unsafe_allow_html=True)
with rc3:
    if not regime_info.get("use_supertrend"):
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">Supertrend: <span style="color:#475569">OFF</span></div>', unsafe_allow_html=True)
    else:
        c = "#4ade80" if regime_info["above_supertrend"] else "#f87171"; l = "ABOVE" if regime_info["above_supertrend"] else "BELOW"
        if not regime_info["bench_data_ok"] or not regime_info.get("supertrend_ok"): c="#475569"; l="N/A"
        p, m = regime_info.get("st_period", 1), regime_info.get("st_multiplier", 2.5)
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">ST({p},{m}): <span style="color:{c}">{l}</span></div>', unsafe_allow_html=True)
with rc4:
    if regime_info["bench_close"] is not None:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#94a3b8">Bench: <span style="color:#f8fafc">{regime_info["bench_close"]:,.2f}</span></div>', unsafe_allow_html=True)

st.markdown("")

# ── Negative regime halt ──────────────────────────────────────────────────────
if regime_info.get("negative_regime"):
    bc, bst = regime_info.get("bench_close"), regime_info.get("bench_supertrend")
    p, m = regime_info.get("st_period", 1), regime_info.get("st_multiplier", 2.5)
    gap = ((bc - bst) / bst * 100) if (bc and bst) else 0.0
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#2d0a0a 0%,#3d1010 100%);
                border:1px solid #991b1b;border-radius:10px;padding:1.75rem 2rem;margin-bottom:1rem;">
      <div style="font-family:'DM Mono',monospace;font-size:1.25rem;color:#fca5a5;
                  font-weight:600;margin-bottom:0.5rem;">🚫 NEGATIVE REGIME — Scan Halted</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#f87171;line-height:1.7;">
        <span style="color:#fecaca">{benchmark}</span> is <strong>BELOW</strong> its
        weekly Supertrend({p}, {m}). No new long positions while the index is in a downtrend.
      </div>
      <div style="display:flex;gap:2.5rem;margin-top:1rem;font-family:'DM Mono',monospace;font-size:0.82rem;">
        <div><div style="color:#7f1d1d;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Benchmark Close</div>
             <div style="color:#fecaca;font-size:1.2rem;font-weight:600">{bc:,.2f}</div></div>
        <div><div style="color:#7f1d1d;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Weekly Supertrend</div>
             <div style="color:#fca5a5;font-size:1.2rem;font-weight:600">{bst:,.2f}</div></div>
        <div><div style="color:#7f1d1d;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Distance</div>
             <div style="color:#f87171;font-size:1.2rem;font-weight:600">{gap:+.2f}%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="warn-box">📉 Disable the Supertrend filter in the sidebar to scan regardless of regime.</div>', unsafe_allow_html=True)
    st.stop()

if regime_info["regime"] == "RISK OFF":
    st.markdown('<div class="warn-box">⚠️ Market is in Risk OFF regime — no new positions recommended.</div>', unsafe_allow_html=True)

if results_df.empty:
    st.warning("No stocks passed all filters. Try relaxing RSI, retracement, drawdown, or ADR thresholds.")
    st.stop()

# ── Build portfolio ───────────────────────────────────────────────────────────
with st.spinner("Building portfolio (correlation dedup + vol-sizing)..."):
    portfolio_df = build_portfolio(results_df, prices_df, top_n,
                                   use_vol_sizing=use_vol_sizing, max_pos_pct=max_pos_pct,
                                   use_corr_dedup=use_corr_dedup, max_corr=max_corr)

# ── Annotate with company name & TV link ─────────────────────────────────────
ticker_to_name   = dict(zip(constituents["YF_Ticker"], constituents["Company"]))
ticker_to_sector = dict(zip(constituents["YF_Ticker"], constituents.get("Sector", pd.Series(dtype=str))))
for df_ref in [results_df, portfolio_df]:
    if df_ref is None or df_ref.empty:
        continue
    df_ref["Company"]    = df_ref["Ticker"].map(ticker_to_name).fillna(df_ref["Ticker"])
    df_ref["Sector"]     = df_ref["Ticker"].map(ticker_to_sector).fillna("")
    df_ref["Symbol"]     = df_ref["Ticker"]
    df_ref["SymbolLink"] = df_ref["Symbol"].apply(lambda s: f"https://www.tradingview.com/chart/?symbol={s}")

# ── Summary metrics ───────────────────────────────────────────────────────────
green  = results_df[results_df["Zone"] == "🟢 Green"]
blue   = results_df[results_df["Zone"] == "🔵 Blue"]
orange = results_df[results_df["Zone"] == "🟠 Orange"]
adr_hi = results_df[results_df["ADR>3%"] == "✅ Yes"]

port_vol_est = 0.0
if not portfolio_df.empty and "Vol_3M%" in portfolio_df.columns and "Weight%" in portfolio_df.columns:
    port_vol_est = float((portfolio_df["Vol_3M%"] * portfolio_df["Weight%"] / 100).sum())

st.markdown('<div class="section-title">Screening Summary</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
for col, lbl, val, clr in [
    (m1, "Passed Filters",   len(results_df),                              "#f8fafc"),
    (m2, "🟢 Green",         len(green),                                   "#4ade80"),
    (m3, "🔵 Blue",          len(blue),                                    "#60a5fa"),
    (m4, "🟠 Orange",        len(orange),                                  "#fb923c"),
    (m5, "Avg Composite",    f"{results_df['CompositeScore'].mean():.1f}", "#a78bfa"),
    (m6, "Port. Vol (est.)", f"{port_vol_est:.1f}%",                       "#34d399"),
    (m7, "ADR &gt; 3%",      len(adr_hi),                                  "#38bdf8"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="label">{lbl}</div>'
                    f'<div class="value" style="font-size:1.4rem;color:{clr}">{val}</div></div>',
                    unsafe_allow_html=True)

st.markdown("")

# ── Sorting & display columns ─────────────────────────────────────────────────
sort_col       = sort_map[sort_by]
sort_ascending = (sort_col == "Retracement%")
results_sorted = results_df.sort_values(sort_col, ascending=sort_ascending).reset_index(drop=True)

DISPLAY_COLS = [
    "SymbolLink", "Company", "Zone", "CompositeScore", "FinalRank",
    "MomScore", "UPI", "Sharpe",
    "ADR%", "ADR>3%",
    "RS_6M", "RS_3M", "Accel",
    "Retracement%", "MaxDD_63d%",
    "RSI252", "RSI88",
    "Ret_9M%", "Ret_6M%", "Ret_3M%",
    "Vol_3M%", "Close",
]
RENAME_MAP = {
    "SymbolLink": "Symbol", "CompositeScore": "CompScore", "FinalRank": "Rank",
    "RS_6M": "RS_6M%", "RS_3M": "RS_3M%", "MaxDD_63d%": "MaxDD_63d",
}
display_df = results_sorted[[c for c in DISPLAY_COLS if c in results_sorted.columns]].copy().rename(columns=RENAME_MAP)


# ── Render helpers ────────────────────────────────────────────────────────────
def _c(val, lo, hi, lo_c="#f87171", mid_c="#facc15", hi_c="#4ade80", bold=False):
    try:
        v = float(val)
        c = hi_c if v >= hi else (mid_c if v >= lo else lo_c)
        return f"color:{c};font-weight:{'600' if bold else '400'}"
    except:
        return ""

def style_zone(val):
    c = {"🟢 Green": "#4ade80", "🔵 Blue": "#60a5fa", "🟠 Orange": "#fb923c"}.get(str(val), "")
    return f"color:{c};font-weight:600" if c else ""

def style_adr_flag(val):
    return "color:#4ade80;font-weight:600" if "✅" in str(val) else "color:#334155"

def color_adr(val):
    try:
        v = float(str(val).replace("%", ""))
        if v > 5:   return "color:#f87171;font-weight:600"
        if v > 3:   return "color:#facc15;font-weight:600"
        return "color:#94a3b8"
    except: return ""

def render_table(df: pd.DataFrame, height: int = 520):
    fmt = {
        "Rank": "{:.1f}", "CompScore": "{:.1f}", "CompositeScore": "{:.1f}",
        "MomScore": "{:.3f}", "UPI": "{:.3f}", "Sharpe": "{:.1f}",
        "ADR%": "{:.2f}%",
        "RS_6M%": "{:+.2f}%", "RS_3M%": "{:+.2f}%",
        "Retracement%": "{:+.2f}%", "MaxDD_63d": "{:+.2f}%",
        "RSI252": "{:.1f}", "RSI88": "{:.1f}",
        "Ret_9M%": "{:+.2f}%", "Ret_6M%": "{:+.2f}%", "Ret_3M%": "{:+.2f}%",
        "Vol_3M%": "{:.2f}%", "Close": "${:.2f}", "Weight%": "{:.2f}%",
    }
    fmt = {k: v for k, v in fmt.items() if k in df.columns}

    styled = df.style
    if "Zone"   in df.columns: styled = styled.map(style_zone,     subset=["Zone"])
    if "ADR>3%" in df.columns: styled = styled.map(style_adr_flag, subset=["ADR>3%"])
    if "ADR%"   in df.columns: styled = styled.map(color_adr,      subset=["ADR%"])
    if "RS_6M%" in df.columns: styled = styled.map(lambda v: _c(v, 0, 5), subset=["RS_6M%"])
    if "RS_3M%" in df.columns: styled = styled.map(lambda v: _c(v, 0, 5), subset=["RS_3M%"])
    if "Retracement%" in df.columns: styled = styled.map(lambda v: _c(v, -15, -5, bold=True), subset=["Retracement%"])
    if "MaxDD_63d" in df.columns:    styled = styled.map(lambda v: _c(v, -12, -5), subset=["MaxDD_63d"])
    if "UPI"    in df.columns: styled = styled.map(lambda v: _c(v, 0.5, 1.5, bold=True), subset=["UPI"])
    if "Sharpe" in df.columns: styled = styled.map(lambda v: _c(v, 0, 100), subset=["Sharpe"])
    for rc in ("RSI252", "RSI88"):
        if rc in df.columns: styled = styled.map(lambda v: _c(v, 50, 70), subset=[rc])

    grad = [c for c in ["Rank", "CompScore"] if c in df.columns]
    if grad:
        styled = styled.background_gradient(subset=grad, cmap="RdYlGn", vmin=0, vmax=100)

    styled = styled.format(fmt, na_rep="—").set_properties(**{"font-family": "DM Mono, monospace", "font-size": "12px"})

    col_cfg = {}
    if "Symbol" in df.columns:
        col_cfg["Symbol"] = st.column_config.LinkColumn(
            "Symbol", help="Open on TradingView", display_text=r"symbol=(.+)$", width="small")

    st.dataframe(styled, column_config=col_cfg, use_container_width=True, height=height)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"🏆 Portfolio ({len(portfolio_df)})",
    f"🟢 Green ({len(green)})",
    f"🔵 Blue ({len(blue)})",
    f"🟠 Orange ({len(orange)})",
    f"📋 All Passing ({len(results_df)})",
])

with tab1:
    if portfolio_df.empty:
        st.info("Portfolio is empty — try relaxing filters.")
    else:
        sz_note = f"Inverse-vol weights (cap {max_pos_pct}%)" if use_vol_sizing else "Equal weights"
        dd_note = f" · Corr dedup ρ<{max_corr}" if use_corr_dedup else ""
        st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;margin-bottom:0.5rem'>"
                    f"{sz_note}{dd_note} · sorted by <strong style='color:#94a3b8'>{sort_by}</strong></div>",
                    unsafe_allow_html=True)
        p_cols = [c for c in ["SymbolLink", "Company", "Weight%", "Zone", "CompositeScore",
                              "MomScore", "UPI", "Sharpe", "ADR%", "ADR>3%",
                              "RS_6M", "Accel", "Retracement%", "MaxDD_63d%",
                              "RSI252", "RSI88", "Ret_9M%", "Ret_6M%", "Ret_3M%",
                              "Vol_3M%", "Close"] if c in portfolio_df.columns]
        p_df = portfolio_df[p_cols].copy().rename(columns={
            "SymbolLink": "Symbol", "CompositeScore": "CompScore",
            "RS_6M": "RS_6M%", "MaxDD_63d%": "MaxDD_63d"})
        _ps = {"CompScore": "CompScore", "Sharpe": "Sharpe", "UPI": "UPI",
               "MomScore": "MomScore", "Weight%": "Weight%",
               "Retracement%": "Retracement%", "RS_6M%": "RS_6M%", "ADR%": "ADR%"}
        _pc = _ps.get(sort_col, "CompScore") if sort_col != "FinalRank" else "CompScore"
        if _pc in p_df.columns:
            p_df = p_df.sort_values(_pc, ascending=(sort_col == "Retracement%")).reset_index(drop=True)
        render_table(p_df)

with tab2:
    g_df = display_df[display_df["Zone"] == "🟢 Green"]
    render_table(g_df) if not g_df.empty else st.info("No stocks in Green zone with current filters.")

with tab3:
    b_df = display_df[display_df["Zone"] == "🔵 Blue"]
    render_table(b_df) if not b_df.empty else st.info("No stocks in Blue zone.")

with tab4:
    o_df = display_df[display_df["Zone"] == "🟠 Orange"]
    render_table(o_df) if not o_df.empty else st.info("No stocks in Orange zone.")

with tab5:
    lim = len(display_df) if show_all else min(top_n * 3, len(display_df))
    st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b;margin-bottom:0.5rem'>"
                f"Showing {lim} of {len(display_df)} passing stocks · sorted by {sort_by}</div>", unsafe_allow_html=True)
    render_table(display_df.head(lim), height=620)


# ── Analytics ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)
ch1, ch2, ch3 = st.columns(3)
with ch1:
    st.markdown("**Zone Distribution**")
    zc = pd.DataFrame({"Zone": ["🟢 Green", "🔵 Blue", "🟠 Orange"], "Count": [len(green), len(blue), len(orange)]})
    st.bar_chart(zc.set_index("Zone")["Count"])
with ch2:
    st.markdown("**Top 20: Composite Score**")
    st.bar_chart(results_df.head(20)[["Symbol", "CompositeScore"]].set_index("Symbol"))
with ch3:
    if not portfolio_df.empty and "Weight%" in portfolio_df.columns:
        st.markdown("**Portfolio: Weight% (inv-vol)**")
        st.bar_chart(portfolio_df[["Symbol", "Weight%"]].set_index("Symbol"))
    else:
        st.markdown("**Top 20: MomScore**")
        st.bar_chart(results_df.head(20)[["Symbol", "MomScore"]].set_index("Symbol"))

# ADR distribution
st.markdown('<div class="section-title">ADR% Distribution — Top Passing Stocks</div>', unsafe_allow_html=True)
if "ADR%" in results_df.columns:
    st.bar_chart(results_df.head(min(40, len(results_df)))[["Symbol", "ADR%"]].set_index("Symbol"))

# Composite breakdown
st.markdown('<div class="section-title">Composite Score Breakdown — Top Candidates</div>', unsafe_allow_html=True)
cb_cols = [c for c in ["Symbol", "MomScore_pct", "UPI_pct", "Sharpe_pct", "RS_pct", "CompositeScore", "Zone"]
           if c in results_df.columns]
cb = results_df.head(top_n)[cb_cols].copy().rename(columns={
    "MomScore_pct": "Mom%", "UPI_pct": "UPI%", "Sharpe_pct": "Sharpe%",
    "RS_pct": "RS%", "CompositeScore": "Composite"})
fmt_cb  = {c: "{:.1f}" for c in ["Mom%", "UPI%", "Sharpe%", "RS%", "Composite"] if c in cb.columns}
grad_cb = [c for c in ["Mom%", "UPI%", "Sharpe%", "RS%", "Composite"] if c in cb.columns]
st.dataframe(
    cb.style.background_gradient(subset=grad_cb, cmap="RdYlGn", vmin=0, vmax=100)
      .map(style_zone, subset=["Zone"] if "Zone" in cb.columns else [])
      .format(fmt_cb, na_rep="—")
      .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "12px"}),
    use_container_width=True, height=300)

# Sector breakdown
if "Sector" in results_df.columns and results_df["Sector"].astype(str).str.len().gt(0).any():
    st.markdown('<div class="section-title">Sector Breakdown — Top Portfolio</div>', unsafe_allow_html=True)
    sec = results_df.head(top_n)["Sector"].replace("", np.nan).dropna().value_counts().reset_index()
    sec.columns = ["Sector", "Count"]
    if not sec.empty:
        st.bar_chart(sec.set_index("Sector")["Count"])

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("⬇ Full Results (CSV)", results_df.to_csv(index=False),
                       file_name=f"us_mlr_full_{indices_universe.replace(' ','_')}_{period}.csv",
                       mime="text/csv", use_container_width=True)
with dl2:
    st.download_button(f"⬇ Portfolio Top {top_n} (CSV)", portfolio_df.to_csv(index=False),
                       file_name=f"us_mlr_portfolio_{period}.csv", mime="text/csv", use_container_width=True)
with dl3:
    st.download_button("⬇ Green Zone Only (CSV)", green.to_csv(index=False),
                       file_name=f"us_mlr_green_{period}.csv", mime="text/csv", use_container_width=True)

# ── Rules footer ──────────────────────────────────────────────────────────────
with st.expander("📋 v2.0 Entry / Exit / Sizing Rules Reference"):
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
**Entry Rules** *(all must be true)*
1. Benchmark > 200 EMA
2. Benchmark weekly > Supertrend *(configurable)*
3. CBOE VIX within acceptable range
4. Stock Close > 200 EMA
5. 50 EMA > 200 EMA
6. RSI(252) ≥ threshold
7. RSI(88) ≥ threshold
8. RS vs Benchmark (6M) > 0 *(configurable)*
9. Momentum acceleration *(configurable)*
10. All periods positive *(configurable)*
11. Retracement from 52W High ≥ threshold
12. 63-day rolling drawdown ≥ threshold
13. **Composite Rank ≥ 70 (Green Zone)**
""")
    with r2:
        st.markdown("""
**Exit Rules** *(any triggers exit)*
- Close < 200 EMA → full exit
- Composite Rank < 40 (Orange) → full exit
- RS vs benchmark goes negative (6M) → review
- Benchmark weekly < Supertrend
  → reduce 50% or move to cash

**Rebalance**
- Every 21 trading days (~monthly)
- Re-rank by Composite Score
- Replace stocks failing any filter
- Recompute vol-adjusted weights

**Risk-Free Rate**: 5.25% (USD)
""")
    with r3:
        st.markdown("""
**Composite Score Weights**

| Signal | Weight | Rationale |
|--------|--------|-----------|
| MomScore %ile | 35% | Return/vol efficiency |
| UPI %ile | 25% | Drawdown quality |
| Sharpe %ile | 20% | Absolute risk-adj return |
| RS(6M) %ile | 20% | True alpha vs beta |

**ADR Column** *(US build)*
- ADR% = mean((H−L)/mid×100) over N days
- ✅ Yes → ADR > 3% (active / high-range)
- Useful for options & momentum traders

**Portfolio Sizing**
- Weight = 1/σ (inverse-vol), capped
- Correlation dedup: ρ threshold
""")
