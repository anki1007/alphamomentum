from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from typing import List, Dict

st.set_page_config(page_title="Alpha Momentum Screener", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');
:root {
  --app-font: 'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --bg: #0b0e13; --bg-2: #10141b; --border: #1f2732; --border-soft: #1a2230;
  --text: #e6eaee; --text-dim: #b3bdc7; --accent: #7a5cff; --accent-2: #2bb0ff;
}
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--app-font) !important; }
.block-container { padding-top: 3.75rem; }
.hero-title {
  font-weight: 800; font-size: clamp(26px, 4.5vw, 40px); line-height: 1.05; margin: 18px 0 4px 0;
  background: linear-gradient(90deg, var(--accent-2), var(--accent) 60%); -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: .2px;
}
.hero-sub { color: var(--text-dim); font-size: 0.88rem; margin-bottom: 18px; }
section[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-weight: 700; color: var(--text-dim) !important; }
section[data-testid="stSidebar"] .stSelectbox > div { background: #131922 !important; border: 1px solid var(--border) !important; border-radius: 8px; }
.stButton button { background: linear-gradient(180deg, #1b2432, #131922); color: var(--text); border: 1px solid var(--border); border-radius: 10px; font-weight: 700; }
.stButton button:hover { filter: brightness(1.12); border-color: var(--accent); }
.pro-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 14px; padding: 6px 10px 10px 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
.info-pill { display:inline-block; background:#131922; border:1px solid var(--border); border-radius:20px; padding:3px 12px; font-size:0.78rem; color:var(--text-dim); margin:2px 4px 10px 0; }
.warn-box { background: rgba(255,165,0,0.1); border:1px solid rgba(255,165,0,0.35); border-radius:10px; padding:8px 14px; font-size:0.83rem; color:#ffc966; margin-bottom:10px; }
a { text-decoration: none; color: #9ecbff; } a:hover { text-decoration: underline; }
table { border-collapse: collapse; font-size: 0.86rem; width: 100%; color: var(--text); }
thead th { position: sticky; top: 0; z-index: 2; background: #121823; color: var(--text-dim); border-bottom: 1px solid var(--border); padding: 6px 8px; white-space: nowrap; }
tbody td { padding: 6px 8px; border-top: 1px solid var(--border-soft); white-space: nowrap; }
h2, h3, .stMarkdown h2, .stMarkdown h3 { color: var(--text); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Alpha Momentum Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Relative Strength &amp; Momentum across all timeframes — real-time &amp; historical</div>', unsafe_allow_html=True)

# ─────────────────────────── CONFIG ───────────────────────────────────────────
BENCHMARKS: Dict[str, str] = {
    "NIFTY 50":             "^NSEI",
    "Nifty 200":            "^CNX200",
    "Nifty 500":            "^CRSLDX",
    "Nifty Midcap 150":     "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250":   "^NIFTYSMLCAP250.NS",
}
GITHUB_BASE = "https://raw.githubusercontent.com/anki1007/alphamomentum/main/"
CSV_FILES: Dict[str, str] = {
    "Nifty 50":            GITHUB_BASE + "nifty50.csv",
    "Nifty 100":           GITHUB_BASE + "nifty100.csv",
    "Nifty 200":           GITHUB_BASE + "nifty200.csv",
    "Nifty 500":           GITHUB_BASE + "nifty500.csv",
    "Nifty Midcap 150":    GITHUB_BASE + "niftymidcap150.csv",
    "Nifty Mid Small 400": GITHUB_BASE + "niftymidsmallcap400.csv",
    "Nifty Smallcap 250":  GITHUB_BASE + "niftysmallcap250.csv",
    "Nifty Microcap 250":  GITHUB_BASE + "niftymicrocap250.csv",
    "Nifty Total Market":  GITHUB_BASE + "niftytotalmarket.csv",
}

# ── Timeframes ─────────────────────────────────────────────────────────────────
# label → (yf_interval, resample_rule_or_None, bars_per_trading_day, max_days_yf)
# NSE session: 9:15–15:30 = 375 min
TIMEFRAME_CONFIG: Dict[str, tuple] = {
    "30 Min":  ("30m",  None,    12, 60),    # yf max 60 calendar days
    "60 Min":  ("60m",  None,    6,  730),   # yf max 730 days
    "240 Min": ("60m",  "240T",  2,  730),   # fetch 1h, resample → 4h
    "1 Day":   ("1d",   None,    1,  99999),
    "1 Week":  ("1wk",  None,    0.2, 99999),
}

# ── Period in days ─────────────────────────────────────────────────────────────
PERIOD_DAYS_MAP: Dict[str, int] = {
    "30 Days":   30,
    "63 Days":   63,
    "90 Days":   90,
    "126 Days":  126,
    "252 Days":  252,
    "2 Year":    504,
    "3 Year":    756,
    "4 Year":    1008,
    "5 Year":    1260,
}

JDK_WINDOW = 21   # smoothing window for JdK RS-Ratio / RS-Momentum


# ─────────────────────────── HELPERS ──────────────────────────────────────────
def tv_symbol_from_yf(symbol: str) -> str:
    s = symbol.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_chart_url(symbol: str) -> str:
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol_from_yf(symbol)}"

def _pick_close(df: pd.DataFrame | pd.Series, symbol: str) -> pd.Series:
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
    for col in ("Close", "Adj Close"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    return pd.Series(dtype=float)

def resample_ohlcv(series: pd.Series, rule: str) -> pd.Series:
    """Resample a close price series to a coarser bar (e.g. 4h)."""
    df = series.to_frame("close")
    resampled = df["close"].resample(rule).last().dropna()
    return resampled

def jdk_components(price: pd.Series, bench: pd.Series, win: int = JDK_WINDOW):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"] / df["b"])
    m  = rs.rolling(win).mean()
    s  = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs - m) / s).dropna()
    rroc = rs_ratio.pct_change().mul(100)
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom   = (101 + (rroc - m2) / s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def perf_quadrant(x: float, y: float) -> str:
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100:  return "Improving"
    if x < 100 and y < 100:   return "Lagging"
    return "Weakening"

def analyze_momentum(adj: pd.Series, bars_1y: int, bars_6m: int, bars_3m: int, bars_1m: int) -> dict | None:
    """
    Filter & return short-period returns.
    bars_* are bar-counts equivalent to calendar periods for the chosen timeframe.
    """
    if adj is None or adj.empty or len(adj) < bars_1y:
        return None
    ema100 = adj.ewm(span=max(bars_1y // 2, 10), adjust=False).mean()
    try:
        one_year_return = (adj.iloc[-1] / adj.iloc[-bars_1y] - 1.0) * 100.0
    except Exception:
        return None
    high_52w = adj.iloc[-bars_1y:].max()
    within_20pct_high = adj.iloc[-1] >= high_52w * 0.8
    six_month_bars = adj.iloc[-bars_6m:]
    up_days_pct = (six_month_bars.pct_change() > 0).sum() / len(six_month_bars) * 100.0
    if (adj.iloc[-1] >= ema100.iloc[-1] and one_year_return >= 6.5
            and within_20pct_high and up_days_pct > 45.0):
        try:
            r6 = (adj.iloc[-1] / adj.iloc[-bars_6m] - 1.0) * 100.0
            r3 = (adj.iloc[-1] / adj.iloc[-bars_3m] - 1.0) * 100.0
            r1 = (adj.iloc[-1] / adj.iloc[-bars_1m] - 1.0) * 100.0
        except Exception:
            return None
        return {"Return_6M": r6, "Return_3M": r3, "Return_1M": r1}
    return None

@st.cache_data(show_spinner=False)
def load_universe_from_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    cols = {c.strip().lower(): c for c in df.columns}
    required = ["symbol", "company name", "industry"]
    for n in required:
        if n not in cols:
            raise ValueError(f"CSV must include columns: {required}. Missing: {n}")
    df = df[[cols["symbol"], cols["company name"], cols["industry"]]].copy()
    df.columns = ["Symbol", "Name", "Industry"]
    df = df.dropna(subset=["Symbol"])
    df["Symbol"]   = df["Symbol"].astype(str).str.strip()
    df["Name"]     = df["Name"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    return df

def _period_days_to_dates(period_days: int, extra_buffer: int = 60) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Convert a period expressed in *calendar* days to (start, end) timestamps.
    extra_buffer adds extra days so we always have enough bars even after
    weekends / holidays are dropped.
    """
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    end       = today_ist + pd.Timedelta(days=1)
    # multiply calendar days by ~1.4 to account for weekends + holidays
    cal_days  = int(period_days * 1.42) + extra_buffer
    start     = today_ist - pd.Timedelta(days=cal_days)
    return start, end

@st.cache_data(show_spinner=True, ttl=300)  # cache for 5 min → near-real-time refresh
def fetch_prices(
    tickers: List[str],
    benchmark: str,
    period_days: int,
    yf_interval: str,
    resample_rule: str | None,
    max_days_yf: int,
) -> pd.DataFrame:
    # Cap period to what yfinance allows for this interval
    effective_days = min(period_days, max_days_yf)
    start, end = _period_days_to_dates(effective_days)
    try:
        data = yf.download(
            tickers + [benchmark],
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval=yf_interval,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as e:
        msg = str(e)
        if "Rate limited" in msg or "Too Many Requests" in msg:
            st.warning("Yahoo Finance rate limited the request. Please try again in a moment.")
        else:
            st.error(f"Data download failed: {e}")
        return pd.DataFrame()
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            pass
    return data

def row_bg_for_serial(sno: int) -> str:
    if sno <= 30: return "rgba(46, 204, 113, 0.12)"
    if sno <= 60: return "rgba(255, 204, 0, 0.12)"
    if sno <= 90: return "rgba(52, 152, 219, 0.12)"
    return "rgba(231, 76, 60, 0.12)"

def build_table_dataframe(
    raw: pd.DataFrame,
    benchmark: str,
    universe_df: pd.DataFrame,
    resample_rule: str | None,
    bars_per_day: float,
    period_days: int,
) -> pd.DataFrame:
    bench_raw = _pick_close(raw, benchmark).dropna()
    if bench_raw.empty:
        raise RuntimeError(f"Benchmark {benchmark} series empty.")

    # Resample if needed (e.g. 60m → 240m)
    bench = resample_ohlcv(bench_raw, resample_rule) if resample_rule else bench_raw

    # ── Bar-count equivalents for momentum filters ─────────────────────────
    # We express everything as multiples of bars_per_day
    def d2b(days: int) -> int:
        return max(int(round(days * bars_per_day)), 2)

    # Map calendar periods → bar counts
    # Use the selected period_days as the "1-year" reference for RS lookback
    bars_1y  = d2b(min(period_days, 252))   # cap at 252 days for 1Y reference
    bars_6m  = d2b(min(period_days // 2, 126))
    bars_3m  = d2b(min(period_days // 4, 63))
    bars_1m  = d2b(min(period_days // 12, 21))

    # RS lookback: use the full selected period
    rs_cutoff_bars = d2b(period_days)
    rs_cutoff_ts   = bench.index[-rs_cutoff_bars] if len(bench) > rs_cutoff_bars else bench.index[0]
    bench_rs       = bench.loc[bench.index >= rs_cutoff_ts]

    rows = []
    for _, rec in universe_df.iterrows():
        sym, name, industry = rec.Symbol, rec.Name, rec.Industry
        s_raw = _pick_close(raw, sym).dropna()
        if s_raw.empty:
            continue

        s = resample_ohlcv(s_raw, resample_rule) if resample_rule else s_raw

        if analyze_momentum(s, bars_1y, bars_6m, bars_3m, bars_1m) is None:
            continue

        s_rs = s.loc[s.index >= rs_cutoff_ts]
        rr, mm = jdk_components(s_rs, bench_rs)
        if rr.empty or mm.empty:
            continue

        ix = rr.index.intersection(mm.index)
        rows.append({
            "Name":         name,
            "Industry":     industry,
            "Return_6M":    float((s.iloc[-1] / s.iloc[-bars_6m] - 1) * 100) if len(s) >= bars_6m else np.nan,
            "Rank_6M":      np.nan,
            "Return_3M":    float((s.iloc[-1] / s.iloc[-bars_3m] - 1) * 100) if len(s) >= bars_3m else np.nan,
            "Rank_3M":      np.nan,
            "Return_1M":    float((s.iloc[-1] / s.iloc[-bars_1m] - 1) * 100) if len(s) >= bars_1m else np.nan,
            "Rank_1M":      np.nan,
            "RS-Ratio":     float(rr.loc[ix].iloc[-1]),
            "RS-Momentum":  float(mm.loc[ix].iloc[-1]),
            "Performance":  perf_quadrant(float(rr.loc[ix].iloc[-1]), float(mm.loc[ix].iloc[-1])),
            "Symbol":       sym,
            "Chart":        tradingview_chart_url(sym),
        })

    if not rows:
        raise RuntimeError(
            "No tickers passed the filters. "
            "Try a larger Period or check if the timeframe has enough history."
        )
    df = pd.DataFrame(rows)

    for c in ("Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    df["Rank_6M"]    = df["Return_6M"].rank(ascending=False, method="min").astype("Int64")
    df["Rank_3M"]    = df["Return_3M"].rank(ascending=False, method="min").astype("Int64")
    df["Rank_1M"]    = df["Return_1M"].rank(ascending=False, method="min").astype("Int64")
    df["Final_Rank"] = (
        df["Rank_6M"].fillna(0) + df["Rank_3M"].fillna(0) + df["Rank_1M"].fillna(0)
    ).astype("Int64")

    df = df.sort_values("Final_Rank", kind="mergesort").reset_index(drop=True)
    df.insert(0, "S.No", np.arange(1, len(df) + 1, dtype=int))
    df["Position"] = df["S.No"].astype(int)

    order = [
        "S.No", "Name", "Industry",
        "Return_6M", "Rank_6M",
        "Return_3M", "Rank_3M",
        "Return_1M", "Rank_1M",
        "RS-Ratio", "RS-Momentum", "Performance",
        "Final_Rank", "Position", "Chart", "Symbol",
    ]
    return df[order]

def style_rows(df: pd.DataFrame):
    def _row_style(r: pd.Series):
        bg = row_bg_for_serial(int(r["S.No"]))
        return [f"background-color: {bg}"] * len(df.columns)
    styler = df.style.apply(lambda rr: _row_style(rr), axis=1)
    text_cols  = ["Name", "Industry"]
    center_cols = [c for c in df.columns if c not in text_cols]
    styler = styler.set_properties(subset=text_cols, **{"text-align": "left"})
    styler = styler.set_properties(
        subset=center_cols,
        **{"text-align": "center", "font-variant-numeric": "tabular-nums"},
    )
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    return styler


# ─────────────────────────── SIDEBAR UI ───────────────────────────────────────
st.sidebar.header("Controls")

indices_universe = st.sidebar.selectbox(
    "Indices Universe", list(CSV_FILES.keys()), index=0
)
benchmark_key = st.sidebar.selectbox(
    "Benchmark", list(BENCHMARKS.keys()), index=2
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Timeframe**")
timeframe_label = st.sidebar.selectbox(
    "Bar Size",
    list(TIMEFRAME_CONFIG.keys()),
    index=3,          # default: 1 Day
    help="Intraday bars (30 / 60 / 240 Min) fetch near-real-time data. Daily & Weekly use EOD prices.",
)

st.sidebar.markdown("**Lookback Period**")
period_label = st.sidebar.selectbox(
    "Period",
    list(PERIOD_DAYS_MAP.keys()),
    index=4,          # default: 252 Days
    help="Calendar days of history used for RS & momentum calculation.",
)

do_load = st.sidebar.button("Load / Refresh", use_container_width=True)

# Unpack config for the chosen timeframe
yf_interval, resample_rule, bars_per_day, max_days_yf = TIMEFRAME_CONFIG[timeframe_label]
period_days = PERIOD_DAYS_MAP[period_label]

# Warn when period exceeds yfinance intraday limits
intraday = timeframe_label in ("30 Min", "60 Min", "240 Min")
if intraday and period_days > max_days_yf:
    st.sidebar.markdown(
        f'<div class="warn-box">⚠ {timeframe_label} data is limited to '
        f'<b>{max_days_yf} calendar days</b> by Yahoo Finance. '
        f'Period will be capped automatically.</div>',
        unsafe_allow_html=True,
    )

# Auto-run on first load
if "ran_once" not in st.session_state:
    st.session_state.ran_once = True
    do_load = True


# ─────────────────────────── ACTION ───────────────────────────────────────────
if do_load:
    try:
        uni_url     = CSV_FILES[indices_universe]
        universe_df = load_universe_from_csv(uni_url)
        benchmark   = BENCHMARKS[benchmark_key]
        tickers     = universe_df["Symbol"].tolist()

        # ── Info pills ──────────────────────────────────────────────────────
        data_type = "Near Real-Time (intraday)" if intraday else "EOD Historical"
        effective_period_days = min(period_days, max_days_yf)
        st.markdown(
            f'<span class="info-pill">📊 {timeframe_label}</span>'
            f'<span class="info-pill">📅 {period_label}'
            + (f" → capped {max_days_yf}d" if period_days > max_days_yf else "")
            + f'</span>'
            f'<span class="info-pill">🔄 {data_type}</span>'
            f'<span class="info-pill">🏦 {benchmark_key}</span>',
            unsafe_allow_html=True,
        )

        with st.spinner(f"Fetching {timeframe_label} prices ({data_type})…"):
            raw = fetch_prices(
                tickers,
                benchmark,
                period_days=effective_period_days,
                yf_interval=yf_interval,
                resample_rule=resample_rule,
                max_days_yf=max_days_yf,
            )
        if raw.empty:
            st.stop()

        df = build_table_dataframe(
            raw,
            benchmark,
            universe_df,
            resample_rule=resample_rule,
            bars_per_day=bars_per_day,
            period_days=effective_period_days,
        )

        # ── Display ─────────────────────────────────────────────────────────
        ui_cols = [
            "S.No", "Name", "Industry",
            "Return_6M", "Rank_6M",
            "Return_3M", "Rank_3M",
            "Return_1M", "Rank_1M",
            "RS-Ratio", "RS-Momentum", "Performance",
            "Final_Rank", "Position", "Chart",
        ]
        display_df = df[ui_cols].copy()

        display_df["Name"] = display_df.apply(
            lambda r: f'<a href="{r["Chart"]}" target="_blank" rel="noopener noreferrer">{r["Name"]}</a>',
            axis=1,
        )
        display_df = display_df.drop(columns=["Chart"])

        int_cols    = ["S.No", "Rank_6M", "Rank_3M", "Rank_1M", "Final_Rank", "Position"]
        two_dec_cols = ["Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum"]

        for c in int_cols:
            display_df[c] = display_df[c].map(lambda v: "-" if pd.isna(v) else f"{int(v)}")
        for c in two_dec_cols:
            display_df[c] = display_df[c].map(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}")

        st.subheader("Alpha Momentum Screener Results")
        table_html = style_rows(display_df).to_html()
        st.markdown(f'<div class="pro-card">{table_html}</div>', unsafe_allow_html=True)

        st.caption(
            f"{len(df)} results  •  {indices_universe}  •  "
            f"{timeframe_label} bars  •  {period_label} lookback  •  {benchmark_key}"
        )

        # ── CSV export ───────────────────────────────────────────────────────
        export_df = df.drop(columns=["Symbol"]).copy()
        for c in ("Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum"):
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce").round(2)
        for c in ("S.No", "Rank_6M", "Rank_3M", "Rank_1M", "Final_Rank", "Position"):
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce").astype("Int64")

        fname = (
            f"{indices_universe.replace(' ', '').lower()}_"
            f"{timeframe_label.replace(' ', '').lower()}_"
            f"{period_label.replace(' ', '').lower()}_momentum.csv"
        )
        st.download_button(
            "⬇ Export CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(str(e))
