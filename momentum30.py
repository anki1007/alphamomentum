from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from typing import List, Dict

# -------------------- PAGE + THEME --------------------
st.set_page_config(page_title="Alpha Momentum Screener", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');

:root {
  --app-font: 'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --bg: #0b0e13;
  --bg-2: #10141b;
  --border: #1f2732;
  --border-soft: #1a2230;
  --text: #e6eaee;
  --text-dim: #b3bdc7;
  --accent: #7a5cff;
  --accent-2: #2bb0ff;
}

/* App background + text */
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--app-font) !important; }

/* More top padding so the hero title isn't clipped */
.block-container { padding-top: 1.6rem; }

/* Hero title */
.hero-title {
  font-weight: 800; font-size: clamp(26px, 4.5vw, 40px); line-height: 1.05;
  margin: 18px 0 10px 0;
  background: linear-gradient(90deg, var(--accent-2), var(--accent) 60%);
  -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: .2px;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-weight: 700; color: var(--text-dim) !important; }

/* Buttons */
.stButton button {
  background: linear-gradient(180deg, #1b2432, #131922);
  color: var(--text); border: 1px solid var(--border); border-radius: 10px;
}
.stButton button:hover { filter: brightness(1.06); }

/* Card */
.pro-card {
  background: var(--bg-2); border: 1px solid var(--border); border-radius: 14px;
  padding: 6px 10px 10px 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}

/* Table */
a { text-decoration: none; color: #9ecbff; }
a:hover { text-decoration: underline; }

table { border-collapse: collapse; font-size: 0.86rem; width: 100%; color: var(--text); }
thead th {
  position: sticky; top: 0; z-index: 2; background: #121823; color: var(--text-dim);
  border-bottom: 1px solid var(--border); padding: 6px 8px; white-space: nowrap;
}
tbody td { padding: 6px 8px; border-top: 1px solid var(--border-soft); white-space: nowrap; }
tbody tr:hover td { background: rgba(255,255,255,0.02) !important; }

/* Subheader */
h2, .stMarkdown h2 { color: var(--text); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Alpha Momentum Screener</div>', unsafe_allow_html=True)

# -------------------- CONFIG --------------------
BENCHMARKS: Dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",             # default benchmark
    "Nifty Midcap 150": "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250": "^NIFTYSMLCAP250.NS",
}

GITHUB_BASE = "https://raw.githubusercontent.com/anki1007/alphamomentum/main/"
CSV_FILES: Dict[str, str] = {
    "Nifty 200":           GITHUB_BASE + "nifty200.csv",      # default universe
    "Nifty 500":           GITHUB_BASE + "nifty500.csv",
    "Nifty Midcap 150":    GITHUB_BASE + "niftymidcap150.csv",
    "Nifty Mid Small 400": GITHUB_BASE + "niftymidsmallcap400.csv",
    "Nifty Smallcap 250":  GITHUB_BASE + "niftysmallcap250.csv",
    "Nifty Total Market":  GITHUB_BASE + "niftytotalmarket.csv",
}

RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21
DEFAULT_TOP_N = 30

# -------------------- HELPERS --------------------
def tv_symbol_from_yf(symbol: str) -> str:
    s = symbol.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_chart_url(symbol: str) -> str:
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol_from_yf(symbol)}"

def _pick_close(df: pd.DataFrame | pd.Series, symbol: str) -> pd.Series:
    """Return a clean close/adj close series from a yfinance download() result."""
    if isinstance(df, pd.Series):
        return pd.to_numeric(df, errors="coerce").dropna()

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        for lvl in ("Close", "Adj Close"):
            col = (symbol, lvl)
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                s.index = pd.to_datetime(s.index)
                return s
        return pd.Series(dtype=float)
    else:
        for col in ("Close", "Adj Close"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                s.index = pd.to_datetime(s.index)
                return s
        return pd.Series(dtype=float)

def _zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win).mean()
    s = x.rolling(win).std(ddof=0).replace(0, np.nan)
    return (x - m) / s

def jdk_components(price: pd.Series, bench: pd.Series, win: int = JDK_WINDOW):
    """Robust JdK RS-Ratio & RS-Momentum using rolling z-scores."""
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty or len(df) < win + 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    rs = 100 * (df["p"] / df["b"])
    rs_ratio = 100 + _zscore(rs, win)
    rroc = rs_ratio.pct_change().mul(100)
    rs_mom = 101 + _zscore(rroc, win)

    rs_ratio = rs_ratio.dropna()
    rs_mom = rs_mom.dropna()
    common = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[common], rs_mom.loc[common]

def perf_quadrant(x: float, y: float) -> str:
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100:  return "Improving"
    if x < 100 and y < 100:   return "Lagging"
    return "Weakening"

def analyze_momentum(adj: pd.Series) -> bool:
    """Gate-keeper filter to avoid illiquid/weak names."""
    if adj is None or adj.empty or len(adj) < RS_LOOKBACK_DAYS:
        return False

    ema100 = adj.ewm(span=100, adjust=False).mean()
    try:
        one_year_return = (adj.iloc[-1] / adj.iloc[-RS_LOOKBACK_DAYS] - 1.0) * 100.0
    except Exception:
        return False

    high_52w = adj.iloc[-RS_LOOKBACK_DAYS:].max()
    within_20pct_high = adj.iloc[-1] >= high_52w * 0.8

    if len(adj) < 126:
        return False

    six_month = adj.iloc[-126:]
    up_days_pct = (six_month.pct_change() > 0).sum() / len(six_month) * 100.0

    return (adj.iloc[-1] >= ema100.iloc[-1]
            and one_year_return >= 6.5
            and within_20pct_high
            and up_days_pct > 45.0)

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
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    return df

def _period_years_to_dates(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    years_map = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}
    years = years_map.get(period, 2)  # default 2y
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    end = today_ist + pd.Timedelta(days=1)      # exclusive, ensures today included
    start = today_ist - pd.DateOffset(years=years)
    return start, end

@st.cache_data(show_spinner=True)
def fetch_prices(tickers: List[str], benchmark: str, period: str) -> pd.DataFrame:
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
            st.warning("Yahoo Finance rate limited the request. Please try again shortly.")
        else:
            st.error(f"Data download failed: {e}")
        return pd.DataFrame()

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            pass
    return data

# ---------- Row band colors (dark mode) ----------
def row_bg_for_serial(sno: int) -> str:
    if sno <= 30: return "rgba(46, 204, 113, 0.12)"   # green tint
    if sno <= 60: return "rgba(255, 204, 0, 0.12)"    # amber tint
    if sno <= 90: return "rgba(52, 152, 219, 0.12)"   # blue tint
    return "rgba(231, 76, 60, 0.12)"                  # red tint

def build_table_dataframe(raw: pd.DataFrame, benchmark: str, universe_df: pd.DataFrame) -> pd.DataFrame:
    bench = _pick_close(raw, benchmark).dropna()
    if bench.empty:
        raise RuntimeError(f"Benchmark {benchmark} series empty.")

    cutoff = bench.index.max() - pd.Timedelta(days=RS_LOOKBACK_DAYS + 5)
    bench_rs = bench.loc[bench.index >= cutoff].copy()

    rows = []
    for _, rec in universe_df.iterrows():
        sym, name, industry = rec.Symbol, rec.Name, rec.Industry
        s = _pick_close(raw, sym).dropna()
        if s.empty:
            continue

        if not analyze_momentum(s):
            continue

        s_rs = s.loc[s.index >= cutoff].copy()
        rr, mm = jdk_components(s_rs, bench_rs)
        if rr.empty or mm.empty:
            continue

        ix = rr.index.intersection(mm.index)
        last_rr = float(rr.loc[ix].iloc[-1])
        last_mm = float(mm.loc[ix].iloc[-1])

        rows.append({
            "Name": name,
            "Industry": industry,
            "Return_6M": float((s.iloc[-1] / s.iloc[-126] - 1) * 100) if len(s) >= 126 else np.nan,
            "Return_3M": float((s.iloc[-1] / s.iloc[-63]  - 1) * 100)  if len(s) >= 63  else np.nan,
            "Return_1M": float((s.iloc[-1] / s.iloc[-21]  - 1) * 100)  if len(s) >= 21  else np.nan,
            "RS-Ratio": round(last_rr, 2),
            "RS-Momentum": round(last_mm, 2),
            "Performance": perf_quadrant(last_rr, last_mm),
            "Symbol": sym,
            "Chart": tradingview_chart_url(sym),
        })

    if not rows:
        raise RuntimeError("No tickers passed the filters. Try a longer Period (e.g., 3y) or widen filters.")

    df = pd.DataFrame(rows)

    # Round returns to 2dp for consistency
    for c in ("Return_6M", "Return_3M", "Return_1M"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    # Ranks
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min")

    # Final rank (cast to integer-like), Position = S.No later
    df["Final_Rank"] = (df["Rank_6M"] + df["Rank_3M"] + df["Rank_1M"]).round(0).astype("Int64")

    df = df.sort_values(["Final_Rank", "Rank_6M", "Rank_3M", "Rank_1M"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "S.No", np.arange(1, len(df) + 1))
    df["Position"] = df["S.No"].astype("Int64")

    order = ["S.No", "Name", "Industry",
             "Return_6M", "Rank_6M",
             "Return_3M", "Rank_3M",
             "Return_1M", "Rank_1M",
             "RS-Ratio", "RS-Momentum", "Performance",
             "Final_Rank", "Position", "Chart", "Symbol"]
    return df[order]

def style_rows(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    UI formatting: all numeric columns -> 2 decimals,
    EXCEPT Final_Rank and Position -> 0 decimals.
    """
    def _row_style(r: pd.Series):
        bg = row_bg_for_serial(int(r["S.No"]))
        return [f"background-color: {bg}"] * len(df.columns)

    styler = df.style.apply(lambda rr: _row_style(rr), axis=1).format(escape=None)

    # Identify columns
    text_cols = ["Name", "Industry"]
    num_cols = [c for c in df.columns if c not in text_cols]
    int_like_exceptions = [c for c in ("Final_Rank", "Position") if c in df.columns]

    # Alignment
    styler = styler.set_properties(subset=text_cols, **{"text-align": "left"})
    styler = styler.set_properties(subset=num_cols, **{"text-align": "right", "font-variant-numeric": "tabular-nums"})

    # Formats
    fmt_map = {c: "{:.2f}" for c in num_cols}
    for c in int_like_exceptions:
        fmt_map[c] = "{:.0f}"  # keep these integer-looking

    styler = styler.format(fmt_map, na_rep="–")
    styler = styler.set_table_styles([{"selector": "table", "props": "border-collapse: collapse;"}])

    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass

    return styler

# -------------------- SIDEBAR --------------------
st.sidebar.header("Controls")
indices_universe = st.sidebar.selectbox("Indices Universe", list(CSV_FILES.keys()), index=0)    # Nifty 200
benchmark_key    = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()), index=2)          # Nifty 500
timeframe        = st.sidebar.selectbox("Timeframe (EOD only)", ["1d"], index=0)                # locked 1d
period           = st.sidebar.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1)            # default 2y
top_n            = st.sidebar.number_input("Show Top N (table only)", min_value=5, max_value=200, value=DEFAULT_TOP_N, step=5)
do_load          = st.sidebar.button("Load / Refresh", use_container_width=True)

if "ran_once" not in st.session_state:
    st.session_state.ran_once = True
    do_load = True

# -------------------- ACTION --------------------
if do_load:
    try:
        uni_url = CSV_FILES[indices_universe]
        universe_df = load_universe_from_csv(uni_url)

        benchmark = BENCHMARKS[benchmark_key]
        tickers = universe_df["Symbol"].tolist()

        with st.spinner("Fetching EOD prices…"):
            raw = fetch_prices(tickers, benchmark, period=period)

        if raw.empty:
            st.stop()

        df = build_table_dataframe(raw, benchmark, universe_df)

        # ---------- On-screen table (Top N only) ----------
        ui_cols = [
            "S.No", "Name", "Industry",
            "Return_6M", "Rank_6M",
            "Return_3M", "Rank_3M",
            "Return_1M", "Rank_1M",
            "RS-Ratio", "RS-Momentum", "Performance",
            "Final_Rank", "Position", "Chart"
        ]
        display_df = df[ui_cols].copy()

        # Clickable name -> TradingView (safe HTML)
        display_df["Name"] = display_df.apply(
            lambda r: f'<a href="{r["Chart"]}" target="_blank" rel="noopener noreferrer">{r["Name"]}</a>',
            axis=1
        )
        display_df = display_df.drop(columns=["Chart"])

        # Guard top-N against available rows
        top_n = int(min(top_n, len(display_df)))
        st.subheader(f"Alpha Momentum {top_n}")

        display_df_top = display_df.iloc[:top_n].copy()
        table_html = style_rows(display_df_top).to_html()
        st.markdown(f'<div class="pro-card">{table_html}</div>', unsafe_allow_html=True)

        st.caption(f"{len(df)} results • {indices_universe} • {benchmark_key} • 1d EOD • {period}")

        # ---------- CSV export: full ranked table (Symbol hidden) ----------
        csv_bytes = df.drop(columns=["Symbol"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export CSV",
            data=csv_bytes,
            file_name=f"{indices_universe.replace(' ', '').lower()}_momentum.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(str(e))
