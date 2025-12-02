from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from typing import List, Dict

# -------------------- PAGE & GLOBAL STYLE --------------------
st.set_page_config(page_title="Alpha Momentum Screener", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');

:root { --app-font: 'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
html, body, [class*="css"], .stMarkdown, .stText, .stSelectbox, .stButton, .stDataFrame {
  font-family: var(--app-font) !important;
}

.block-container { padding-top: 1.2rem; }

/* Hero title */
.hero-title {
  font-weight: 800; font-size: clamp(28px, 5vw, 48px); line-height: 1.1;
  margin: 6px 0 12px 0;
  background: linear-gradient(90deg, #2bb0ff, #7a5cff 45%, #ff6cab 90%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  letter-spacing: .2px;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label { font-weight: 700; }

/* Table polish */
a { text-decoration: none; }
a:hover { text-decoration: underline; }
table { border-collapse: collapse; font-size: 0.95rem; width: 100%; }
thead th { position: sticky; top: 0; z-index: 1; background: #e9f0f7; }
th, td { border: 1px solid #c9d1d9; padding: 6px 10px; }
tbody tr:hover td { filter: brightness(0.98); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Alpha Momentum Screener</div>', unsafe_allow_html=True)

# -------------------- CONFIG --------------------
BENCHMARKS: Dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 150": "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250": "^NIFTYSMLCAP250.NS",
}
GITHUB_BASE = "https://raw.githubusercontent.com/anki1007/alphamomentum/main/"
CSV_FILES: Dict[str, str] = {
    "Nifty 200":           GITHUB_BASE + "nifty200.csv",
    "Nifty 500":           GITHUB_BASE + "nifty500.csv",
    "Nifty Midcap 150":    GITHUB_BASE + "niftymidcap150.csv",
    "Nifty Mid Small 400": GITHUB_BASE + "niftymidsmallcap400.csv",
    "Nifty Smallcap 250":  GITHUB_BASE + "niftysmallcap250.csv",
    "Nifty Total Market":  GITHUB_BASE + "niftytotalmarket.csv",
}
RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21

# -------------------- HELPERS --------------------
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
    else:
        for col in ("Close", "Adj Close"):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").dropna()
        return pd.Series(dtype=float)

def jdk_components(price: pd.Series, bench: pd.Series, win: int = JDK_WINDOW):
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
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100:  return "Improving"
    if x < 100 and y < 100:   return "Lagging"
    return "Weakening"

def analyze_momentum(adj: pd.Series) -> dict | None:
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
            r3 = (adj.iloc[-1] / adj.iloc[-63]  - 1.0) * 100.0
            r1 = (adj.iloc[-1] / adj.iloc[-21]  - 1.0) * 100.0
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
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    return df

@st.cache_data(show_spinner=True)
def fetch_prices(tickers: List[str], benchmark: str, period: str, interval: str) -> pd.DataFrame:
    """Robust downloader with simple handling for YF rate limiting."""
    try:
        data = yf.download(
            tickers + [benchmark],
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as e:
        msg = str(e)
        if "Rate limited" in msg or "Too Many Requests" in msg:
            st.warning("Yahoo Finance rate limited the request. Please try again in ~1–2 minutes.")
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
    if sno <= 30: return "#dff5df"  # light green
    if sno <= 60: return "#fff6b3"  # light yellow
    if sno <= 90: return "#dfe9ff"  # light blue
    return "#f7d6d6"                # light red

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
        mom = analyze_momentum(s)
        if mom is None:
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
        raise RuntimeError("No tickers passed the filters. Try a longer Period and Timeframe=1d.")
    df = pd.DataFrame(rows)

    # Round and ranks
    for c in ("Return_6M", "Return_3M", "Return_1M"): df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
    df["RS-Ratio"] = pd.to_numeric(df["RS-Ratio"], errors="coerce").round(2)
    df["RS-Momentum"] = pd.to_numeric(df["RS-Momentum"], errors="coerce").round(2)
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min")
    df["Final_Rank"] = df["Rank_6M"] + df["Rank_3M"] + df["Rank_1M"]
    df = df.sort_values("Final_Rank", kind="mergesort").reset_index(drop=True)
    df.insert(0, "S.No", np.arange(1, len(df) + 1))
    df["Position"] = df["S.No"]

    order = ["S.No", "Name", "Industry",
             "Return_6M", "Rank_6M",
             "Return_3M", "Rank_3M",
             "Return_1M", "Rank_1M",
             "RS-Ratio", "RS-Momentum", "Performance",
             "Final_Rank", "Position", "Chart", "Symbol"]
    return df[order]

def style_rows(df: pd.DataFrame):
    """Row banding + alignment; allow HTML on Name."""
    def _row_style(r: pd.Series):
        bg = row_bg_for_serial(int(r["S.No"]))
        return [f"background-color: {bg}"] * len(df.columns)

    styler = df.style.apply(lambda rr: _row_style(rr), axis=1)
    styler = styler.set_properties(subset=["Name", "Industry"], **{"text-align": "left"})
    styler = styler.set_properties(subset=[c for c in df.columns if c not in ("Name", "Industry")],
                                   **{"text-align": "center"})
    styler = styler.format({"Name": lambda v: v}, escape=False)  # Name contains <a>
    try: styler = styler.hide(axis="index")
    except Exception: pass
    return styler

# -------------------- SIDEBAR (DEFAULTS) --------------------
st.sidebar.header("Controls")
# Defaults requested: Benchmark = "Nifty 500", Universe = "Nifty 200", Timeframe = "1d", Period = "1y"
indices_universe = st.sidebar.selectbox("Indices Universe", list(CSV_FILES.keys()), index=0)  # "Nifty 200"
benchmark_key    = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()), index=2)        # "Nifty 500"
timeframe        = st.sidebar.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)           # "1d"
period           = st.sidebar.selectbox("Period", ["1y", "2y", "3y", "5y"], index=0)          # "1y"
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

        with st.spinner("Fetching prices…"):
            raw = fetch_prices(tickers, benchmark, period=period, interval=timeframe)

        if raw.empty:
            st.stop()

        df = build_table_dataframe(raw, benchmark, universe_df)

        # Build UI view (hyperlink on Name; hide Chart)
        ui_cols = ["S.No", "Name", "Industry",
                   "Return_6M", "Rank_6M",
                   "Return_3M", "Rank_3M",
                   "Return_1M", "Rank_1M",
                   "RS-Ratio", "RS-Momentum", "Performance",
                   "Final_Rank", "Position", "Chart"]
        display_df = df[ui_cols].copy()
        display_df["Name"] = display_df.apply(
            lambda r: f'<a href="{r["Chart"]}" target="_blank" rel="noopener noreferrer">{r["Name"]}</a>', axis=1
        )
        display_df = display_df.drop(columns=["Chart"])

        st.subheader("Screened Momentum Table")
        st.markdown(style_rows(display_df).to_html(), unsafe_allow_html=True)

        st.caption(f"{len(df)} results • {indices_universe} • {benchmark_key} • {timeframe} • {period}")

        # Export (hide Symbol in export to match UI; change if you prefer keeping it)
        csv_bytes = df.drop(columns=["Symbol"]).to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", csv_bytes,
                           file_name=f"{indices_universe.replace(' ', '').lower()}_momentum.csv",
                           mime="text/csv", use_container_width=True)

    except Exception as e:
        st.error(str(e))
