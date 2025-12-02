from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from typing import List, Dict

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Rotational Momentum Screener", layout="wide")

# =========================================================
# THEME
# =========================================================
THEMES = {
    "Dark": {
        "import_font": """
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        """,
        "vars": {
            "--app-font": "'IBM Plex Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            "--mono-font": "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace",
            "--bg": "#0a0b0d",
            "--bg-2": "#111317",
            "--border": "#222836",
            "--border-soft": "#1a1f2a",
            "--text": "#e7ebf0",
            "--text-dim": "#9eabbc",
            "--accent": "#00c3ff",
            "--accent-2": "#ffb300",
            # Performance colors
            "--perf-leading": "rgba(46, 204, 113, 0.25)",
            "--perf-improving": "rgba(52, 152, 219, 0.25)",
            "--perf-lagging": "rgba(231, 76, 60, 0.25)",
            "--perf-weakening": "rgba(255, 204, 0, 0.30)",
            # Row banding
            "--band-top": "rgba(46, 204, 113, 0.12)",
            "--band-mid1": "rgba(255, 204, 0, 0.12)",
            "--band-mid2": "rgba(52, 152, 219, 0.12)",
            "--band-rest": "rgba(231, 76, 60, 0.12)",
        },
    },
    "Light": {
        "import_font": """
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        """,
        "vars": {
            "--app-font": "'IBM Plex Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            "--mono-font": "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace",
            "--bg": "#f6f7f9",
            "--bg-2": "#ffffff",
            "--border": "#d9dee5",
            "--border-soft": "#e8ecf2",
            "--text": "#151a22",
            "--text-dim": "#596579",
            "--accent": "#007aff",
            "--accent-2": "#ff8a00",
            "--perf-leading": "rgba(46, 204, 113, 0.20)",
            "--perf-improving": "rgba(52, 152, 219, 0.20)",
            "--perf-lagging": "rgba(231, 76, 60, 0.20)",
            "--perf-weakening": "rgba(255, 204, 0, 0.25)",
            "--band-top": "rgba(46, 204, 113, 0.12)",
            "--band-mid1": "rgba(255, 204, 0, 0.12)",
            "--band-mid2": "rgba(52, 152, 219, 0.12)",
            "--band-rest": "rgba(231, 76, 60, 0.10)",
        },
    },
    # renamed duplicate
    "Dark Condensed": {
        "import_font": """
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        """,
        "vars": {
            "--app-font": "'IBM Plex Sans Condensed', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            "--mono-font": "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace",
            "--bg": "#0b0c10",
            "--bg-2": "#0f1116",
            "--border": "#242a38",
            "--border-soft": "#1a1f2b",
            "--text": "#e5ecf3",
            "--text-dim": "#9aa7bb",
            "--accent": "#10e7ff",
            "--accent-2": "#ffcd00",
            "--perf-leading": "rgba(46, 204, 113, 0.25)",
            "--perf-improving": "rgba(52, 152, 219, 0.25)",
            "--perf-lagging": "rgba(231, 76, 60, 0.25)",
            "--perf-weakening": "rgba(255, 204, 0, 0.30)",
            "--band-top": "rgba(46, 204, 113, 0.14)",
            "--band-mid1": "rgba(255, 204, 0, 0.14)",
            "--band-mid2": "rgba(52, 152, 219, 0.14)",
            "--band-rest": "rgba(231, 76, 60, 0.14)",
        },
    },
}

def render_theme_css(theme_name: str):
    theme = THEMES[theme_name]
    vars_css = "\n".join(f"{k}: {v};" for k, v in theme["vars"].items())
    st.markdown(f"""
    <style>
    {theme["import_font"]}

    :root {{
        {vars_css}
    }}

    html, body, .stApp {{
      background: var(--bg) !important;
      color: var(--text) !important;
      font-family: var(--app-font) !important;
    }}
    .block-container {{ padding-top: 1.6rem; }}

    .hero-title {{
      font-weight: 700;
      font-size: clamp(24px, 4vw, 36px);
      line-height: 1.05;
      margin: 14px 0 8px 0;
      background: linear-gradient(90deg, var(--accent), var(--accent-2) 70%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      letter-spacing: .2px;
    }}

    section[data-testid="stSidebar"] {{
      background: var(--bg-2) !important;
      border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] * {{ color: var(--text) !important; }}
    section[data-testid="stSidebar"] label {{ font-weight: 600; color: var(--text-dim) !important; }}

    .stButton button {{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.20));
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .stButton button:hover {{ filter: brightness(1.06); }}

    .pro-card {{
      background: var(--bg-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 6px 10px 10px 10px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }}

    a {{ text-decoration: none; color: var(--accent); }}
    a:hover {{ text-decoration: underline; }}

    /* Table base */
    table {{
      border-collapse: collapse;
      font-size: 0.86rem;
      width: 100%;
      color: var(--text);
    }}
    thead th {{
      position: sticky; top: 0; z-index: 2;
      background: color-mix(in oklab, var(--bg-2), #000 8%);
      color: var(--text-dim);
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
      white-space: nowrap;
      font-weight: 600;
      text-align: center;   /* center all headers */
    }}
    tbody td {{
      padding: 6px 8px;
      border-top: 1px solid var(--border-soft);
      white-space: nowrap;
    }}
    tbody tr:hover td {{ background: rgba(255,255,255,0.03) !important; }}

    h2, .stMarkdown h2 {{ color: var(--text); }}
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# UI header
# =========================================================
st.sidebar.header("Appearance")
theme_choice = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=0)
render_theme_css(theme_choice)

st.markdown('<div class="hero-title">Rotational Momentum Screener</div>', unsafe_allow_html=True)

# =========================================================
# DATA / CONFIG
# =========================================================
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

# =========================================================
# Helpers
# =========================================================
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
    within_40pct_high = adj.iloc[-1] >= high_52w * 0.6
    if len(adj) < 126:
        return None
    six_month = adj.iloc[-126:]
    up_days_pct = (six_month.pct_change() > 0).sum() / len(six_month) * 100.0
    if (adj.iloc[-1] >= ema100.iloc[-1] and one_year_return >= 6.5 and
        within_40pct_high and up_days_pct > 30.0):
        try:
            r6 = (adj.iloc[-1] / adj.iloc[-126] - 1.0) * 100.0
            r3 = (adj.iloc[-1] / adj.iloc[-63]  - 1.0) * 100.0
            r1 = (adj.iloc[-1] / adj.iloc[-21]  - 1.0) * 100.0
        except Exception:
            return None
        return {"Return_6M": r6, "Return_3M": r3, "Return_1M": r1}
    return None

@st.cache_data(show_spinner=False, ttl=3600)
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
    years = years_map.get(period, 2)
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    end = today_ist + pd.Timedelta(days=1)
    start = today_ist - pd.DateOffset(years=years)
    return start, end

@st.cache_data(show_spinner=True, ttl=1800)
def fetch_prices(tickers: List[str], benchmark: str, period: str, interval: str = "1d") -> pd.DataFrame:
    interval = "1d"
    start, end = _period_years_to_dates(period)
    try:
        data = yf.download(
            tickers + [benchmark],
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval=interval,
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

def row_bg_for_serial(sno: int) -> str:
    if sno <= 30: return "var(--band-top)"
    if sno <= 60: return "var(--band-mid1)"
    if sno <= 90: return "var(--band-mid2)"
    return "var(--band-rest)"

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
        raise RuntimeError("No tickers passed the filters. Try a longer Period (e.g., 3y) with 1d timeframe.")
    df = pd.DataFrame(rows)

  
    for c in ("Return_6M", "Return_3M", "Return_1M"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
   
    df["RS-Ratio"] = pd.to_numeric(df["RS-Ratio"], errors="coerce").round(2)
    df["RS-Momentum"] = pd.to_numeric(df["RS-Momentum"], errors="coerce").round(2)
  
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min").astype(int)
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min").astype(int)
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min").astype(int)
    df["Final_Rank"] = (df["Rank_6M"] + df["Rank_3M"] + df["Rank_1M"]).astype(int)

    # Sort by RS first as per UI
    df = df.sort_values(by=["RS-Momentum", "RS-Ratio"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "S.No", np.arange(1, len(df) + 1).astype(int))
    df["Position"] = df["S.No"].astype(int)

    order = ["S.No", "Name", "Industry",
             "Return_6M", "Rank_6M",
             "Return_3M", "Rank_3M",
             "Return_1M", "Rank_1M",
             "RS-Ratio", "RS-Momentum", "Performance",
             "Final_Rank", "Position", "Chart", "Symbol"]
    return df[order]

def style_rows(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _row_style(r: pd.Series):
        bg = row_bg_for_serial(int(r["S.No"]))
        return [f"background-color: {bg}"] * len(df.columns)

    styler = df.style.apply(lambda rr: _row_style(rr), axis=1).format(escape=None)


    left_cols = ["Name", "Industry"]
    center_cols = [c for c in df.columns if c not in left_cols]

    styler = styler.set_properties(subset=left_cols, **{"text-align": "left"})
    styler = styler.set_properties(subset=center_cols, **{
        "text-align": "center",
        "font-variant-numeric": "tabular-nums"
    })

 
    base_rules = [{"selector": "table", "props": "border-collapse: collapse;"}]
    styler = styler.set_table_styles(base_rules)

   
    perf_color_map = {
        "Leading":   "var(--perf-leading)",
        "Improving": "var(--perf-improving)",
        "Lagging":   "var(--perf-lagging)",
        "Weakening": "var(--perf-weakening)",
    }
    perf_colors = df["Performance"].map(lambda v: perf_color_map.get(str(v), "transparent"))
    perf_styles = pd.DataFrame("", index=df.index, columns=df.columns)
    perf_styles.loc[:, "Performance"] = perf_colors.map(lambda c: f"background-color: {c}; font-weight: 700;")
    styler = styler.set_td_classes(pd.DataFrame("", index=df.index, columns=df.columns))
    styler = styler.set_properties(**{}).apply(lambda _: perf_styles, axis=None)

    try:
        styler = styler.hide(axis="index")
    except Exception:
        try:
            styler = styler.hide_index()
        except Exception:
            pass
    return styler


st.sidebar.header("Controls")
indices_universe = st.sidebar.selectbox("Indices Universe", list(CSV_FILES.keys()), index=0)
benchmark_key    = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()), index=2)
timeframe        = st.sidebar.selectbox("Timeframe (EOD only)", ["1d"], index=0)
period           = st.sidebar.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1)
do_load          = st.sidebar.button("Load / Refresh", use_container_width=True)

if "ran_once" not in st.session_state:
    st.session_state.ran_once = True
    do_load = True


if do_load:
    try:
        uni_url = CSV_FILES[indices_universe]
        universe_df = load_universe_from_csv(uni_url)

        benchmark = BENCHMARKS[benchmark_key]
        tickers = universe_df["Symbol"].tolist()

        with st.spinner("Fetching EOD prices…"):
            raw = fetch_prices(tickers, benchmark, period=period, interval="1d")

        if raw.empty:
            st.stop()

        df = build_table_dataframe(raw, benchmark, universe_df)

      
        ui_cols = [
            "S.No", "Name", "Industry",
            "Return_6M", "Rank_6M",
            "Return_3M", "Rank_3M",
            "Return_1M", "Rank_1M",
            "RS-Ratio", "RS-Momentum", "Performance",
            "Final_Rank", "Position", "Chart"
        ]
        display_df = df[ui_cols].copy()
        display_df["Name"] = display_df.apply(
            lambda r: f'<a href="{r["Chart"]}" target="_blank" rel="noopener noreferrer">{r["Name"]}</a>',
            axis=1
        )
        display_df = display_df.drop(columns=["Chart"])

     
        table_html = style_rows(display_df).to_html()
        st.markdown(f'<div class="pro-card">{table_html}</div>', unsafe_allow_html=True)

        st.caption(f"{len(df)} results in universe • {indices_universe} • {benchmark_key} • 1d EOD • {period}")

        # Export CSV
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
