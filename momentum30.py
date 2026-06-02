from __future__ import annotations

import html as _html
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
  font-weight: 800; font-size: clamp(26px, 4.5vw, 40px); line-height: 1.05; margin: 18px 0 10px 0;
  background: linear-gradient(90deg, var(--accent-2), var(--accent) 60%); -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: .2px;
}
section[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-weight: 700; color: var(--text-dim) !important; }
.stButton button { background: linear-gradient(180deg, #1b2432, #131922); color: var(--text); border: 1px solid var(--border); border-radius: 10px; }
.stButton button:hover { filter: brightness(1.06); }
.pro-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 14px; padding: 6px 10px 10px 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
a { text-decoration: none; color: #9ecbff; } a:hover { text-decoration: underline; }
table { border-collapse: collapse; font-size: 0.86rem; width: 100%; color: var(--text); }
thead th { position: sticky; top: 0; z-index: 2; background: #121823; color: var(--text-dim); border-bottom: 1px solid var(--border); padding: 6px 8px; white-space: nowrap; }
tbody td { padding: 6px 8px; border-top: 1px solid var(--border-soft); white-space: nowrap; }
h2, .stMarkdown h2 { color: var(--text); }
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
    "Nifty 50":           GITHUB_BASE + "nifty50.csv",
    "Nifty 100":           GITHUB_BASE + "nifty100.csv",
    "Nifty 200":           GITHUB_BASE + "nifty200.csv",
    "Nifty 500":           GITHUB_BASE + "nifty500.csv",
    "Nifty Midcap 150":    GITHUB_BASE + "niftymidcap150.csv",
    "Nifty Mid Small 400": GITHUB_BASE + "niftymidsmallcap400.csv",
    "Nifty Smallcap 250":  GITHUB_BASE + "niftysmallcap250.csv",
    "Nifty Total Market":  GITHUB_BASE + "niftytotalmarket.csv",
}
RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21

# Risk-adjusted metric settings (used by UPI & Sharpe)
RISK_METRIC_WINDOW = 252        # trailing trading days (~1 year)
RISK_FREE_ANNUAL = 0.065        # annualized risk-free rate as a decimal (6.5%)
TRADING_DAYS = 252.0

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

# -------------------- RISK-ADJUSTED METRICS --------------------
def ulcer_index(price: pd.Series, win: int = RISK_METRIC_WINDOW) -> float:
    """Ulcer Index: RMS of percentage drawdowns from the running peak (lower = smoother)."""
    s = pd.to_numeric(price, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    if len(s) > win:
        s = s.iloc[-win:]
    running_max = s.cummax()
    dd = ((s - running_max) / running_max) * 100.0
    return float(np.sqrt(np.mean(np.square(dd))))

def ulcer_performance_index(price: pd.Series, win: int = RISK_METRIC_WINDOW,
                            rf_annual: float = RISK_FREE_ANNUAL) -> float:
    """Ulcer Performance Index (Martin Ratio) = excess annualized return / Ulcer Index."""
    s = pd.to_numeric(price, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    if len(s) > win:
        s = s.iloc[-win:]
    ui = ulcer_index(s, win)
    if not np.isfinite(ui) or ui == 0:
        return np.nan
    n = len(s)
    total_ret = s.iloc[-1] / s.iloc[0] - 1.0
    ann_ret = (1.0 + total_ret) ** (TRADING_DAYS / n) - 1.0
    return float((ann_ret - rf_annual) * 100.0 / ui)

def sharpe_ratio(price: pd.Series, win: int = RISK_METRIC_WINDOW,
                 rf_annual: float = RISK_FREE_ANNUAL) -> float:
    """Annualized Sharpe Ratio from daily returns (higher = better risk-adjusted return)."""
    s = pd.to_numeric(price, errors="coerce").dropna()
    if len(s) < 3:
        return np.nan
    if len(s) > win:
        s = s.iloc[-win:]
    rets = s.pct_change().dropna()
    sd = rets.std(ddof=0)
    if not np.isfinite(sd) or sd == 0 or len(rets) < 2:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS
    return float(((rets.mean() - rf_daily) / sd) * np.sqrt(TRADING_DAYS))

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

def _period_years_to_dates(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    years_map = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}
    years = years_map.get(period, 2)
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    end = today_ist + pd.Timedelta(days=1)
    start = today_ist - pd.DateOffset(years=years)
    return start, end

@st.cache_data(show_spinner=True)
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
        try: data.index = pd.to_datetime(data.index)
        except Exception: pass
    return data

def row_bg_for_serial(sno: int) -> str:
    if sno <= 30: return "rgba(46, 204, 113, 0.12)"
    if sno <= 60: return "rgba(255, 204, 0, 0.12)"
    if sno <= 90: return "rgba(52, 152, 219, 0.12)"
    return "rgba(231, 76, 60, 0.12)"

def build_table_dataframe(raw: pd.DataFrame, benchmark: str, universe_df: pd.DataFrame) -> pd.DataFrame:
    bench = _pick_close(raw, benchmark).dropna()
    if bench.empty: raise RuntimeError(f"Benchmark {benchmark} series empty.")

    cutoff = bench.index.max() - pd.Timedelta(days=RS_LOOKBACK_DAYS + 5)
    bench_rs = bench.loc[bench.index >= cutoff].copy()

    rows = []
    for _, rec in universe_df.iterrows():
        sym, name, industry = rec.Symbol, rec.Name, rec.Industry
        s = _pick_close(raw, sym).dropna()
        if s.empty: continue
        if analyze_momentum(s) is None: continue

        s_rs = s.loc[s.index >= cutoff].copy()
        rr, mm = jdk_components(s_rs, bench_rs)
        if rr.empty or mm.empty: continue

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
            "UPI": ulcer_performance_index(s),
            "Sharpe": sharpe_ratio(s),
            "Symbol": sym,
            "Chart": tradingview_chart_url(sym),
        })

    if not rows:
        raise RuntimeError("No tickers passed the filters. Try a longer Period (e.g., 3y) with 1d timeframe.")
    df = pd.DataFrame(rows)

    # round numerics
    for c in ("Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum", "UPI", "Sharpe"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    # ranks as ints
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min").astype("Int64")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min").astype("Int64")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min").astype("Int64")
    df["Final_Rank"] = (df["Rank_6M"].fillna(0) + df["Rank_3M"].fillna(0) + df["Rank_1M"].fillna(0)).astype("Int64")

    df = df.sort_values("Final_Rank", kind="mergesort").reset_index(drop=True)
    df.insert(0, "S.No", np.arange(1, len(df) + 1, dtype=int))
    # Position = momentum-based position (stays fixed even if the view is re-sorted by UPI / Sharpe)
    df["Position"] = df["S.No"].astype(int)

    order = ["S.No", "Name", "Industry",
             "Return_6M", "Rank_6M",
             "Return_3M", "Rank_3M",
             "Return_1M", "Rank_1M",
             "RS-Ratio", "RS-Momentum", "Performance",
             "UPI", "Sharpe",
             "Final_Rank", "Position", "Chart", "Symbol"]
    return df[order]

def apply_sort(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Reorder the view by the chosen key and re-number S.No (Position stays momentum-based)."""
    sort_map = {
        "Final Rank":   ("Final_Rank", True),    # ascending: rank 1 is best
        "UPI":          ("UPI", False),          # descending: higher is better
        "Sharpe Ratio": ("Sharpe", False),       # descending: higher is better
    }
    col, asc = sort_map.get(sort_by, ("Final_Rank", True))
    out = df.sort_values(col, ascending=asc, kind="mergesort", na_position="last").reset_index(drop=True)
    out["S.No"] = np.arange(1, len(out) + 1, dtype=int)
    return out

def style_rows(df: pd.DataFrame):
    def _row_style(r: pd.Series):
        bg = row_bg_for_serial(int(r["S.No"]))
        return [f"background-color: {bg}"] * len(df.columns)
    styler = df.style.apply(lambda rr: _row_style(rr), axis=1)

    # Center all except Name & Industry
    text_cols = ["Name", "Industry"]
    center_cols = [c for c in df.columns if c not in text_cols]

    styler = styler.set_properties(subset=text_cols, **{"text-align": "left"})
    styler = styler.set_properties(subset=center_cols, **{"text-align": "center", "font-variant-numeric": "tabular-nums"})

    try: styler = styler.hide(axis="index")
    except Exception: pass
    return styler

# -------------------- SORTABLE HTML TABLE (client-side, click headers) --------------------
TABLE_COLUMNS = [
    ("S.No",        "S.No",        "num"),
    ("Name",        "Name",        "text"),
    ("Industry",    "Industry",    "text"),
    ("Return_6M",   "Return_6M",   "num"),
    ("Rank_6M",     "Rank_6M",     "num"),
    ("Return_3M",   "Return_3M",   "num"),
    ("Rank_3M",     "Rank_3M",     "num"),
    ("Return_1M",   "Return_1M",   "num"),
    ("Rank_1M",     "Rank_1M",     "num"),
    ("RS-Ratio",    "RS-Ratio",    "num"),
    ("RS-Momentum", "RS-Momentum", "num"),
    ("Performance", "Performance", "text"),
    ("UPI",         "UPI",         "num"),
    ("Sharpe",      "Sharpe",      "num"),
    ("Final_Rank",  "Final_Rank",  "num"),
    ("Position",    "Position",    "num"),
]
INT_KEYS = {"S.No", "Rank_6M", "Rank_3M", "Rank_1M", "Final_Rank", "Position"}
DEC2_KEYS = {"Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum", "UPI", "Sharpe"}

def build_sortable_table_html(df: pd.DataFrame) -> tuple[str, int]:
    """Self-contained HTML table with click-to-sort headers, rendered in an iframe via components.html."""
    # ---- header ----
    head_cells = []
    for i, (key, label, typ) in enumerate(TABLE_COLUMNS):
        align = "left" if key in ("Name", "Industry") else "center"
        head_cells.append(
            f'<th class="sortable" data-col="{i}" data-type="{typ}" '
            f'style="text-align:{align}">{_html.escape(label)}<span class="arrow"></span></th>'
        )
    thead = "<thead><tr>" + "".join(head_cells) + "</tr></thead>"

    # ---- body ----
    body_rows = []
    for _, r in df.iterrows():
        tds = []
        for key, label, typ in TABLE_COLUMNS:
            v = r[key]
            if key == "Name":
                name = _html.escape(str(v))
                url = _html.escape(str(r["Chart"]), quote=True)
                disp = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{name}</a>'
                sortval = _html.escape(str(v).lower(), quote=True)
                tds.append(f'<td data-sort="{sortval}" style="text-align:left">{disp}</td>')
                continue
            if typ == "text":
                disp = _html.escape(str(v)) if not pd.isna(v) else "-"
                sortval = _html.escape(str(v).lower(), quote=True) if not pd.isna(v) else ""
                align = "left" if key == "Industry" else "center"
                tds.append(f'<td data-sort="{sortval}" style="text-align:{align}">{disp}</td>')
                continue
            # numeric
            if pd.isna(v):
                disp, sortval = "-", ""
            elif key in INT_KEYS:
                disp = sortval = f"{int(v)}"
            else:
                disp, sortval = f"{float(v):.2f}", f"{float(v)}"
            tds.append(
                f'<td data-sort="{sortval}" '
                f'style="text-align:center;font-variant-numeric:tabular-nums">{disp}</td>'
            )
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"

    css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');
      :root { --bg-2:#10141b; --border:#1f2732; --border-soft:#1a2230; --text:#e6eaee; --text-dim:#b3bdc7; }
      html, body { margin:0; padding:0; background:transparent;
        font-family:'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
      .pro-card { background:var(--bg-2); border:1px solid var(--border); border-radius:14px;
        padding:6px 10px 10px 10px; box-shadow:0 6px 18px rgba(0,0,0,0.35); }
      table { border-collapse:collapse; font-size:0.86rem; width:100%; color:var(--text); }
      thead th { position:sticky; top:0; z-index:2; background:#121823; color:var(--text-dim);
        border-bottom:1px solid var(--border); padding:6px 8px; white-space:nowrap; }
      th.sortable { cursor:pointer; user-select:none; }
      th.sortable:hover { color:var(--text); background:#172033; }
      th .arrow { color:#9ecbff; font-size:0.8em; }
      tbody td { padding:6px 8px; border-top:1px solid var(--border-soft); white-space:nowrap; }
      a { text-decoration:none; color:#9ecbff; } a:hover { text-decoration:underline; }
    </style>
    """

    script = """
    <script>
    (function(){
      const table = document.getElementById('ams-table');
      const tbody = table.tBodies[0];
      function recolor(){
        const rows = tbody.querySelectorAll('tr');
        rows.forEach((r,i)=>{
          const n=i+1; let bg;
          if(n<=30) bg='rgba(46,204,113,0.12)';
          else if(n<=60) bg='rgba(255,204,0,0.12)';
          else if(n<=90) bg='rgba(52,152,219,0.12)';
          else bg='rgba(231,76,60,0.12)';
          r.style.backgroundColor=bg;
        });
      }
      function clearArrows(except){
        table.querySelectorAll('th.sortable').forEach(th=>{
          if(th!==except){ th.removeAttribute('data-dir');
            const a=th.querySelector('.arrow'); if(a) a.textContent=''; }
        });
      }
      table.querySelectorAll('th.sortable').forEach(th=>{
        th.addEventListener('click', ()=>{
          const col=parseInt(th.dataset.col,10);
          const type=th.dataset.type;
          const cur=th.dataset.dir;
          const dir = !cur ? (type==='num'?'desc':'asc') : (cur==='asc'?'desc':'asc');
          clearArrows(th);
          th.dataset.dir=dir;
          const a=th.querySelector('.arrow'); if(a) a.textContent = dir==='asc'?' \\u25B2':' \\u25BC';
          const mul = dir==='asc'?1:-1;
          const rows = Array.from(tbody.querySelectorAll('tr'));
          rows.sort((ra,rb)=>{
            let x=ra.children[col].dataset.sort;
            let y=rb.children[col].dataset.sort;
            const xn=(x===''||x===undefined), yn=(y===''||y===undefined);
            if(xn&&yn) return 0; if(xn) return 1; if(yn) return -1;  /* blanks always last */
            if(type==='num'){ x=parseFloat(x); y=parseFloat(y); }
            if(x<y) return -1*mul; if(x>y) return 1*mul; return 0;
          });
          rows.forEach(r=>tbody.appendChild(r));
          recolor();
        });
      });
      recolor();
    })();
    </script>
    """

    table_html = f'<div class="pro-card"><table id="ams-table">{thead}{tbody}</table></div>'
    full_html = css + table_html + script
    height = min(820, 96 + 34 * (len(df) + 1))
    return full_html, height

# -------------------- UI --------------------
st.sidebar.header("Controls")
indices_universe = st.sidebar.selectbox("Indices Universe", list(CSV_FILES.keys()), index=0)
benchmark_key    = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()), index=2)
timeframe        = st.sidebar.selectbox("Timeframe (EOD only)", ["1d"], index=0)
period           = st.sidebar.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1)
sort_by          = st.sidebar.selectbox("Sort by", ["Final Rank", "UPI", "Sharpe Ratio"], index=0)
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
            raw = fetch_prices(tickers, benchmark, period=period, interval="1d")
        if raw.empty: st.stop()

        df = build_table_dataframe(raw, benchmark, universe_df)
        df = apply_sort(df, sort_by)

        st.subheader("Alpha Momentum 30")
        table_html, table_height = build_sortable_table_html(df)
        components.html(table_html, height=table_height, scrolling=True)

        st.caption(
            f"{len(df)} results • {indices_universe} • {benchmark_key} • 1d EOD • {period} "
            f"• default sort: {sort_by} • click any column header to re-sort"
        )

        # CSV export (keep numeric types & original column name)
        export_df = df.drop(columns=["Symbol"]).copy()
        for c in ("Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum", "UPI", "Sharpe"):
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce").round(2)
        for c in ("S.No", "Rank_6M", "Rank_3M", "Rank_1M", "Final_Rank", "Position"):
            export_df[c] = pd.to_numeric(export_df[c], errors="coerce").astype("Int64")

        st.download_button(
            "Export CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{indices_universe.replace(' ', '').lower()}_momentum.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(str(e))
