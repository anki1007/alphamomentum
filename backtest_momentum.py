# app.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ================= UI THEME =================
st.set_page_config(page_title="Alpha Momentum Screener + Backtest", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');
:root { --app-font: 'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; --bg: #0b0e13; --bg-2: #10141b; --border: #1f2732; --border-soft: #1a2230; --text: #e6eaee; --text-dim: #b3bdc7; --accent: #7a5cff; --accent-2: #2bb0ff; }
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--app-font) !important; }
.block-container { padding-top: 3.75rem; }
.hero-title { font-weight: 800; font-size: clamp(26px, 4.5vw, 40px); line-height: 1.05; margin: 18px 0 10px 0; background: linear-gradient(90deg, var(--accent-2), var(--accent) 60%); -webkit-background-clip: text; background-clip: text; color: transparent; letter-spacing: .2px; }
section[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-weight: 700; color: var(--text-dim) !important; }
.stButton button { background: linear-gradient(180deg, #1b2432, #131922); color: var(--text); border: 1px solid var(--border); border-radius: 10px; }
.stButton button:hover { filter: brightness(1.06); }
.pro-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 14px; padding: 6px 10px 10px 10px; box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
a { text-decoration: none; color: #9ecbff; } a:hover { text-decoration: underline; }
.metric-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }
.small { color: var(--text-dim); font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">Alpha Momentum Screener + Backtest</div>', unsafe_allow_html=True)

# ================= CONFIG =================
BENCHMARKS: Dict[str, str] = {"NIFTY 50": "^NSEI", "Nifty 200": "^CNX200", "Nifty 500": "^CRSLDX"}
GITHUB_BASE = "https://raw.githubusercontent.com/anki1007/alphamomentum/main/"
CSV_FILES: Dict[str, str] = {
    "Nifty 50": GITHUB_BASE + "nifty50.csv",
    "Nifty 100": GITHUB_BASE + "nifty100.csv",
    "Nifty 200": GITHUB_BASE + "nifty200.csv",
    "Nifty 500": GITHUB_BASE + "nifty500.csv",
    "Nifty Midcap 150": GITHUB_BASE + "niftymidcap150.csv",
    "Nifty Mid Small 400": GITHUB_BASE + "niftymidsmallcap400.csv",
    "Nifty Smallcap 250": GITHUB_BASE + "niftysmallcap250.csv",
    "Nifty Total Market": GITHUB_BASE + "niftytotalmarket.csv",
}
RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21
BATCH_SIZE = 60  # yfinance-friendly batching

# ================= SHARED HELPERS =================
def tv_symbol_from_yf(symbol: str) -> str:
    s = symbol.strip().upper()
    return "NSE:" + (s[:-3] if s.endswith(".NS") else s)

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
    rs = 100.0 * (df["p"] / df["b"])  # relative strength vs benchmark
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100.0 + (rs - m) / s).dropna()
    rroc = rs_ratio.pct_change().mul(100.0)
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom = (101.0 + (rroc - m2) / s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def analyze_momentum(adj: pd.Series) -> bool:
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
    return (adj.iloc[-1] >= ema100.iloc[-1] and one_year_return >= 6.5 and within_20pct_high and up_days_pct > 45.0)

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
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
    for c in ["Symbol", "Name", "Industry"]:
        df[c] = df[c].astype(str).str.strip()
    return df

def _period_years_to_dates(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    years_map = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}
    years = years_map.get(period, 2)
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    end = today_ist + pd.Timedelta(days=1)
    start = today_ist - pd.DateOffset(years=years)
    return start, end

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_prices_batched(tickers: List[str], benchmark: str, period: str, interval: str = "1d") -> pd.DataFrame:
    start, end = _period_years_to_dates(period)
    universe = list(dict.fromkeys(tickers))
    batches = [universe[i:i+BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]
    dfs = []
    for i, b in enumerate(batches, 1):
        try:
            dfb = yf.download(
                b + [benchmark],
                start=start.date().isoformat(),
                end=end.date().isoformat(),
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            dfs.append(dfb)
        except Exception as e:
            st.warning(f"Batch {i} failed ({len(b)} tickers): {e}")
    if not dfs:
        return pd.DataFrame()
    data = pd.concat(dfs, axis=1)
    if not isinstance(data.index, pd.DatetimeIndex):
        try: data.index = pd.to_datetime(data.index)
        except Exception: pass
    return data

# ================= Screener table =================
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
        if s.empty or (not analyze_momentum(s)): 
            continue
        s_rs = s.loc[s.index >= cutoff].copy()
        rr, mm = jdk_components(s_rs, bench_rs)
        if rr.empty or mm.empty:
            continue
        ix = rr.index.intersection(mm.index)
        rr_last = float(rr.loc[ix].iloc[-1]); mm_last = float(mm.loc[ix].iloc[-1])

        def ret_n(days: int) -> float | np.nan:
            if len(s) >= days:
                return float((s.iloc[-1] / s.iloc[-days] - 1) * 100)
            return np.nan

        rows.append({
            "Name": name,
            "Industry": industry,
            "Return_6M": ret_n(126), "Rank_6M": np.nan,
            "Return_3M": ret_n(63),  "Rank_3M": np.nan,
            "Return_1M": ret_n(21),  "Rank_1M": np.nan,
            "RS-Ratio": rr_last, "RS-Momentum": mm_last,
            "Performance": ("Leading" if rr_last>=100 and mm_last>=100 else "Improving" if rr_last<100 and mm_last>=100 else "Lagging" if rr_last<100 and mm_last<100 else "Weakening"),
            "Symbol": sym, "Chart": tradingview_chart_url(sym),
        })
    if not rows:
        raise RuntimeError("No tickers passed the filters. Try a longer Period (e.g., 3y) with 1d timeframe.")
    df = pd.DataFrame(rows)
    for c in ("Return_6M", "Return_3M", "Return_1M", "RS-Ratio", "RS-Momentum"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    def safe_rank(series: pd.Series) -> pd.Series:
        s = series.copy(); s.loc[~np.isfinite(s)] = np.nan
        return s.rank(ascending=False, method="min")

    df["Rank_6M"] = safe_rank(df["Return_6M"]).astype("Int64")
    df["Rank_3M"] = safe_rank(df["Return_3M"]).astype("Int64")
    df["Rank_1M"] = safe_rank(df["Return_1M"]).astype("Int64")
    df["Final_Rank"] = (df[["Rank_6M","Rank_3M","Rank_1M"]].fillna(df[["Rank_6M","Rank_3M","Rank_1M"]].max().max()+1).sum(axis=1))
    df = df.sort_values(["Final_Rank", "RS-Ratio", "RS-Momentum"], ascending=[True, False, False]).reset_index(drop=True)
    df.insert(0, "S.No", np.arange(1, len(df) + 1, dtype=int)); df["Position"] = df["S.No"].astype(int)
    order = ["S.No","Name","Industry","Return_6M","Rank_6M","Return_3M","Rank_3M","Return_1M","Rank_1M","RS-Ratio","RS-Momentum","Performance","Final_Rank","Position","Chart","Symbol"]
    return df[order]

def style_rows(df: pd.DataFrame):
    def _row_style(r: pd.Series):
        bg = row_bg_for_serial(int(r["S.No"]))
        return [f"background-color: {bg}"] * len(df.columns)
    styler = df.style.apply(lambda rr: _row_style(rr), axis=1)
    text_cols = ["Name", "Industry"]; center_cols = [c for c in df.columns if c not in text_cols]
    styler = styler.set_properties(subset=text_cols, **{"text-align": "left"})
    styler = styler.set_properties(subset=center_cols, **{"text-align": "center", "font-variant-numeric": "tabular-nums"})
    try: styler = styler.hide(axis="index")
    except Exception:
        try: styler = styler.hide_index()
        except Exception: pass
    return styler

# ================= Backtest core + metrics =================
def make_rebalance_dates(idx: pd.DatetimeIndex, code: str) -> List[pd.Timestamp]:
    idx = pd.DatetimeIndex(idx).sort_values().unique()
    if code.upper() == "W": key = pd.Grouper(freq="W-FRI")
    elif code.upper() == "F": key = pd.Grouper(freq="2W-FRI")
    elif code.upper() == "M": key = pd.Grouper(freq="M")
    elif code.upper() == "Q": key = pd.Grouper(freq="Q")
    else: raise ValueError("rebalance must be one of 'W','F','M','Q'")
    df = pd.DataFrame(index=idx, data={"flag": 1})
    return df.groupby(key).tail(1).index.tolist()

def rank_on_date(raw: pd.DataFrame, universe: List[str], benchmark: str, asof: pd.Timestamp) -> pd.DataFrame:
    bench = _pick_close(raw, benchmark); bench = bench[bench.index <= asof].dropna()
    if bench.empty:
        return pd.DataFrame(columns=["Symbol","RS_Ratio","RS_Momentum","Score"]).set_index("Symbol")
    cutoff = bench.index.max() - pd.Timedelta(days=RS_LOOKBACK_DAYS + 5); bench_rs = bench.loc[bench.index >= cutoff]
    rows = []
    for sym in universe:
        s = _pick_close(raw, sym); s = s[s.index <= asof].dropna()
        if s.empty or len(s) < RS_LOOKBACK_DAYS: 
            continue
        if not analyze_momentum(s): 
            continue
        s_rs = s.loc[s.index >= cutoff]
        rr, mm = jdk_components(s_rs, bench_rs)
        if rr.empty or mm.empty: 
            continue
        ix = rr.index.intersection(mm.index)
        rows.append((sym, float(rr.loc[ix].iloc[-1]), float(mm.loc[ix].iloc[-1])))
    if not rows:
        return pd.DataFrame(columns=["Symbol","RS_Ratio","RS_Momentum","Score"]).set_index("Symbol")
    df = pd.DataFrame(rows, columns=["Symbol","RS_Ratio","RS_Momentum"]).set_index("Symbol")
    df["rank_rr"] = df["RS_Ratio"].rank(ascending=False, method="min")
    df["rank_mm"] = df["RS_Momentum"].rank(ascending=False, method="min")
    df["Score"] = df["rank_rr"] + df["rank_mm"]
    return df.sort_values(["Score","RS_Ratio","RS_Momentum"], ascending=[True, False, False])

def backtest(
    raw_prices: pd.DataFrame,
    universe: List[str],
    benchmark: str,
    top_n: int = 30,
    rebalance: str = "M",
    start_date: Optional[pd.Timestamp] = None,
    init_capital: float = 1_000_000.0,
) -> Dict:
    bench = _pick_close(raw_prices, benchmark).dropna()
    if bench.empty: raise ValueError("Benchmark series is empty.")
    start = max(bench.index.min(), (start_date or bench.index.min())); end = bench.index.max()
    dates = bench.loc[(bench.index >= start) & (bench.index <= end)].index
    rbd = [d for d in make_rebalance_dates(dates, rebalance) if d >= start and d <= end]
    if not rbd: raise ValueError("No rebalance dates in the selected range.")
    price_map = {sym: _pick_close(raw_prices, sym).dropna() for sym in (set(universe)|{benchmark})}
    price_map = {k:v for k,v in price_map.items() if not v.empty}
    ret_map = {k: v.pct_change().fillna(0.0) for k, v in price_map.items()}

    port_ret = pd.Series(0.0, index=dates); holdings: Dict[pd.Timestamp, List[str]] = {}

    for i, d0 in enumerate(rbd):
        d1 = rbd[i + 1] if i + 1 < len(rbd) else dates[-1]
        ranks = rank_on_date(raw_prices, universe, benchmark, d0)
        basket = ranks.head(top_n).index.tolist(); holdings[d0] = basket
        if not basket: 
            continue
        w = 1.0 / len(basket)
        period_idx = dates[(dates >= d0) & (dates <= d1)]
        if len(period_idx) == 0: 
            continue
        daily_sum = pd.Series(0.0, index=period_idx); denom = pd.Series(0.0, index=period_idx)
        for sym in basket:
            r = ret_map.get(sym)
            if r is None: 
                continue
            rr = r.reindex(period_idx).dropna()
            daily_sum.loc[rr.index] = daily_sum.loc[rr.index] + w * rr
            denom.loc[rr.index] = denom.loc[rr.index] + w
        valid = denom > 0
        period_ret = pd.Series(0.0, index=period_idx)
        period_ret.loc[valid.index[valid]] = (daily_sum.loc[valid] / denom.loc[valid]).values
        port_ret.loc[period_idx] = period_ret

    nav = (1.0 + port_ret).cumprod() * (init_capital); nav.name = "NAV"
    bench_ret = ret_map[benchmark].reindex(nav.index).fillna(0.0)
    return {"nav": nav, "daily_returns": port_ret, "holdings": holdings, "benchmark_ret": bench_ret, "init_capital": init_capital}

# ----- Metrics -----
def _drawdown(nav: pd.Series) -> Tuple[pd.Series, float, float]:
    cummax = nav.cummax(); dd = nav / cummax - 1.0
    return dd, float(dd.iloc[-1]), float(dd.min())

def _cagr(nav: pd.Series) -> float:
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0: return 0.0
    years = days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)

def _xirr(dates: List[pd.Timestamp], values: List[float], guess: float = 0.1) -> float:
    def f(rate: float) -> float:
        return sum(v / ((1 + rate) ** ((d - dates[0]).days / 365.0)) for v, d in zip(values, dates))
    def df(rate: float) -> float:
        return sum(-((d - dates[0]).days / 365.0) * v / ((1 + rate) ** (((d - dates[0]).days / 365.0) + 1)) for v, d in zip(values, dates))
    r = guess
    for _ in range(100):
        y = f(r); dy = df(r)
        if abs(dy) < 1e-12: break
        r_new = r - y / dy
        if abs(r_new - r) < 1e-10: r = r_new; break
        r = r_new
    return float(r)

def _alpha_beta(port_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float]:
    x = bench_ret.values.reshape(-1); y = port_ret.values.reshape(-1)
    x = np.nan_to_num(x, nan=0.0); y = np.nan_to_num(y, nan=0.0)
    X = np.vstack([np.ones_like(x), x]).T
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = coeffs[0], coeffs[1]
    alpha_annual = (1 + alpha_daily) ** 252 - 1
    return float(alpha_annual), float(beta)

def _rolling_returns(nav: pd.Series, years: int) -> pd.Series:
    days = int(round(365.25 * years))
    if len(nav) <= days: return pd.Series(dtype=float)
    rr = nav.pct_change(periods=days)
    ann = (1 + rr) ** (365.25 / days) - 1
    return ann.dropna()

def build_metrics(bt: Dict) -> Dict:
    nav: pd.Series = bt["nav"].copy()
    port_ret: pd.Series = bt["daily_returns"].copy()
    bench_ret: pd.Series = bt["benchmark_ret"].copy()
    init_cap = float(bt["init_capital"])

    current_cap = float(nav.iloc[-1])
    nav_norm = nav / init_cap

    mu_d = float(port_ret.mean()); sig_d = float(port_ret.std(ddof=0))
    sharpe = (mu_d / sig_d * math.sqrt(252)) if sig_d > 0 else 0.0

    downside = port_ret.copy(); downside[downside > 0] = 0
    dd_sigma = float(downside.std(ddof=0))
    sortino = (mu_d / dd_sigma * math.sqrt(252)) if dd_sigma > 0 else 0.0

    dd_series, curr_dd, max_dd = _drawdown(nav_norm)
    calmar = (_cagr(nav_norm) / abs(max_dd)) if max_dd < 0 else 0.0

    gains = port_ret[port_ret > 0].sum(); losses = -port_ret[port_ret < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")
    win_rate = float((port_ret > 0).mean()); loss_rate = 1.0 - win_rate

    var95 = float(-np.percentile(port_ret.dropna(), 5)) if len(port_ret.dropna()) else 0.0
    mret = (nav_norm.resample("M").last().pct_change()).dropna()
    avg_monthly_ret = float(mret.mean()) if len(mret) else 0.0

    cagr = _cagr(nav_norm)
    xirr = _xirr([nav.index[0], nav.index[-1]], [-init_cap, current_cap])
    alpha, beta = _alpha_beta(port_ret.reindex_like(bench_ret), bench_ret)

    roll = {"1y": _rolling_returns(nav_norm, 1), "3y": _rolling_returns(nav_norm, 3), "5y": _rolling_returns(nav_norm, 5), "7y": _rolling_returns(nav_norm, 7)}

    summary = {
        "Capital Invested": init_cap,
        "Current Capital": current_cap,
        "NAV": float(nav_norm.iloc[-1]),
        "CAGR": cagr,
        "XIRR": xirr,
        "Profit Factor": profit_factor,
        "VaR(95%)": var95,
        "Win%": win_rate,
        "Loss%": loss_rate,
        "Average Monthly Return": avg_monthly_ret,
        "Current Drawdown": curr_dd,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Alpha": alpha,
        "Beta": beta,
    }
    return {"summary": summary, "nav": nav, "daily_returns": port_ret, "drawdown": dd_series, "rolling": roll}

# ================= SIDEBAR =================
st.sidebar.header("Controls")
indices_universe = st.sidebar.selectbox("Indices Universe", list(CSV_FILES.keys()), index=3)
benchmark_key    = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()), index=2)
period           = st.sidebar.selectbox("History Window", ["1y", "2y", "3y", "5y"], index=2)

st.sidebar.markdown("---")
top_n = st.sidebar.radio("Top-N Basket", [15, 30], index=1, horizontal=True)
rebalance = st.sidebar.selectbox("Rebalance", ["Weekly", "Fortnightly", "Monthly", "Quarterly"], index=2)
rb_map = {"Weekly": "W", "Fortnightly": "F", "Monthly": "M", "Quarterly": "Q"}

load_btn = st.sidebar.button("Run Screener + Backtest", use_container_width=True)
if "ran_once" not in st.session_state:
    st.session_state.ran_once = True
    load_btn = True

# ================= ACTION =================
if load_btn:
    try:
        # 1) Universe + Prices
        uni_url = CSV_FILES[indices_universe]
        universe_df = load_universe_from_csv(uni_url)
        benchmark = BENCHMARKS[benchmark_key]
        tickers = universe_df["Symbol"].tolist()

        with st.spinner("Fetching EOD prices…"):
            raw = fetch_prices_batched(tickers, benchmark, period=period, interval="1d")
        if raw.empty:
            st.stop()

        bench_close = _pick_close(raw, benchmark).dropna()
        last_dt = pd.to_datetime(bench_close.index.max()).strftime("%Y-%m-%d")
        st.caption(f"Data as of: {last_dt} (IST)")

        # 2) Screener
        df = build_table_dataframe(raw, benchmark, universe_df)
        ui_cols = ["S.No","Name","Industry","Return_6M","Rank_6M","Return_3M","Rank_3M","Return_1M","Rank_1M","RS-Ratio","RS-Momentum","Performance","Final_Rank","Position","Chart"]
        display_df = df[ui_cols].copy()
        display_df["Name"] = display_df.apply(lambda r: f'<a href="{r["Chart"]}" target="_blank" rel="noopener noreferrer">{r["Name"]}</a>', axis=1)
        display_df = display_df.drop(columns=["Chart"]).copy()
        int_cols = ["S.No","Rank_6M","Rank_3M","Rank_1M","Final_Rank","Position"]
        two_dec_cols = ["Return_6M","Return_3M","Return_1M","RS-Ratio","RS-Momentum"]
        for c in int_cols: display_df[c] = display_df[c].map(lambda v: "-" if pd.isna(v) else f"{int(v)}")
        for c in two_dec_cols: display_df[c] = display_df[c].map(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}")

        st.subheader("Alpha Momentum Table")
        table_html = style_rows(display_df.head(30)).to_html(escape=False)  # make links clickable
        st.markdown(f'<div class="pro-card">{table_html}</div>', unsafe_allow_html=True)
        st.caption(f"{len(df)} results • {indices_universe} • {benchmark_key} • 1d EOD • {period}")

        # 3) Backtest using Top-N & Rebalance
        st.subheader("Backtest")
        bt = backtest(
            raw_prices=raw,
            universe=universe_df["Symbol"].tolist(),
            benchmark=benchmark,
            top_n=int(top_n),
            rebalance=rb_map[rebalance],
            init_capital=1_000_000.0,
        )
        metrics = build_metrics(bt)

        nav = metrics["nav"]
        draw = metrics["drawdown"]
        daily = metrics["daily_returns"]
        roll = metrics["rolling"]
        summary = metrics["summary"]

        # Metric tiles
        cols = st.columns(4)
        as_pct = lambda x: f"{x*100:.2f}%"
        as_ratio = lambda x: f"{x:.2f}"
        as_money = lambda x: f"₹{x:,.0f}"
        with cols[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Capital Invested", as_money(summary["Capital Invested"]))
            st.metric("Current Capital", as_money(summary["Current Capital"]))
            st.metric("NAV (× initial)", f"{summary['NAV']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("CAGR", as_pct(summary["CAGR"]))
            st.metric("XIRR", as_pct(summary["XIRR"]))
            st.metric("Avg Monthly Return", as_pct(summary["Average Monthly Return"]))
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sharpe", as_ratio(summary["Sharpe Ratio"]))
            st.metric("Sortino", as_ratio(summary["Sortino Ratio"]))
            st.metric("Calmar", as_ratio(summary["Calmar Ratio"]))
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Max DD", as_pct(summary["Max Drawdown"]))
            st.metric("Current DD", as_pct(summary["Current Drawdown"]))
            st.metric("VaR (95%)", as_pct(summary["VaR(95%)"]))
            st.markdown('</div>', unsafe_allow_html=True)

        cols2 = st.columns(4)
        with cols2[0]: st.metric("Win%", as_pct(summary["Win%"]))
        with cols2[1]: st.metric("Loss%", as_pct(summary["Loss%"]))
        with cols2[2]: st.metric("Profit Factor", as_ratio(summary["Profit Factor"]))
        with cols2[3]: st.metric("Alpha / Beta", f"{as_pct(summary['Alpha'])} / {summary['Beta']:.2f}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**NAV** (₹)")
            st.line_chart(nav.rename("Portfolio NAV"))
        with c2:
            st.write("**Drawdown**")
            st.line_chart(draw.rename("Drawdown"))

        st.markdown("---")
        st.write("**Rolling Returns (annualized)**")
        rcols = st.columns(4)
        for i, key in enumerate(["1y","3y","5y","7y"]):
            with rcols[i]:
                rr = roll[key]
                if rr.empty:
                    st.write(key + ": not enough history")
                else:
                    st.line_chart(rr.rename(key + " Rolling"))
                    st.caption(f"Avg: {rr.mean()*100:.2f}%  •  Min: {rr.min()*100:.2f}%  •  Max: {rr.max()*100:.2f}%")

        with st.expander("Holdings by Rebalance Date"):
            hold_df = pd.Series({pd.to_datetime(k).strftime('%Y-%m-%d'): v for k, v in bt["holdings"].items()}).rename("Tickers").to_frame()
            st.dataframe(hold_df, use_container_width=True)

        st.markdown("---")
        ret_df = pd.DataFrame({"Date": nav.index, "NAV": nav.values, "DailyReturn": daily.values, "Drawdown": draw.reindex(nav.index).values})
        st.download_button(
            "Download NAV & Returns CSV",
            data=ret_df.to_csv(index=False).encode("utf-8"),
            file_name=f"backtest_top{top_n}_{rb_map[rebalance]}_{benchmark_key.replace(' ','')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(str(e))
