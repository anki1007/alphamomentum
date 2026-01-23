import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# CONFIG & CSS
# =========================================================
st.set_page_config(page_title="Indian Market Ratio Terminal", layout="wide")
RATIO_MULTIPLIER = 1000

# Custom CSS: Title, Tabs, Table Alignment and Sorting
st.markdown(
    """
    <style>
    /* 1. MASSIVE NEON TITLE - SINGLE LINE */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@900&display=swap');
    
    .neon-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 30px 0;
        width: 100%;
    }
    
    .neon-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 4vw !important;
        font-weight: 500;
        text-transform: uppercase;
        color: #fff;
        text-align: center;
        background: linear-gradient(90deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 242, 96, 0.4), 0 0 30px rgba(5, 117, 230, 0.4);
        margin: 0;
        line-height: 1.2;
        white-space: nowrap;
    }

    /* 2. NEON 3D TABS - CENTERED */
    div[data-baseweb="tab-list"] { 
        gap: 20px; 
        background: transparent; 
        padding: 20px 0;
        display: flex;
        justify-content: center;
    }
    
    button[data-baseweb="tab"] {
        background: linear-gradient(145deg, #1a1a1a, #222);
        color: #00f260;
        border-radius: 8px;
        padding: 12px 25px;
        font-size: 18px; 
        font-weight: 700;
        border: 1px solid #333;
        transition: all 0.2s ease;
        font-family: 'Orbitron', sans-serif;
    }

    button[data-baseweb="tab"]:hover {
        background: #2a2a2a;
        color: #fff;
        border-color: #00f260;
        box-shadow: 0 0 15px rgba(0, 242, 96, 0.4);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border: 1px solid #00f260;
        box-shadow: inset 0 0 10px rgba(0, 242, 96, 0.2);
    }
    
    /* 3. CUSTOM HTML TABLE STYLING */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 15px;
        background-color: #1e1e1e;
        color: #e0e0e0;
        margin: 20px 0;
    }
    
    .custom-table thead {
        background-color: #2a2a2a;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .custom-table th {
        padding: 12px 8px;
        text-align: center;
        font-weight: 700;
        border: 1px solid #333;
        font-size: 16px;
        cursor: pointer;
        user-select: none;
        position: relative;
    }
    
    .custom-table th:hover {
        background-color: #353535;
    }
    
    .custom-table th.sortable::after {
        content: ' ⇅';
        color: #666;
        font-size: 12px;
    }
    
    .custom-table th.sort-asc::after {
        content: ' ▲';
        color: #00f260;
    }
    
    .custom-table th.sort-desc::after {
        content: ' ▼';
        color: #00f260;
    }
    
    .custom-table td {
        padding: 10px 8px;
        text-align: center;
        border: 1px solid #333;
        font-size: 15px;
    }
    
    /* Left align for Symbol and Industry columns */
    .custom-table th.left-align,
    .custom-table td.left-align {
        text-align: left !important;
        padding-left: 20px !important;
    }
    
    .custom-table tbody tr:hover {
        background-color: #252525;
    }
    
    .table-container {
        max-height: 900px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #333;
        border-radius: 5px;
    }
    
    /* Scrollbar styling */
    .table-container::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    .table-container::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    .table-container::-webkit-scrollbar-thumb {
        background: #00f260;
        border-radius: 5px;
    }
    
    .table-container::-webkit-scrollbar-thumb:hover {
        background: #0575e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SYMBOL MASTER
# =========================================================
SYMBOL_MASTER = [
    {"name":"NIFTY 50","yahoo":"^NSEI","industry":"NIFTY"},
    {"name":"NIFTY NEXT 50","yahoo":"^NSMIDCP","industry":"NIFTY JR"},
    {"name":"NIFTY 100","yahoo":"^CNX100","industry":"LARGE CAP 100"},
    {"name":"NIFTY 200","yahoo":"^CNX200","industry":"LARGE MIDCAP 200"},
    {"name":"NIFTY LARGE MIDCAP 250","yahoo":"NIFTY_LARGEMID250.NS","industry":"LARGE MIDCAP 250"},
    {"name":"NIFTY MIDCAP 50","yahoo":"^NSEMDCP50","industry":"MIDCAP 50"},
    {"name":"NIFTY MIDCAP 100","yahoo":"NIFTY_MIDCAP_100.NS","industry":"MIDCAP 100"},
    {"name":"NIFTY MIDCAP 150","yahoo":"NIFTYMIDCAP150.NS","industry":"MIDCAP 150"},
    {"name":"NIFTY MID SMALLCAP 400","yahoo":"NIFTYMIDSML400.NS","industry":"MID SMALL CAP"},
    {"name":"NIFTY SMALLCAP 100","yahoo":"^CNXSC","industry":"SMALL CAP"},
    {"name":"NIFTY SMALLCAP 250","yahoo":"NIFTYSMLCAP250.NS","industry":"SMALL CAP"},
    {"name":"NIFTY MICROCAP 250","yahoo":"NIFTY_MICROCAP250.NS","industry":"MICRO CAP"},
    {"name":"NIFTY 500","yahoo":"^CRSLDX","industry":"BROAD MARKET"},
    {"name":"NIFTY AUTO","yahoo":"^CNXAUTO","industry":"AUTO"},
    {"name":"NIFTY COMMODITIES","yahoo":"^CNXCMDT","industry":"COMMODITIES"},
    {"name":"NIFTY CONSUMPTION","yahoo":"^CNXCONSUM","industry":"CONSUMPTION"},
    {"name":"NIFTY ENERGY","yahoo":"^CNXENERGY","industry":"ENERGY"},
    {"name":"NIFTY FMCG","yahoo":"^CNXFMCG","industry":"FMCG"},
    {"name":"NIFTY INFRA","yahoo":"^CNXINFRA","industry":"INFRA"},
    {"name":"NIFTY IT","yahoo":"^CNXIT","industry":"IT"},
    {"name":"NIFTY MEDIA","yahoo":"^CNXMEDIA","industry":"MEDIA"},
    {"name":"NIFTY METAL","yahoo":"^CNXMETAL","industry":"METAL"},
    {"name":"NIFTY MNC","yahoo":"^CNXMNC","industry":"MNC"},
    {"name":"NIFTY PHARMA","yahoo":"^CNXPHARMA","industry":"PHARMA"},
    {"name":"NIFTY PSE","yahoo":"^CNXPSE","industry":"PSE"},
    {"name":"NIFTY PSU BANK","yahoo":"^CNXPSUBANK","industry":"PSUBANK"},
    {"name":"NIFTY REALTY","yahoo":"^CNXREALTY","industry":"REALTY"},
    {"name":"NIFTY SERVICE","yahoo":"^CNXSERVICE","industry":"SERVICE"},
    {"name":"NIFTY BANK","yahoo":"^NSEBANK","industry":"BANKNIFTY"},
    {"name":"NIFTY CHEMICAL","yahoo":"NIFTY_CHEMICALS.NS","industry":"CHEMICAL"},
    {"name":"NIFTY CONSUMER DURABLES","yahoo":"NIFTY_CONSR_DURBL.NS","industry":"CONSUMER DURABLES"},
    {"name":"NIFTY CPSE","yahoo":"NIFTY_CPSE.NS","industry":"CPSE"},
    {"name":"NIFTY FINANCIAL SERVICE","yahoo":"NIFTY_FIN_SERVICE.NS","industry":"FINANCIAL SERVICE"},
    {"name":"NIFTY HEALTHCARE","yahoo":"NIFTY_HEALTHCARE.NS","industry":"HEALTHCARE"},
    {"name":"NIFTY INDIA DEFENCE","yahoo":"NIFTY_IND_DEFENCE.NS","industry":"DEFENCE"},
    {"name":"NIFTY DIGITAL","yahoo":"NIFTY_IND_DIGITAL.NS","industry":"DIGITAL"},
    {"name":"NIFTY OIL AND GAS","yahoo":"NIFTY_OIL_AND_GAS.NS","industry":"OIL AND GAS"},
    {"name":"NIFTY PRIVATE BANK","yahoo":"NIFTYPVTBANK.NS","industry":"PRIVATE BANK"},
]

ALL_NAMES = [s["name"] for s in SYMBOL_MASTER]
BENCHMARK_INDEX = "NIFTY 500"

# =========================================================
# DATA FETCH
# =========================================================
@st.cache_data
def fetch_all_prices():
    tickers = [s["yahoo"] for s in SYMBOL_MASTER]
    try:
        df = yf.download(tickers, start="2000-01-01", group_by='ticker', progress=False, auto_adjust=True, threads=True)
    except Exception as e:
        st.error(f"Critical Download Error: {e}")
        return {}
    
    out = {}
    for s in SYMBOL_MASTER:
        t = s["yahoo"]
        try:
            if t in df.columns.levels[0]:
                series = df[t]["Close"].dropna()
                if not series.empty:
                    out[s["name"]] = series
                else:
                    out[s["name"]] = None
            else:
                out[s["name"]] = None
        except Exception:
            out[s["name"]] = None
    return out

prices = fetch_all_prices()

# =========================================================
# UTILS & HELPERS
# =========================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def safe_ratio(a, b):
    if a is None or b is None or a.empty or b.empty: return None
    df = pd.concat([a, b], axis=1).dropna()
    if df.empty: return None
    return (df.iloc[:, 0] / df.iloc[:, 1]) * RATIO_MULTIPLIER

def rs_calc(r, n):
    return ((r / r.shift(n)) - 1) * 100

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# =========================================================
# HTML TABLE GENERATOR WITH SORTING
# =========================================================
def generate_sortable_table(df, left_align_cols=[], table_id="table"):
    """Generate sortable HTML table with proper alignment"""
    
    # Start HTML with complete styling
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {
        margin: 0;
        padding: 0;
        background-color: transparent;
        font-family: Arial, sans-serif;
    }
    
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 15px;
        background-color: #1e1e1e;
        color: #e0e0e0;
        margin: 0;
    }
    
    .custom-table thead {
        background-color: #2a2a2a;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .custom-table th {
        padding: 12px 8px;
        text-align: center;
        font-weight: 700;
        border: 1px solid #333;
        font-size: 16px;
        cursor: pointer;
        user-select: none;
        position: relative;
    }
    
    .custom-table th:hover {
        background-color: #353535;
    }
    
    .custom-table th.sortable::after {
        content: ' ⇅';
        color: #666;
        font-size: 12px;
    }
    
    .custom-table th.sort-asc::after {
        content: ' ▲';
        color: #00f260;
    }
    
    .custom-table th.sort-desc::after {
        content: ' ▼';
        color: #00f260;
    }
    
    .custom-table td {
        padding: 10px 8px;
        text-align: center;
        border: 1px solid #333;
        font-size: 15px;
    }
    
    .custom-table th.left-align,
    .custom-table td.left-align {
        text-align: left !important;
        padding-left: 20px !important;
    }
    
    .custom-table tbody tr:hover {
        background-color: #252525;
    }
    
    .table-container {
        max-height: 900px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #333;
        border-radius: 5px;
    }
    
    .table-container::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    .table-container::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    .table-container::-webkit-scrollbar-thumb {
        background: #00f260;
        border-radius: 5px;
    }
    
    .table-container::-webkit-scrollbar-thumb:hover {
        background: #0575e6;
    }
    </style>
    </head>
    <body>
    """
    
    html += f'<div class="table-container"><table class="custom-table" id="{table_id}"><thead><tr>'
    
    # Headers with sortable class
    for idx, col in enumerate(df.columns):
        align_class = 'left-align' if col in left_align_cols else ''
        html += f'<th class="sortable {align_class}" onclick="sortTable({idx})">{col}</th>'
    html += '</tr></thead><tbody>'
    
    # Rows
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            align_class = 'left-align' if col in left_align_cols else ''
            value = row[col]
            
            # Format and color
            if pd.isna(value) or value == '-':
                cell_content = '-'
                cell_value = '-999999999'  # Sort to end
                color = ''
            elif col in ['Trend']:
                if value == 'Bullish':
                    color = 'style="color: #00ff99; font-weight: bold;"'
                elif value == 'Bearish':
                    color = 'style="color: #ff4d4d; font-weight: bold;"'
                else:
                    color = ''
                cell_content = value
                cell_value = value
            elif col in ['Above 100', 'Above 200']:
                if value == 'Yes':
                    color = 'style="color: #00ff99;"'
                    cell_value = '1'  # For sorting
                elif value == 'No':
                    color = 'style="color: #ff4d4d;"'
                    cell_value = '0'  # For sorting
                else:
                    color = ''
                    cell_value = '-999999999'
                cell_content = value
            elif col in ['Status']:
                if value == 'STRONG BUY':
                    color = 'style="color: #00ff99; font-weight: bold;"'
                else:
                    color = ''
                cell_content = value
                cell_value = value
            elif col.startswith('RS '):
                try:
                    val = float(value)
                    color = f'style="color: {"#00ff99" if val > 0 else "#ff4d4d"};"'
                    cell_content = f'{val:.2f}'
                    cell_value = val
                except:
                    color = ''
                    cell_content = str(value)
                    cell_value = -999999999
            elif col.startswith('EMA ') or col.startswith('RSI '):
                try:
                    val = float(value)
                    ltp = row.get('LTP', 0)
                    if col.startswith('EMA '):
                        color = f'style="color: {"#00ff99" if ltp > val else "#ff4d4d"};"'
                    else:  # RSI
                        color = f'style="color: {"#00ff99" if val >= 50 else "#ff4d4d"};"'
                    cell_content = f'{val:.2f}'
                    cell_value = val
                except:
                    color = ''
                    cell_content = str(value)
                    cell_value = -999999999
            elif col in ['SL No.', 'No']:
                try:
                    cell_content = str(int(value))
                    cell_value = int(value)
                except:
                    cell_content = str(value)
                    cell_value = -999999999
                color = ''
            elif col in ['LTP', 'Ratio'] or isinstance(value, (int, float)):
                try:
                    val = float(value)
                    cell_content = f'{val:.2f}'
                    cell_value = val
                except:
                    cell_content = str(value)
                    cell_value = -999999999
                color = ''
            else:
                cell_content = str(value)
                cell_value = str(value)
                color = ''
            
            html += f'<td class="{align_class}" {color} data-value="{cell_value}">{cell_content}</td>'
        html += '</tr>'
    
    html += '</tbody></table></div>'
    
    # Add JavaScript for sorting
    html += f"""
    <script>
    function sortTable(columnIndex) {{
        const table = document.getElementById('{table_id}');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const header = table.querySelectorAll('th')[columnIndex];
        
        // Determine sort direction
        let isAscending = true;
        if (header.classList.contains('sort-asc')) {{
            isAscending = false;
        }}
        
        // Remove all sort classes
        table.querySelectorAll('th').forEach(th => {{
            th.classList.remove('sort-asc', 'sort-desc');
        }});
        
        // Add appropriate class
        if (isAscending) {{
            header.classList.add('sort-asc');
        }} else {{
            header.classList.add('sort-desc');
        }}
        
        // Sort rows
        rows.sort((a, b) => {{
            const aValue = a.querySelectorAll('td')[columnIndex].getAttribute('data-value');
            const bValue = b.querySelectorAll('td')[columnIndex].getAttribute('data-value');
            
            // Try numeric comparison first
            const aNum = parseFloat(aValue);
            const bNum = parseFloat(bValue);
            
            if (!isNaN(aNum) && !isNaN(bNum)) {{
                return isAscending ? aNum - bNum : bNum - aNum;
            }}
            
            // Fall back to string comparison
            if (isAscending) {{
                return aValue.localeCompare(bValue);
            }} else {{
                return bValue.localeCompare(aValue);
            }}
        }});
        
        // Reattach rows
        rows.forEach(row => tbody.appendChild(row));
    }}
    </script>
    </body>
    </html>
    """
    
    return html

# =========================================================
# PLOTTING
# =========================================================
def draw_colored_vectorized(fig, x, y, cond, width, row, name):
    y_green = y.copy()
    y_red = y.copy()
    y_green[~cond] = np.nan
    y_red[cond] = np.nan
    fig.add_trace(go.Scatter(x=x, y=y_green, mode="lines", line=dict(color="#00ff99", width=width), name=f"{name} (Bullish)", showlegend=False), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=y_red, mode="lines", line=dict(color="#ff4d4d", width=width), name=f"{name} (Bearish)", showlegend=False), row=row, col=1)

def plot_ratio(numerator, denominator, rs_periods):
    r = safe_ratio(prices[numerator], prices[denominator])
    if r is None:
        st.warning(f"Unable to calculate ratio: Data missing for {numerator} or {denominator}")
        return

    # Layout sizing
    if not rs_periods:
        rows = 1
        row_heights = [1.0]
        subplot_titles = ["Ratio Trend"]
        chart_height = 600
    else:
        rows = 1 + len(rs_periods)
        row_heights = [0.6] + [0.4/len(rs_periods)] * len(rs_periods)
        subplot_titles = ["Ratio Trend"] + [f"Relative Strength ({p})" for p in rs_periods]
        chart_height = 600 + (150 * len(rs_periods))

    fig = make_subplots(
        rows=rows, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08, 
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # Main Chart
    draw_colored_vectorized(fig, r.index, r.values, r.diff() >= 0, 3, 1, "Ratio")
    e200 = ema(r, 200)
    draw_colored_vectorized(fig, r.index, e200.values, r > e200, 3, 1, "EMA 200")

    # RS Charts
    rs_annotations = []
    if rs_periods:
        for idx, p in enumerate(rs_periods):
            row_idx = idx + 2
            rs_line = rs_calc(r, p)
            draw_colored_vectorized(fig, rs_line.index, rs_line.values, rs_line >= 0, 2, row_idx, f"RS {p}")
            fig.add_hline(y=0, row=row_idx, col=1, line=dict(color="gray", dash="dash", width=2))
            if not rs_line.empty:
                rs_annotations.append(f"RS {p}: {rs_line.iloc[-1]:.2f}")

    # Info Box Construction
    rs_text_block = "<br>" + "<br>".join(rs_annotations) if rs_annotations else ""
    
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=(f"<b>{numerator} / {denominator}</b> : {r.iloc[-1]:.2f}<br><b>EMA 200</b> : {e200.iloc[-1]:.2f}" + rs_text_block),
        showarrow=False, align="left", 
        font=dict(size=16, color="white", family="Arial Black"), 
        bgcolor="rgba(0,0,0,0.75)", bordercolor="#888", borderwidth=2
    )

    fig.update_layout(
        height=chart_height, 
        margin=dict(l=15, r=15, t=60, b=15), 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15, color="#e0e0e0") 
    )
    
    fig.update_xaxes(showgrid=False, tickfont=dict(size=13))
    fig.update_yaxes(showgrid=True, gridcolor="#333", tickfont=dict(size=13))
    fig.update_annotations(font=dict(size=18, color="#ffffff", weight="bold"))

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# UI
# =========================================================
st.markdown("### 🎯")
st.markdown('<div class="neon-container"><div class="neon-text">INDIAN MARKET RATIO TERMINAL</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Market Ratios", "🏭 Sector Ratios", "📊 Analytics Table", "📟 Technical Dashboard", "🔍 Opportunity Scanner"]
)

# -------- TAB 1: Market Ratios --------
with tab1:
    defaults = ["NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200"]
    cols = st.columns(2) + st.columns(2)
    for i, (sym, col) in enumerate(zip(defaults, cols)):
        with col:
            c1, c2, c3 = st.columns([1.5, 2, 2])
            with c1:
                rs_select = st.multiselect("RS Period", [252,126,63,21], [252], key=f"rs1{i}")
            with c2:
                num = st.selectbox("Numerator", ALL_NAMES, ALL_NAMES.index(sym), key=f"n1{i}")
            with c3:
                den = st.selectbox("Benchmark", ALL_NAMES, ALL_NAMES.index(BENCHMARK_INDEX), key=f"d1{i}")
            
            plot_ratio(num, den, rs_select)

# -------- TAB 2: Sector Ratios --------
with tab2:
    defaults = ["NIFTY BANK", "NIFTY AUTO", "NIFTY IT", "NIFTY PHARMA"]
    cols = st.columns(2) + st.columns(2)
    for i, (sym, col) in enumerate(zip(defaults, cols)):
        with col:
            c1, c2, c3 = st.columns([1.5, 2, 2])
            with c1:
                rs_select = st.multiselect("RS Period", [252,126,63,21], [252], key=f"rs2{i}")
            with c2:
                num = st.selectbox("Numerator", ALL_NAMES, ALL_NAMES.index(sym), key=f"n2{i}")
            with c3:
                den = st.selectbox("Benchmark", ALL_NAMES, ALL_NAMES.index(BENCHMARK_INDEX), key=f"d2{i}")
            
            plot_ratio(num, den, rs_select)

# -------- TAB 3: Analytics Table --------
with tab3:
    st.subheader("Relative Strength Analytics")
    base_name = st.selectbox("Select Base Index", ALL_NAMES, index=ALL_NAMES.index(BENCHMARK_INDEX))
    base_series = prices.get(base_name)
    
    rows = []
    if base_series is not None and not base_series.empty:
        for i, s in enumerate(SYMBOL_MASTER, start=1):
            ps = prices.get(s["name"])
            
            if ps is None or ps.empty:
                rows.append({
                    "SL No.": i, "Symbol": s["name"], "Industry": s["industry"],
                    "Status": "DATA MISSING", 
                    "Ratio": "-", 
                    "Above 100": "-", 
                    "Above 200": "-", 
                    "Trend": "-",
                    "RS 21": "-", 
                    "RS 63": "-", 
                    "RS 126": "-", 
                    "RS 252": "-"
                })
                continue
            
            r = safe_ratio(ps, base_series)
            if r is None or r.empty: 
                continue
            
            ema100 = ema(r, 100)
            ema200 = ema(r, 200)

            rows.append({
                "SL No.": i,
                "Symbol": s["name"],
                "Industry": s["industry"],
                "Status": "OK",
                "Ratio": r.iloc[-1],
                "Above 100": "Yes" if r.iloc[-1] > ema100.iloc[-1] else "No",
                "Above 200": "Yes" if r.iloc[-1] > ema200.iloc[-1] else "No",
                "Trend": "Bullish" if r.iloc[-1] > ema200.iloc[-1] else "Bearish",
                "RS 21": rs_calc(r, 21).iloc[-1],
                "RS 63": rs_calc(r, 63).iloc[-1],
                "RS 126": rs_calc(r, 126).iloc[-1],
                "RS 252": rs_calc(r, 252).iloc[-1],
            })

    df_an = pd.DataFrame(rows)
    
    # Auto-sort by RS 252 (high to low)
    if not df_an.empty:
        # Convert RS 252 to numeric, handling '-' values
        df_an['RS 252'] = pd.to_numeric(df_an['RS 252'], errors='coerce')
        df_an = df_an.sort_values('RS 252', ascending=False, na_position='last')
        
        html_table = generate_sortable_table(df_an, left_align_cols=['Symbol', 'Industry'], table_id="analytics_table")
        components.html(html_table, height=920, scrolling=True)
    else:
        st.warning("No data available.")

# -------- TAB 4: Technical Dashboard --------
with tab4:
    ema_periods = [5, 9, 21, 30, 52, 75, 88, 125, 137, 208, 252]
    rsi_periods = [5, 9, 14, 21, 30, 52, 75, 88, 125, 137, 208, 252]

    rows = []
    for i, s in enumerate(SYMBOL_MASTER, start=1):
        series = prices.get(s["name"])
        
        if series is None or series.empty:
            row = { "SL No.": i, "Name": s["name"], "Industry": s["industry"], "LTP": "-" }
            for p in ema_periods: row[f"EMA {p}"] = "-"
            for p in rsi_periods: row[f"RSI {p}"] = "-"
            rows.append(row)
            continue
        
        ltp = float(series.iloc[-1])
        row = { "SL No.": i, "Name": s["name"], "Industry": s["industry"], "LTP": ltp }

        for p in ema_periods:
            ev = float(ema(series, p).iloc[-1])
            row[f"EMA {p}"] = ev

        for p in rsi_periods:
            rv = float(rsi(series, p).iloc[-1])
            row[f"RSI {p}"] = rv

        rows.append(row)

    df_tech = pd.DataFrame(rows)

    if not df_tech.empty:
        # Calculate sort score: how many EMAs are below LTP + sum of RSI values
        def calc_score(row):
            if row['LTP'] == '-':
                return -999999
            
            ltp = row['LTP']
            ema_score = 0
            rsi_score = 0
            
            # Count EMAs below LTP
            for p in ema_periods:
                ema_val = row.get(f'EMA {p}', '-')
                if ema_val != '-' and ltp > ema_val:
                    ema_score += 1
            
            # Sum RSI values
            for p in rsi_periods:
                rsi_val = row.get(f'RSI {p}', '-')
                if rsi_val != '-':
                    rsi_score += rsi_val
            
            # Combined score: EMA count * 10000 + RSI sum
            return ema_score * 10000 + rsi_score
        
        df_tech['_sort_score'] = df_tech.apply(calc_score, axis=1)
        df_tech = df_tech.sort_values('_sort_score', ascending=False)
        df_tech = df_tech.drop('_sort_score', axis=1)
        
        html_table = generate_sortable_table(df_tech, left_align_cols=['Name', 'Industry'], table_id="tech_table")
        components.html(html_table, height=920, scrolling=True)

# -------- TAB 5: Opportunity Scanner --------
with tab5:
    st.markdown("### 🔍 Opportunity Scanner")
    st.markdown("Find indices where **LTP > EMA(200, 100, 50, 21)** AND **RS Ratio(252, 126, 63, 21) > 0**")
    
    col_scan, col_base = st.columns([1, 2])
    
    with col_scan:
        st.write("")
        st.write("") 
        scan_btn = st.button("RUN HIGHLIGHT SCAN", type="primary", use_container_width=True)
    with col_base:
        scan_base = st.selectbox("Benchmark for RS Check", ALL_NAMES, index=ALL_NAMES.index(BENCHMARK_INDEX), key="scan_base")

    if scan_btn:
        scan_results = []
        base_s = prices.get(scan_base)
        
        progress_bar = st.progress(0)
        
        for i, s in enumerate(SYMBOL_MASTER):
            progress_bar.progress((i + 1) / len(SYMBOL_MASTER))
            p_series = prices.get(s["name"])
            
            if p_series is None or p_series.empty or base_s is None or base_s.empty: continue
            
            ltp = p_series.iloc[-1]
            e21 = ema(p_series, 21).iloc[-1]
            e50 = ema(p_series, 50).iloc[-1]
            e100 = ema(p_series, 100).iloc[-1]
            e200 = ema(p_series, 200).iloc[-1]
            
            price_cond = (ltp > e200) and (ltp > e100) and (ltp > e50) and (ltp > e21)
            
            if price_cond:
                rat = safe_ratio(p_series, base_s)
                if rat is not None and not rat.empty:
                    rs21 = rs_calc(rat, 21).iloc[-1]
                    rs63 = rs_calc(rat, 63).iloc[-1]
                    rs126 = rs_calc(rat, 126).iloc[-1]
                    rs252 = rs_calc(rat, 252).iloc[-1]
                    
                    rs_cond = (rs21 > 0) and (rs63 > 0) and (rs126 > 0) and (rs252 > 0)
                    
                    if rs_cond:
                        scan_results.append({
                            "Symbol": s["name"],
                            "Industry": s["industry"],
                            "LTP": ltp,
                            "Status": "STRONG BUY",
                            "RS 21": rs21,
                            "RS 63": rs63,
                            "RS 126": rs126,
                            "RS 252": rs252
                        })

        progress_bar.empty()
        
        if not scan_results:
            st.warning("No opportunities found matching strict criteria.")
        else:
            df_scan = pd.DataFrame(scan_results)
            
            # Auto-sort by RS 252 (high to low)
            df_scan = df_scan.sort_values('RS 252', ascending=False)
            
            st.success(f"Found {len(df_scan)} Opportunities!")
            
            html_table = generate_sortable_table(df_scan, left_align_cols=['Symbol', 'Industry'], table_id="scan_table")
            components.html(html_table, height=1080, scrolling=True)
