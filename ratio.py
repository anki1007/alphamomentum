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
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@900&display=swap');
    
    /* 1. MASSIVE NEON TITLE */
    @keyframes pulse-title {
        30% { text-shadow: 0 0 10px rgba(0, 255, 255, 0.5), 0 0 20px rgba(255, 0, 255, 0.5); }
        30% { text-shadow: 0 0 30px rgba(0, 255, 255, 0.8), 0 0 50px rgba(255, 0, 255, 0.8), 0 0 70px rgba(255, 255, 255, 0.5); }
        30% { text-shadow: 0 0 10px rgba(0, 255, 255, 0.5), 0 0 20px rgba(255, 0, 255, 0.5); }
    }

    .neon-container {
        display: flex;
        justify-content: flex-start; /* LEFT ALIGNMENT FOR MAIN TITLE */
        align-items: center;
        padding: 40px 20px;
        width: 100%;
        background: radial-gradient(circle at top left, rgba(20,20,20,0.8) 0%, rgba(0,0,0,0) 70%);
        gap: 20px;
    }
    
    .neon-text {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5vw !important;
        font-weight: 900;
        text-transform: uppercase;
        color: #fff;
        text-align: left;
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse-title 3s infinite;
        margin: 0;
        line-height: 1.2;
        letter-spacing: 2px;
    }

    /* NEON BOLT STYLE (ELECTRIC BLUE) */
    .neon-bolt {
        font-size: 3.5vw;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #0099ff;
        animation: pulse-bolt 1.5s infinite alternate;
    }
    
    @keyframes pulse-bolt {
        from { text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff; opacity: 0.8; }
        to { text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff, 0 0 60px #fff; opacity: 1; }
    }

    /* 2. NEON 3D TABS */
    div[data-baseweb="tab-list"] { 
        gap: 25px; 
        background: transparent; 
        padding: 30px 10px;
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    button[data-baseweb="tab"] {
        background: #121212;
        border-radius: 12px;
        padding: 15px 30px;
        font-size: 22px !important; 
        font-weight: 800;
        border: 2px solid #333;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        position: relative;
        overflow: hidden;
    }

    button[data-baseweb="tab"]:hover {
        transform: translateY(-3px) scale(1.02);
        background: #1e1e1e;
    }
    
    /* --- EMOJI RESET --- */
    .emoji-style, 
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p::before {
        font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
        font-weight: normal;
        margin-right: 10px;
        display: inline-block;
        text-shadow: none !important; 
        -webkit-text-fill-color: white !important;
        color: white !important;
        filter: none !important;
        opacity: 1 !important;
    }
    
    .header-dart {
        font-size: 3.0vw;
        -webkit-text-fill-color: initial !important;
        text-shadow: none !important;
        filter: none !important;
    }

    /* Tab 1: Market Ratio (Lime) */
    button[data-baseweb="tab"]:nth-child(1) { color: #39ff14; border-bottom: 4px solid #39ff14; }
    button[data-baseweb="tab"]:nth-child(1):hover { box-shadow: 0 0 20px rgba(57, 255, 20, 0.4); text-shadow: 0 0 8px #39ff14; }
    button[data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(57, 255, 20, 0.1), transparent);
        border-color: #39ff14;
        box-shadow: 0 0 25px rgba(57, 255, 20, 0.6);
        color: #fff;
        text-shadow: 0 0 10px #39ff14;
    }
    button[data-baseweb="tab"]:nth-child(1) div[data-testid="stMarkdownContainer"] p::before { content: "📈"; }

    /* Tab 2: Sector Ratio (Orange) */
    button[data-baseweb="tab"]:nth-child(2) { color: #ff6600; border-bottom: 4px solid #ff6600; }
    button[data-baseweb="tab"]:nth-child(2):hover { box-shadow: 0 0 20px rgba(255, 102, 0, 0.4); text-shadow: 0 0 8px #ff6600; }
    button[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(255, 102, 0, 0.1), transparent);
        border-color: #ff6600;
        box-shadow: 0 0 25px rgba(255, 102, 0, 0.6);
        color: #fff;
        text-shadow: 0 0 10px #ff6600;
    }
    button[data-baseweb="tab"]:nth-child(2) div[data-testid="stMarkdownContainer"] p::before { content: "🏭"; }

    /* Tab 3: Analytics (Purple) */
    button[data-baseweb="tab"]:nth-child(3) { color: #bc13fe; border-bottom: 4px solid #bc13fe; }
    button[data-baseweb="tab"]:nth-child(3):hover { box-shadow: 0 0 20px rgba(188, 19, 254, 0.4); text-shadow: 0 0 8px #bc13fe; }
    button[data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(188, 19, 254, 0.1), transparent);
        border-color: #bc13fe;
        box-shadow: 0 0 25px rgba(188, 19, 254, 0.6);
        color: #fff;
        text-shadow: 0 0 10px #bc13fe;
    }
    button[data-baseweb="tab"]:nth-child(3) div[data-testid="stMarkdownContainer"] p::before { content: "📊"; }

    /* Tab 4: Technical (Cyan) */
    button[data-baseweb="tab"]:nth-child(4) { color: #00ffff; border-bottom: 4px solid #00ffff; }
    button[data-baseweb="tab"]:nth-child(4):hover { box-shadow: 0 0 20px rgba(0, 255, 255, 0.4); text-shadow: 0 0 8px #00ffff; }
    button[data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(0, 255, 255, 0.1), transparent);
        border-color: #00ffff;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
        color: #fff;
        text-shadow: 0 0 10px #00ffff;
    }
    button[data-baseweb="tab"]:nth-child(4) div[data-testid="stMarkdownContainer"] p::before { content: "📟"; }

    /* Tab 5: Scanner (Pink) */
    button[data-baseweb="tab"]:nth-child(5) { color: #ff1493; border-bottom: 4px solid #ff1493; }
    button[data-baseweb="tab"]:nth-child(5):hover { box-shadow: 0 0 20px rgba(255, 20, 147, 0.4); text-shadow: 0 0 8px #ff1493; }
    button[data-baseweb="tab"]:nth-child(5)[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(255, 20, 147, 0.1), transparent);
        border-color: #ff1493;
        box-shadow: 0 0 25px rgba(255, 20, 147, 0.6);
        color: #fff;
        text-shadow: 0 0 10px #ff1493;
    }
    button[data-baseweb="tab"]:nth-child(5) div[data-testid="stMarkdownContainer"] p::before { content: "🔍"; }
    
    /* 3. NEON SECTION TEXT STYLES - CENTERED */
    .neon-header-wrapper {
        display: flex;
        justify-content: center; /* CENTER ALIGNMENT */
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        font-weight: 700;
        text-align: center;
    }
    
    .border-gold { border-bottom: 1px solid rgba(255, 215, 0, 0.3); }
    .border-blue { border-bottom: 1px solid rgba(0, 255, 255, 0.3); }
    .border-lime { border-bottom: 1px solid rgba(57, 255, 20, 0.3); }

    .neon-text-gold {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
    }

    .neon-text-blue {
        background: linear-gradient(90deg, #00ffff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    .neon-text-lime {
        background: linear-gradient(90deg, #39ff14, #32cd32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 15px rgba(57, 255, 20, 0.5);
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
@st.cache_data(ttl=3600)  # Cache for 1 hour, then refresh
def fetch_all_prices():
    tickers = [s["yahoo"] for s in SYMBOL_MASTER]
    # Get today's date for fetching latest data
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    try:
        df = yf.download(tickers, start="2000-01-01", end=end_date, group_by='ticker', progress=False, auto_adjust=True, threads=True)
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
    # Use position to calculate ratio to safely handle duplicates
    df = pd.concat([a, b], axis=1).dropna()
    if df.empty: return None
    return (df.iloc[:, 0] / df.iloc[:, 1]) * RATIO_MULTIPLIER

def rs_calc(r, n):
    return ((r / r.shift(n)) - 1) * 100

def rsi(series, period=14):
    # Safe check for flat line (Numer == Denom)
    if series.nunique() <= 1:
        return pd.Series(50, index=series.index)
        
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid zero division
    rs = avg_gain / avg_loss.replace(0, np.nan)
    res = 100 - (100 / (1 + rs))
    return res.fillna(100) # If avg_loss was 0 (pure uptrend), RSI is 100

# =========================================================
# HTML TABLE GENERATOR WITH SORTING
# =========================================================
def generate_sortable_table(df, left_align_cols=[], table_id="table"):
    """Generate sortable HTML table with proper alignment"""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
    
    body {
        margin: 0;
        padding: 0;
        background-color: transparent;
        font-family: Arial, sans-serif;
    }
    
    .custom-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 15px;
        background-color: #121212;
        color: #e0e0e0;
        margin: 0;
        border: 1px solid #333;
    }
    
    .custom-table thead {
        background-color: #1a1a1a;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    /* MODIFIED TABLE HEADERS - MAGENTA/PURPLE GRADIENT */
    .custom-table th {
        padding: 15px 10px;
        text-align: center;
        border-bottom: 2px solid #ff00ff;
        border-right: 1px solid #333;
        font-size: 14px;
        cursor: pointer;
        user-select: none;
        position: relative;
        font-family: 'Orbitron', sans-serif;
        
        /* Neon Magenta/Purple Gradient Text */
        background: linear-gradient(90deg, #ff00ff, #8b00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 10px rgba(255, 0, 255, 0.3);
        
        transition: all 0.2s ease;
    }
    
    .custom-table th:hover {
        background-color: #252525;
        text-shadow: 0 0 14px rgba(255, 0, 255, 0.7);
        transform: scale(1.02);
    }
    
    .custom-table th.sortable::after {
        content: ' ⇅';
        -webkit-text-fill-color: #ffcc00; /* Yellow */
        font-size: 12px;
        padding-left: 5px;
    }
    
    .custom-table th.sort-asc::after {
        content: ' ▲';
        -webkit-text-fill-color: #00f260;
    }
    
    .custom-table th.sort-desc::after {
        content: ' ▼';
        -webkit-text-fill-color: #ff4d4d;
    }
    
    .custom-table td {
        padding: 12px 8px;
        text-align: center;
        border-bottom: 1px solid #333;
        border-right: 1px solid #333;
        font-size: 10px;
    }
    
    .custom-table th.left-align,
    .custom-table td.left-align {
        text-align: left !important;
        padding-left: 20px !important;
    }
    
    .custom-table tbody tr:hover {
        background-color: #1f1f1f;
        box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.05);
    }
    
    .table-container {
        max-height: 900px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #333;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    
    /* Scrollbar styling */
    .table-container::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    .table-container::-webkit-scrollbar-track {
        background: #121212;
    }
    
    .table-container::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ff00ff, #8b00ff);
        border-radius: 6px;
        border: 2px solid #121212;
    }
    
    .table-container::-webkit-scrollbar-thumb:hover {
        background: #ff00ff;
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
    # Convert inputs to numpy arrays for consistent handling
    if hasattr(x, 'values'):
        x_arr = x.values
    else:
        x_arr = np.array(x)
    
    if hasattr(cond, 'values'):
        cond_arr = cond.values
    else:
        cond_arr = np.array(cond)
    
    y_arr = np.array(y)
    
    # Draw line segment by segment
    # For each point i, we draw a line from point i-1 to point i
    # The color is determined by cond[i] (whether point i is above or below previous point)
    for i in range(1, len(y_arr)):
        # Get the two points for this line segment
        x_segment = x_arr[i-1:i+1]
        y_segment = y_arr[i-1:i+1]
        
        # Color based on whether current point is >= previous point
        if cond_arr[i]:
            color = "#00ff99"  # Green - going up
        else:
            color = "#ff4d4d"  # Red - going down
        
        # Add this line segment
        fig.add_trace(
            go.Scatter(
                x=x_segment, 
                y=y_segment, 
                mode="lines", 
                line=dict(color=color, width=width), 
                name=name, 
                showlegend=False,
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Value:</b> %{y:.2f}<extra></extra>'
            ), 
            row=row, 
            col=1
        )

def plot_ratio(numerator, denominator, rs_periods, chart_key):
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
    # Simple logic: Green when ratio >= previous day, Red when ratio < previous day
    # Handle NaN from shift by filling with False for first value
    prev_day_cond = (r >= r.shift(1)).fillna(False)
    draw_colored_vectorized(fig, r.index, r.values, prev_day_cond, 3, 1, "Ratio")
    
    # EMA 200 - Green when ratio > EMA 200, Red when ratio < EMA 200
    # Draw as smooth continuous line by grouping same-color segments
    e200 = ema(r, 200)
    e200_cond = r > e200  # Green when ratio above EMA 200
    
    # Convert to arrays for easier handling
    e200_cond_arr = e200_cond.values
    x_arr = r.index.values if hasattr(r.index, 'values') else np.array(r.index)
    y_arr = e200.values
    
    # Group consecutive points with same color
    i = 0
    while i < len(e200_cond_arr):
        current_color = e200_cond_arr[i]
        start_idx = i
        
        # Find where color changes
        while i < len(e200_cond_arr) and e200_cond_arr[i] == current_color:
            i += 1
        
        end_idx = i
        
        # Include one extra point for smooth connection
        if end_idx < len(e200_cond_arr):
            x_segment = x_arr[start_idx:end_idx + 1]
            y_segment = y_arr[start_idx:end_idx + 1]
        else:
            x_segment = x_arr[start_idx:end_idx]
            y_segment = y_arr[start_idx:end_idx]
        
        color = "#00ff99" if current_color else "#ff4d4d"
        
        fig.add_trace(
            go.Scatter(
                x=x_segment,
                y=y_segment,
                mode="lines",
                line=dict(color=color, width=2),
                name="EMA 200",
                showlegend=False,
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>EMA 200:</b> %{y:.2f}<extra></extra>'
            ),
            row=1,
            col=1
        )
    
    # EMA 100 - Neon Yellow when ratio > EMA 100, Red when ratio < EMA 100
    e100 = ema(r, 100)
    e100_cond = r > e100
    
    # Convert to arrays
    e100_cond_arr = e100_cond.values
    y100_arr = e100.values
    
    # Group consecutive points with same color
    i = 0
    while i < len(e100_cond_arr):
        current_color = e100_cond_arr[i]
        start_idx = i
        
        # Find where color changes
        while i < len(e100_cond_arr) and e100_cond_arr[i] == current_color:
            i += 1
        
        end_idx = i
        
        # Include one extra point for smooth connection
        if end_idx < len(e100_cond_arr):
            x_segment = x_arr[start_idx:end_idx + 1]
            y_segment = y100_arr[start_idx:end_idx + 1]
        else:
            x_segment = x_arr[start_idx:end_idx]
            y_segment = y100_arr[start_idx:end_idx]
        
        color = "#ffff00" if current_color else "#ff4d4d"
        
        fig.add_trace(
            go.Scatter(
                x=x_segment,
                y=y_segment,
                mode="lines",
                line=dict(color=color, width=2),
                name="EMA 100",
                showlegend=False,
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>EMA 100:</b> %{y:.2f}<extra></extra>'
            ),
            row=1,
            col=1
        )

    # RS Charts
    rs_annotations = []
    if rs_periods:
        for idx, p in enumerate(rs_periods):
            row_idx = idx + 2
            rs_line = rs_calc(r, p)
            draw_colored_vectorized(fig, rs_line.index, rs_line.values, rs_line >= 0, 2, row_idx, f"RS {p}")
            fig.add_hline(y=0, row=row_idx, col=1, line=dict(color="gray", dash="dash", width=2))
            
            # Auto-adjust y-axis range for this RS chart
            if not rs_line.empty:
                rs_min = rs_line.min()
                rs_max = rs_line.max()
                # Add 10% padding to top and bottom
                padding = (rs_max - rs_min) * 0.1
                fig.update_yaxes(
                    range=[rs_min - padding, rs_max + padding],
                    row=row_idx, 
                    col=1
                )
                rs_annotations.append(f"RS {p}: {rs_line.iloc[-1]:.2f}")

    # Info Box Construction
    rs_text_block = "<br>" + "<br>".join(rs_annotations) if rs_annotations else ""
    
    fig.add_annotation(
        xref="paper", yref="paper",  # Use paper coordinates so it stays fixed when zooming
        x=0.01, y=0.98,  # Position at top-left corner (1% from left, 98% from bottom)
        text=(f"<b>{numerator} / {denominator}</b> : {r.iloc[-1]:.2f}<br><b>EMA 100</b> : {e100.iloc[-1]:.2f}<br><b>EMA 200</b> : {e200.iloc[-1]:.2f}" + rs_text_block),
        showarrow=False, align="left", 
        font=dict(size=16, color="white", family="Arial Black"), 
        bgcolor="rgba(0,0,0,0.75)", bordercolor="#888", borderwidth=2,
        xanchor="left", yanchor="top"
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

    st.plotly_chart(fig, use_container_width=True, key=chart_key)

# =========================================================
# UI
# =========================================================
st.markdown("### ")
# UPDATED HEADER: Left Aligned, Dart + Electric Bolts
st.markdown(
    """
    <div class="neon-container">
        <span class="header-dart emoji-style">🎯</span>
        <span class="neon-bolt">⚡</span>
        <div class="neon-text">INDIAN MARKET RATIO TERMINAL</div>
        <span class="neon-bolt">⚡</span>
    </div>
    """, 
    unsafe_allow_html=True
)

# NOTE: Emojis are removed here to prevent tab styling interference. 
# They are re-injected via CSS.
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Market Ratios", "Sector Ratios", "Analytics Table", "Technical Dashboard", "Opportunity Scanner"]
)

# -------- TAB 1: Market Ratios (Lime) --------
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
            
            # Pass unique key to resolve Duplicate Element ID error
            plot_ratio(num, den, rs_select, chart_key=f"chart_mk_ratio_{i}")

# -------- TAB 2: Sector Ratios (Orange) --------
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
            
            # Pass unique key to resolve Duplicate Element ID error
            plot_ratio(num, den, rs_select, chart_key=f"chart_sec_ratio_{i}")

# -------- TAB 3: Analytics Table (Purple) --------
with tab3:
    # Emoji separate from neon text
    st.markdown('<div class="neon-header-wrapper border-gold"><span class="emoji-style">📊</span> <span class="neon-text-gold">Relative Strength Analytics</span></div>', unsafe_allow_html=True)
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
                rows.append({
                    "SL No.": i,
                    "Symbol": s["name"],
                    "Industry": s["industry"],
                    "Status": "NO RATIO DATA",
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
            
            # Calculate EMAs with available data
            try:
                ema100 = ema(r, 100)
                ema200 = ema(r, 200)
                ratio_val = r.iloc[-1]
                
                # Check if we have enough data for EMA comparisons
                if len(r) >= 200 and not ema200.empty:
                    above_100 = "Yes" if ratio_val > ema100.iloc[-1] else "No"
                    above_200 = "Yes" if ratio_val > ema200.iloc[-1] else "No"
                    trend = "Bullish" if ratio_val > ema200.iloc[-1] else "Bearish"
                elif len(r) >= 100 and not ema100.empty:
                    above_100 = "Yes" if ratio_val > ema100.iloc[-1] else "No"
                    above_200 = "-"
                    trend = "-"
                else:
                    above_100 = "-"
                    above_200 = "-"
                    trend = "-"
            except:
                ratio_val = r.iloc[-1]
                above_100 = "-"
                above_200 = "-"
                trend = "-"
            
            # Calculate RS values with validation (only if enough data)
            try:
                rs21_val = rs_calc(r, 21).iloc[-1] if len(r) > 21 else "-"
                if rs21_val != "-" and pd.isna(rs21_val):
                    rs21_val = "-"
            except:
                rs21_val = "-"
                
            try:
                rs63_val = rs_calc(r, 63).iloc[-1] if len(r) > 63 else "-"
                if rs63_val != "-" and pd.isna(rs63_val):
                    rs63_val = "-"
            except:
                rs63_val = "-"
                
            try:
                rs126_val = rs_calc(r, 126).iloc[-1] if len(r) > 126 else "-"
                if rs126_val != "-" and pd.isna(rs126_val):
                    rs126_val = "-"
            except:
                rs126_val = "-"
                
            try:
                rs252_val = rs_calc(r, 252).iloc[-1] if len(r) > 252 else "-"
                if rs252_val != "-" and pd.isna(rs252_val):
                    rs252_val = "-"
            except:
                rs252_val = "-"

            rows.append({
                "SL No.": i,
                "Symbol": s["name"],
                "Industry": s["industry"],
                "Status": "OK",
                "Ratio": ratio_val,
                "Above 100": above_100,
                "Above 200": above_200,
                "Trend": trend,
                "RS 21": rs21_val,
                "RS 63": rs63_val,
                "RS 126": rs126_val,
                "RS 252": rs252_val,
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

# -------- TAB 4: Technical Dashboard (Cyan) --------
with tab4:
    # Emoji separate from neon text
    st.markdown('<div class="neon-header-wrapper border-blue"><span class="emoji-style">📟</span> <span class="neon-text-blue">Multi-Timeframe Technical Dashboard</span></div>', unsafe_allow_html=True)
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
        
        # Only need at least 5 data points to calculate something
        if len(series) < 5:
            row = { "SL No.": i, "Name": s["name"], "Industry": s["industry"], "LTP": "-" }
            for p in ema_periods: row[f"EMA {p}"] = "-"
            for p in rsi_periods: row[f"RSI {p}"] = "-"
            rows.append(row)
            continue
        
        ltp = float(series.iloc[-1])
        row = { "SL No.": i, "Name": s["name"], "Industry": s["industry"], "LTP": ltp }

        for p in ema_periods:
            # Check if we have enough data for this EMA period
            if len(series) > p:  # Need more than period for proper calculation
                try:
                    ema_val = ema(series, p)
                    if not ema_val.empty and not pd.isna(ema_val.iloc[-1]):
                        ev = float(ema_val.iloc[-1])
                        row[f"EMA {p}"] = ev
                    else:
                        row[f"EMA {p}"] = "-"
                except:
                    row[f"EMA {p}"] = "-"
            else:
                row[f"EMA {p}"] = "-"

        for p in rsi_periods:
            # Check if we have enough data for this RSI period
            if len(series) > p + 1:  # Need more than period+1 for diff calculation
                try:
                    rsi_val = rsi(series, p)
                    if not rsi_val.empty and not pd.isna(rsi_val.iloc[-1]):
                        rv = float(rsi_val.iloc[-1])
                        # Check if RSI is valid (not all same values)
                        if 0 <= rv <= 100:
                            row[f"RSI {p}"] = rv
                        else:
                            row[f"RSI {p}"] = "-"
                    else:
                        row[f"RSI {p}"] = "-"
                except:
                    row[f"RSI {p}"] = "-"
            else:
                row[f"RSI {p}"] = "-"

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

# -------- TAB 5: Opportunity Scanner (Pink) --------
with tab5:
    # Emoji separate from neon text
    st.markdown('<div class="neon-header-wrapper border-lime"><span class="emoji-style">🔍</span> <span class="neon-text-lime">Opportunity Scanner</span></div>', unsafe_allow_html=True)
    #st.markdown("Find indices where **LTP > EMA(200, 100, 50, 21)** AND **RS Ratio(252, 126, 63, 21) > 0**")
    
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
            components.html(html_table, height=600, scrolling=True)

# FOOTER LINE
st.markdown("""<div class="footer">Made with ❤️ by <b>Stallions</b> | ©2026 Stallions - All Rights Reserved</div>""", text_alignment="center",unsafe_allow_html=True)
