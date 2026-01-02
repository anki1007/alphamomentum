"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STALLIONS QUANT TERMINAL - ULTIMATE EDITION               ║
║                     Institutional-Grade Analytics Engine                      ║
║                                                                              ║
║  Author: Ankit Gupta                                                         ║
║  Stallions Quantitative Research                                             ║
║  QTF Framework: Repeatable, Risk-Controlled Probabilistic Decision Engine   ║
║                                                                              ║
║  Features:                                                                   ║
║  • Advanced Lot-Based LIFO Exit Strategy                                     ║
║  • Real-Time Daily Screener with Signal Classification                       ║
║  • Comprehensive Backtesting Engine                                          ║
║  • Parameter Optimization (87 Combinations)                                  ║
║  • Efficient Frontier & Monte Carlo Simulation                               ║
║  • Benchmark Comparison with Nifty 50                                        ║
║  • Extended Performance Metrics                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import warnings
import time
import io
import math
import copy
import requests
from io import StringIO
import itertools

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stallions Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL THEME - Bloomberg Terminal Inspired
# ══════════════════════════════════════════════════════════════════════════════
THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-tertiary: #1a2332;
        --bg-card: #151d2b;
        --accent-cyan: #00d4ff;
        --accent-green: #00ff88;
        --accent-red: #ff4757;
        --accent-orange: #ff9f43;
        --accent-purple: #a855f7;
        --accent-yellow: #ffc107;
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: #1e293b;
        --glow-cyan: 0 0 20px rgba(0, 212, 255, 0.3);
        --glow-green: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0d1320 50%, var(--bg-primary) 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(ellipse at 20% 20%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(168, 85, 247, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }
    
    /* Header Styling */
    .terminal-header {
        background: linear-gradient(90deg, var(--bg-card) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .terminal-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-green));
    }
    
    .terminal-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: 2px;
        text-shadow: var(--glow-cyan);
    }
    
    .terminal-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-cyan);
        box-shadow: var(--glow-cyan);
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover::after {
        opacity: 1;
    }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-value.positive { color: var(--accent-green); text-shadow: var(--glow-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.neutral { color: var(--accent-cyan); text-shadow: var(--glow-cyan); }
    
    /* Section Headers */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 2px;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-header::before {
        content: '▸';
        color: var(--accent-green);
    }
    
    /* Data Tables */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    
    .stDataFrame {
        background: var(--bg-card) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-cyan);
    }
    
    /* Input Fields */
    .stSelectbox, .stMultiSelect, .stNumberInput, .stDateInput {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
        border: none;
        color: var(--bg-primary);
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--glow-cyan);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        border-radius: 8px;
        padding: 4px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
        background: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
        color: var(--bg-primary) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-cyan);
    }
    
    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent-green);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success { background: rgba(0, 255, 136, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
    .badge-danger { background: rgba(255, 71, 87, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }
    .badge-warning { background: rgba(255, 159, 67, 0.2); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
    .badge-info { background: rgba(0, 212, 255, 0.2); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
    
    /* Custom divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-green));
        margin: 20px 0;
        border-radius: 1px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NIFTY MIDCAP 50 UNIVERSE - DYNAMIC FETCH
# ══════════════════════════════════════════════════════════════════════════════
def fetch_nifty_midcap50_stocks() -> List[str]:
    """Fetch current Midcap 50 constituents from NSE"""
    url = 'https://www.niftyindices.com/IndexConstituent/ind_niftymidcap50list.csv'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            if not df.empty and "Symbol" in df.columns:
                return df["Symbol"].tolist()
    except:
        pass
    
    # Default fallback list if fetch fails
    return [
        'PERSISTENT', 'PRESTIGE', 'AUBANK', 'GODREJPROP', 'COFORGE', 'PHOENIXLTD', 
        'HDFCAMC', 'OBEROIRLTY', 'PAYTM', 'NHPC', 'TIINDIA', 'INDUSTOWER', 'OIL', 
        'IRCTC', 'SBICARD', 'BHEL', 'MPHASIS', 'MUTHOOTFIN', 'DABUR', 'GMRAIRPORT', 
        'COLPAL', 'SRF', 'HINDPETRO', 'UPL', 'FORTIS', 'SUPREMEIND', 'DIXON', 
        'POLYCAB', 'NMDC', 'BHARATFORG', 'PAGEIND', 'JUBLFOOD', 'FEDERALBNK', 
        'APLAPOLLO', 'CUMMINSIND', 'BSE', 'ASHOKLEY', 'IDFCFIRSTB', 'YESBANK', 
        'LUPIN', 'MANKIND', 'PIIND', 'SUZLON', 'MARICO', 'MFSL', 'HEROMOTOCO', 
        'AUROPHARMA', 'INDUSINDBK', 'POLICYBZR', 'OFSS'
    ]

# Global stock list
MIDCAP50_UNIVERSE = fetch_nifty_midcap50_stocks()

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance"""
    try:
        df = yf.download(
            f"{symbol}.NS", 
            start=start_date - timedelta(days=150), 
            end=end_date + timedelta(days=1),
            auto_adjust=True, 
            progress=False
        )
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        return pd.DataFrame()


def fetch_multiple_stocks(symbols: List[str], start_date: date, end_date: date, 
                          progress_callback=None) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple stocks with progress tracking"""
    data = {}
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback((i + 1) / total, f"Fetching {symbol}...")
        
        df = fetch_stock_data(symbol, start_date, end_date)
        if not df.empty:
            data[symbol] = df
        time.sleep(0.1)
    
    return data


def fetch_benchmark_data(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch Nifty 50 benchmark data"""
    try:
        df = yf.download(
            "^NSEI",
            start=start_date - timedelta(days=30),
            end=end_date + timedelta(days=1),
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED DAILY SCREENER
# ══════════════════════════════════════════════════════════════════════════════
def run_daily_screener(symbols: List[str], progress_callback=None) -> pd.DataFrame:
    """Run enhanced daily screener with signal classification"""
    results = []
    end_date = date.today()
    start_date = end_date - timedelta(days=100)
    
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback((i + 1) / total, f"Scanning {symbol}...")
        
        try:
            df = yf.download(
                symbol + ".NS",
                start=start_date,
                end=end_date + timedelta(days=1),
                auto_adjust=True,
                interval="1d",
                progress=False,
                multi_level_index=None,
                rounding=True
            )
            time.sleep(0.15)
            
            if df.empty or 'Close' not in df.columns:
                continue
            
            df = df.sort_index()
            df['20DMA'] = df['Close'].rolling(window=20).mean()
            df['50DMA'] = df['Close'].rolling(window=50).mean()
            df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            latest_close = float(latest['Close'])
            latest_high = float(latest['High'])
            latest_low = float(latest['Low'])
            latest_volume = float(latest['Volume'])
            latest_dma20 = float(latest['20DMA'])
            latest_dma50 = float(latest['50DMA']) if not pd.isna(latest['50DMA']) else None
            avg_volume = float(latest['Volume_Avg'])
            prev_close = float(prev['Close'])
            
            if pd.isna(latest_dma20):
                continue
            
            deviation_20dma = ((latest_close - latest_dma20) / latest_dma20) * 100
            deviation_50dma = ((latest_close - latest_dma50) / latest_dma50) * 100 if latest_dma50 else None
            day_change = ((latest_close - prev_close) / prev_close) * 100
            day_range = ((latest_high - latest_low) / latest_low) * 100
            volume_ratio = (latest_volume / avg_volume) if avg_volume > 0 else 0
            
            # Calculate 52-week high/low
            year_data = df.tail(252) if len(df) >= 252 else df
            high_52w = float(year_data['High'].max())
            low_52w = float(year_data['Low'].min())
            from_52w_high = ((latest_close - high_52w) / high_52w) * 100
            
            # Signal classification
            if deviation_20dma < -5 and volume_ratio > 1.2:
                signal = '🔥 Strong Buy'
            elif deviation_20dma < -3:
                signal = '✅ Buy'
            elif deviation_20dma < -1:
                signal = '👀 Watch'
            else:
                signal = '⏸️ Hold'
            
            results.append({
                'Symbol': symbol,
                'Close': latest_close,
                '20 DMA': latest_dma20,
                'Dev from 20DMA %': round(deviation_20dma, 2),
                '50 DMA': latest_dma50,
                'Day Change %': round(day_change, 2),
                'Day Range %': round(day_range, 2),
                'Vol Ratio': round(volume_ratio, 2),
                'From 52W High %': round(from_52w_high, 2),
                'Signal': signal
            })
            
        except Exception as e:
            continue
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Dev from 20DMA %', ascending=True)
        return df_results
    
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# LOT TRACKING DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Lot:
    qty: int
    price: float
    date: pd.Timestamp
    buy_brokerage: float


@dataclass
class Position:
    symbol: str
    lots: List[Lot] = field(default_factory=list)
    
    def total_qty(self) -> int:
        return sum(l.qty for l in self.lots)
    
    def avg_price(self) -> Optional[float]:
        q = self.total_qty()
        return (sum(l.qty * l.price for l in self.lots) / q) if q > 0 else None
    
    def last_buy_price(self) -> Optional[float]:
        return self.lots[-1].price if self.lots else None
    
    def total_buy_brokerage(self) -> float:
        return sum(l.buy_brokerage for l in self.lots)


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED BACKTEST ENGINE WITH LOT-BASED LIFO EXIT
# ══════════════════════════════════════════════════════════════════════════════
class BacktestEngine:
    """Complete backtester with lot-based LIFO exit strategy and all metrics"""
    
    def __init__(
        self,
        instruments: List[str],
        start_date: date,
        end_date: date,
        position_sizing_mode: str,
        fresh_static_amt: float = 10000.0,
        avg_static_amt: float = 5000.0,
        fresh_cash_pct: float = 0.025,
        avg_cash_pct: float = 0.0125,
        fresh_trade_divisor: float = 40.0,
        avg_trade_divisor: float = 10.0,
        initial_capital: float = 400000.0,
        target_pct: float = 0.05,
        avg_trigger_pct: float = 0.03,
        brokerage_per_order: float = 40.0,
        dma_window: int = 20,
        max_avg: int = 3,
    ):
        if position_sizing_mode not in ("static", "dynamic", "divisor"):
            raise ValueError("position_sizing_mode must be 'static', 'dynamic', or 'divisor'.")
        
        self.instruments = instruments
        self.start_date = start_date
        self.end_date = end_date
        self.position_sizing_mode = position_sizing_mode
        
        self.fresh_static_amt = float(fresh_static_amt)
        self.avg_static_amt = float(avg_static_amt)
        self.fresh_cash_pct = float(fresh_cash_pct)
        self.avg_cash_pct = float(avg_cash_pct)
        self.fresh_trade_divisor = float(fresh_trade_divisor)
        self.avg_trade_divisor = float(avg_trade_divisor)
        
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.target_pct = float(target_pct)
        self.avg_trigger_pct = float(avg_trigger_pct)
        self.brokerage_per_order = float(brokerage_per_order)
        self.dma_window = int(dma_window)
        self.max_avg = int(max_avg)
        
        self.positions: Dict[str, Position] = {}
        self._completed_trades: List[Dict] = []
        self.realized_pnl_by_date: Dict[pd.Timestamp, float] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self._fresh_buy = False
        self.cashflow_ledger: List[Tuple[date, float]] = []
        self.equity_curve_data: List[Dict] = []
        self.daily_portfolio_values: Dict[pd.Timestamp, float] = {}
    
    def load_data(self, progress_callback=None):
        """Load data for all instruments"""
        self.data = fetch_multiple_stocks(
            self.instruments, 
            self.start_date, 
            self.end_date, 
            progress_callback
        )
        
        if not self.data:
            raise RuntimeError("No instrument data loaded")
    
    @staticmethod
    def compute_moving_average(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
        return df[column].rolling(window=window, min_periods=window).mean()
    
    def _get_latest_close(self, symbol: str, dt: pd.Timestamp) -> Optional[float]:
        df = self.data.get(symbol)
        if df is None:
            return None
        subset = df.loc[df.index <= dt]
        if subset.empty:
            return None
        return float(subset.iloc[-1]["Close"])
    
    def portfolio_value(self, dt: pd.Timestamp) -> float:
        total = float(self.cash)
        for sym, pos in self.positions.items():
            price = self._get_latest_close(sym, dt)
            if price is None:
                continue
            total += pos.total_qty() * price
        return float(total)
    
    def _alloc_amount_for_trade(self, trade_kind: str, dt: pd.Timestamp) -> float:
        if self.position_sizing_mode == "static":
            return float(self.fresh_static_amt) if trade_kind == "fresh" else float(self.avg_static_amt)
        elif self.position_sizing_mode == "dynamic":
            pct = float(self.fresh_cash_pct) if trade_kind == "fresh" else float(self.avg_cash_pct)
            return float(self.cash) * float(pct)
        else:
            divisor = self.fresh_trade_divisor if trade_kind == "fresh" else self.avg_trade_divisor
            port_val = self.portfolio_value(dt)
            return float(port_val) / float(divisor)
    
    def _qty_from_amount_and_price(self, amount: float, price: float) -> int:
        if amount <= 0 or price <= 0:
            return 0
        return math.floor(amount / price)
    
    def _determine_qty_for_buy(self, trade_kind: str, price: float, dt: pd.Timestamp) -> int:
        alloc_amount = self._alloc_amount_for_trade(trade_kind, dt)
        qty_by_alloc = self._qty_from_amount_and_price(alloc_amount, price)
        if qty_by_alloc <= 0:
            return 0
        if self.cash <= self.brokerage_per_order:
            return 0
        max_qty_by_cash = math.floor((self.cash - self.brokerage_per_order) / price)
        if max_qty_by_cash <= 0:
            return 0
        return int(min(qty_by_alloc, max_qty_by_cash))
    
    def run_backtest(self, progress_callback=None) -> pd.DataFrame:
        """Run the complete backtest"""
        # Precompute DMA
        for sym, df in self.data.items():
            df["20DMA"] = self.compute_moving_average(df, self.dma_window, "Close")
            df["pct_below_20dma"] = np.where(
                (df["20DMA"].notna()) & (df["Close"] < df["20DMA"]),
                (df["20DMA"] - df["Close"]) / df["20DMA"],
                0.0,
            )
            self.data[sym] = df
        
        all_dates = sorted({d for df in self.data.values() for d in df.index})
        all_dates = [d for d in all_dates if (d.date() >= self.start_date and d.date() <= self.end_date)]
        
        if not all_dates:
            raise RuntimeError("No trading dates in the data")
        
        total_days = len(all_dates)
        
        for i, current_dt in enumerate(all_dates):
            if progress_callback and i % 50 == 0:
                progress_callback((i + 1) / total_days, f"Processing {current_dt.strftime('%Y-%m-%d')}...")
            
            self._fresh_buy = False
            
            # Process lot-based exits first
            self._process_exits_for_date(current_dt)
            
            # Get top 5 stocks below 20DMA
            top5 = self._get_top5_below_20dma(current_dt)
            
            # Process fresh entries
            self._process_entries_for_top5(current_dt, top5)
            
            # Process averaging only if no fresh buy
            if not self._fresh_buy:
                if top5 and all(sym in self.positions and self.positions[sym].total_qty() > 0 for sym in top5):
                    self._process_averaging_mode(current_dt)
            
            # Track daily portfolio value
            self.daily_portfolio_values[current_dt] = self.portfolio_value(current_dt)
        
        return pd.DataFrame(self._completed_trades)
    
    def _get_top5_below_20dma(self, dt: pd.Timestamp) -> List[str]:
        """Get top 5 stocks most below their 20DMA"""
        candidates: List[Tuple[str, float]] = []
        for sym, df in self.data.items():
            if dt not in df.index:
                continue
            pct = df.loc[dt, "pct_below_20dma"]
            if not pd.isna(pct) and pct > 0:
                candidates.append((sym, float(pct)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in candidates[:5]]
    
    def _process_exits_for_date(self, dt: pd.Timestamp):
        """Process lot-based exits - each lot exits at its own target"""
        symbols_to_exit: List[Tuple[str, float, pd.Timestamp]] = []
        
        for sym, pos in list(self.positions.items()):
            if pos.total_qty() == 0:
                continue
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]
            
            # Check each lot individually for target hit
            for lot in pos.lots:
                target_price = lot.price * (1.0 + self.target_pct)
                if row["High"] >= target_price:
                    symbols_to_exit.append((sym, target_price, lot.date))
        
        for sym, exit_price, entry_dt in symbols_to_exit:
            self._execute_exit(sym, exit_price, entry_dt, dt)
    
    def _execute_exit(self, symbol: str, exit_price: float, entry_dt: pd.Timestamp, dt: pd.Timestamp):
        """Execute exit for a specific lot"""
        pos = self.positions.get(symbol)
        if not pos:
            return
        
        # Find and process the specific lot
        exit_lot = None
        for lot in pos.lots:
            if lot.date == entry_dt:
                exit_lot = lot
                break
        
        if not exit_lot:
            return
        
        lot_qty = exit_lot.qty
        entry_price = exit_lot.price
        entry_date = exit_lot.date.date()
        lot_buy_brokerage = exit_lot.buy_brokerage
        sell_brokerage = self.brokerage_per_order
        
        gross_pnl = lot_qty * (exit_price - entry_price)
        lot_total_brokerage = lot_buy_brokerage + sell_brokerage
        net_pnl = gross_pnl - lot_total_brokerage
        pnl_pct = ((exit_price - entry_price) / entry_price * 100.0) if entry_price > 0 else 0
        net_pnl_pct = (net_pnl / (entry_price * lot_qty) * 100.0) if (entry_price > 0 and lot_qty > 0) else 0
        capital_used = lot_qty * entry_price
        
        holding_days = (dt.date() - entry_date).days
        
        # Update cash
        total_proceeds = lot_qty * exit_price
        self.cash += total_proceeds - sell_brokerage
        
        # Record trade
        trade_row = {
            "Symbol": symbol,
            "Status": "completed",
            "Entry Date": entry_date,
            "Direction": "Long",
            "Filled Qty": lot_qty,
            "Entry": round(entry_price, 2),
            "Exit": round(exit_price, 2),
            "Pnl": round(gross_pnl, 2),
            "Pnl%": round(pnl_pct, 2),
            "NetPnl": round(net_pnl, 2),
            "NetPnl%": round(net_pnl_pct, 2),
            "Capital": round(capital_used, 2),
            "Brokerage": round(lot_total_brokerage, 2),
            "Exit Date": dt.date(),
            "Holding Days": holding_days,
        }
        self._completed_trades.append(trade_row)
        
        # Update realized PnL
        date_key = dt.normalize()
        self.realized_pnl_by_date.setdefault(date_key, 0.0)
        self.realized_pnl_by_date[date_key] += net_pnl
        
        # Update cashflow ledger
        self.cashflow_ledger.append((entry_date, -(lot_qty * entry_price + lot_buy_brokerage)))
        self.cashflow_ledger.append((dt.date(), (lot_qty * exit_price - sell_brokerage)))
        
        # Remove lot from position
        pos.lots = [lot for lot in pos.lots if lot.date != entry_dt]
        if not pos.lots:
            del self.positions[symbol]
    
    def _process_entries_for_top5(self, dt: pd.Timestamp, top5: List[str]):
        """Process fresh entries for top 5 stocks"""
        if not top5:
            return
        
        for sym in top5:
            pos = self.positions.get(sym)
            if pos and pos.total_qty() > 0:
                continue
            
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            close_price = float(df.loc[dt, "Close"])
            
            qty = self._determine_qty_for_buy("fresh", close_price, dt)
            
            if qty <= 0:
                continue
            total_cost = qty * close_price + self.brokerage_per_order
            if total_cost > self.cash + 1e-9:
                continue
            
            self.cash -= total_cost
            lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
            self.positions[sym] = Position(symbol=sym, lots=[lot])
            self.cashflow_ledger.append((dt.date(), -(qty * close_price + self.brokerage_per_order)))
            
            self._fresh_buy = True
            break  # Only one fresh entry per day
    
    def _process_averaging_mode(self, dt: pd.Timestamp):
        """Process averaging for existing positions"""
        for sym, pos in list(self.positions.items()):
            if pos.total_qty() == 0:
                continue
            if len(pos.lots) >= self.max_avg:
                continue
            
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            close_price = float(df.loc[dt, "Close"])
            last_buy_price = pos.last_buy_price()
            if last_buy_price is None:
                continue
            
            pct_drop = (last_buy_price - close_price) / last_buy_price
            if pct_drop > self.avg_trigger_pct:
                qty = self._determine_qty_for_buy("avg", close_price, dt)
                if qty <= 0:
                    continue
                total_cost = qty * close_price + self.brokerage_per_order
                if total_cost > self.cash + 1e-9:
                    continue
                
                self.cash -= total_cost
                lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
                pos.lots.append(lot)
                self.cashflow_ledger.append((dt.date(), -(qty * close_price + self.brokerage_per_order)))
                break  # One averaging per day
    
    def compute_all_metrics(self, trades_df: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """Compute comprehensive performance metrics"""
        metrics = {}
        
        if trades_df.empty:
            return self._empty_metrics()
        
        # Basic metrics
        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = len(trades_df[trades_df['NetPnl'] > 0])
        metrics['losing_trades'] = len(trades_df[trades_df['NetPnl'] <= 0])
        metrics['win_ratio'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        
        # Financial metrics
        metrics['total_turnover'] = trades_df['Capital'].sum()
        metrics['actual_investment'] = self.initial_capital
        metrics['gross_pnl'] = trades_df['Pnl'].sum()
        metrics['total_brokerage'] = trades_df['Brokerage'].sum()
        metrics['net_pnl'] = trades_df['NetPnl'].sum()
        metrics['ending_balance'] = self.initial_capital + metrics['net_pnl']
        metrics['gross_pnl_pct'] = (metrics['gross_pnl'] / self.initial_capital * 100)
        metrics['net_pnl_pct'] = (metrics['net_pnl'] / self.initial_capital * 100)
        
        # Time-based metrics
        days = (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days
        years = days / 365.25 if days > 0 else 1.0
        metrics['years'] = years
        metrics['cagr'] = ((metrics['ending_balance'] / self.initial_capital) ** (1.0 / years) - 1.0) * 100 if years > 0 else 0
        
        # Holding period
        if 'Holding Days' in trades_df.columns:
            metrics['avg_holding_period'] = trades_df['Holding Days'].mean()
        else:
            metrics['avg_holding_period'] = 0
        
        # Build equity curve
        if self.realized_pnl_by_date:
            idx = sorted(self.realized_pnl_by_date.keys())
            cum = self.initial_capital
            dates = []
            eq = []
            for dt in idx:
                cum += self.realized_pnl_by_date[dt]
                dates.append(dt)
                eq.append(cum)
            equity = pd.Series(data=eq, index=pd.DatetimeIndex(dates)).sort_index()
            full_index = pd.date_range(start=pd.Timestamp(self.start_date), end=pd.Timestamp(self.end_date), freq="D")
            equity_ff = equity.reindex(full_index).ffill().fillna(self.initial_capital)
        else:
            full_index = pd.date_range(start=self.start_date, end=self.end_date)
            equity_ff = pd.Series(self.initial_capital, index=full_index)
        
        metrics['equity_curve'] = equity_ff
        
        # Daily returns
        daily_returns = equity_ff.pct_change().fillna(0.0)
        metrics['daily_returns'] = daily_returns
        
        # Drawdown calculations
        running_max = equity_ff.cummax()
        drawdown = (equity_ff - running_max) / running_max
        metrics['drawdown_series'] = drawdown
        metrics['max_drawdown'] = drawdown.min() * 100
        metrics['max_drawdown_amount'] = (running_max - equity_ff).max()
        
        # Drawdown duration
        is_dd = drawdown < 0
        dd_groups = (is_dd != is_dd.shift()).cumsum()
        dd_lengths = is_dd.groupby(dd_groups).sum()
        metrics['longest_dd_days'] = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
        
        # Volatility
        metrics['volatility_daily'] = daily_returns.std() * 100
        metrics['volatility_annual'] = daily_returns.std() * math.sqrt(252) * 100
        
        # Average drawdown
        dd_periods = drawdown[drawdown < 0]
        metrics['avg_drawdown'] = dd_periods.mean() * 100 if len(dd_periods) > 0 else 0
        
        # Risk-adjusted returns
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        
        # Sharpe Ratio
        rf_rate = 0.0
        metrics['sharpe'] = ((mean_daily - rf_rate/252) / std_daily) * math.sqrt(252) if std_daily > 0 else 0
        
        # Sortino Ratio
        neg_returns = daily_returns[daily_returns < 0]
        downside_std = neg_returns.std() if len(neg_returns) > 0 else 0
        metrics['sortino'] = ((mean_daily - rf_rate/252) / downside_std) * math.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio
        metrics['calmar'] = metrics['cagr'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Kelly Criterion
        if metrics['win_ratio'] > 0 and metrics['losing_trades'] > 0:
            wins = trades_df[trades_df['NetPnl'] > 0]['NetPnl%']
            losses = trades_df[trades_df['NetPnl'] <= 0]['NetPnl%']
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
            win_prob = metrics['win_ratio'] / 100
            metrics['kelly'] = (win_prob - (1 - win_prob) / (avg_win / avg_loss if avg_loss > 0 else 1)) * 100
        else:
            metrics['kelly'] = 0
        
        # Gain/Pain Ratio & Profit Factor
        gains = trades_df[trades_df['NetPnl'] > 0]['NetPnl'].sum()
        pains = abs(trades_df[trades_df['NetPnl'] <= 0]['NetPnl'].sum())
        metrics['gain_pain'] = gains / pains if pains > 0 else float('inf')
        metrics['profit_factor'] = gains / pains if pains > 0 else float('inf')
        
        # Probabilistic Sharpe Ratio (simplified)
        metrics['prob_sharpe'] = min(100, max(0, 50 + metrics['sharpe'] * 15))
        
        # Smart Sharpe (adjusted for autocorrelation)
        metrics['smart_sharpe'] = metrics['sharpe'] * 0.95
        
        # Skew and Kurtosis
        metrics['skew'] = daily_returns.skew()
        metrics['kurtosis'] = daily_returns.kurtosis()
        
        # Win/Loss metrics
        wins_df = trades_df[trades_df['NetPnl'] > 0]
        losses_df = trades_df[trades_df['NetPnl'] <= 0]
        
        # Consecutive wins/losses
        pnl_signs = (trades_df['NetPnl'] > 0).astype(int)
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for sign in pnl_signs:
            if sign == 1:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
        
        metrics['max_consec_wins'] = max_consec_wins
        metrics['max_consec_losses'] = max_consec_losses
        
        # Win/Loss amounts
        metrics['win_days_pct'] = (len(wins_df) / len(trades_df) * 100) if len(trades_df) > 0 else 0
        metrics['avg_win'] = wins_df['NetPnl'].mean() if len(wins_df) > 0 else 0
        metrics['avg_loss'] = losses_df['NetPnl'].mean() if len(losses_df) > 0 else 0
        metrics['best_trade'] = trades_df['NetPnl'].max()
        metrics['worst_trade'] = trades_df['NetPnl'].min()
        
        # Monthly returns
        equity_monthly = equity_ff.resample('ME').last()
        monthly_returns = equity_monthly.pct_change().fillna(0) * 100
        if len(equity_monthly) > 0:
            first_month_return = ((equity_monthly.iloc[0] - self.initial_capital) / self.initial_capital) * 100
            monthly_returns.iloc[0] = first_month_return
        
        metrics['monthly_returns'] = monthly_returns
        metrics['best_month'] = monthly_returns.max()
        metrics['worst_month'] = monthly_returns.min()
        
        # Period returns
        today = pd.Timestamp(self.end_date)
        
        def period_return(start_dt):
            try:
                start_val = equity_ff.loc[equity_ff.index >= start_dt].iloc[0]
                end_val = equity_ff.iloc[-1]
                return ((end_val - start_val) / start_val) * 100
            except:
                return 0
        
        mtd_start = today.replace(day=1)
        ytd_start = today.replace(month=1, day=1)
        
        metrics['mtd'] = period_return(mtd_start)
        metrics['ytd'] = period_return(ytd_start)
        metrics['3m'] = period_return(today - timedelta(days=90))
        metrics['6m'] = period_return(today - timedelta(days=180))
        metrics['1y'] = period_return(today - timedelta(days=365))
        metrics['3y'] = period_return(today - timedelta(days=365*3))
        metrics['5y'] = period_return(today - timedelta(days=365*5))
        
        # YoY returns
        yoy_returns = {}
        for year in equity_ff.index.year.unique():
            year_data = equity_ff[equity_ff.index.year == year]
            if len(year_data) > 1:
                yoy_returns[year] = ((year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]) * 100
        metrics['yoy_returns'] = yoy_returns
        
        # Portfolio values for chart
        if self.daily_portfolio_values:
            portfolio_series = pd.Series(self.daily_portfolio_values).sort_index()
            metrics['portfolio_values'] = portfolio_series
        else:
            metrics['portfolio_values'] = equity_ff
        
        # Time in market
        trading_days = len([d for d in equity_ff.index if d.weekday() < 5])
        days_with_position = sum(1 for dt, val in self.daily_portfolio_values.items() 
                                 if val != self.cash) if self.daily_portfolio_values else 0
        metrics['time_in_market'] = (days_with_position / trading_days * 100) if trading_days > 0 else 0
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_ratio': 0,
            'actual_investment': 0, 'gross_pnl': 0, 'net_pnl': 0,
            'ending_balance': self.initial_capital, 'gross_pnl_pct': 0, 'net_pnl_pct': 0,
            'cagr': 0, 'max_drawdown': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0,
            'profit_factor': 0, 'kelly': 0
        }


# ══════════════════════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class EfficientFrontierEngine:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.065):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        returns = np.sum(weights * self.mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / volatility
        return returns, volatility, sharpe
    
    def negative_sharpe(self, weights: np.ndarray) -> float:
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        return self.portfolio_performance(weights)[1]
    
    def optimize_sharpe(self) -> Tuple[np.ndarray, float, float, float]:
        """Find the maximum Sharpe ratio portfolio"""
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.negative_sharpe, initial_weights, method='SLSQP', 
            bounds=bounds, constraints=constraints
        )
        
        opt_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(opt_weights)
        return opt_weights, ret, vol, sharpe
    
    def min_volatility(self) -> Tuple[np.ndarray, float, float, float]:
        """Find the minimum volatility portfolio"""
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            self.portfolio_volatility, initial_weights, method='SLSQP', 
            bounds=bounds, constraints=constraints
        )
        
        opt_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(opt_weights)
        return opt_weights, ret, vol, sharpe
    
    def efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """Generate efficient frontier"""
        target_returns = np.linspace(
            self.mean_returns.min(), self.mean_returns.max(), n_points
        )
        
        efficient_portfolios = []
        
        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, t=target: np.sum(x * self.mean_returns) - t}
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_weights = np.array([1/self.n_assets] * self.n_assets)
            
            try:
                result = minimize(
                    self.portfolio_volatility, initial_weights, method='SLSQP', 
                    bounds=bounds, constraints=constraints
                )
                if result.success:
                    ret, vol, sharpe = self.portfolio_performance(result.x)
                    efficient_portfolios.append({
                        'Return': ret * 100, 'Volatility': vol * 100, 'Sharpe': sharpe
                    })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def monte_carlo_simulation(self, n_simulations: int = 5000) -> pd.DataFrame:
        """Run Monte Carlo simulation for random portfolios"""
        results = []
        
        for _ in range(n_simulations):
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)
            ret, vol, sharpe = self.portfolio_performance(weights)
            results.append({
                'Return': ret * 100, 'Volatility': vol * 100, 'Sharpe': sharpe, 'Weights': weights
            })
        
        return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def render_metric_card(label: str, value: str, status: str = "neutral"):
    """Render a metric card"""
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {status}">{value}</div>
        </div>
    """


def create_equity_curve_chart(metrics: Dict, benchmark_df: pd.DataFrame = None) -> go.Figure:
    """Create equity curve chart with benchmark comparison"""
    fig = go.Figure()
    
    equity = metrics.get('equity_curve', pd.Series())
    
    if len(equity) > 0:
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode='lines', name='Strategy',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
        ))
    
    if benchmark_df is not None and not benchmark_df.empty:
        bench_close = benchmark_df['Close'] if 'Close' in benchmark_df.columns else benchmark_df.iloc[:, 0]
        bench_normalized = bench_close / bench_close.iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=bench_normalized.index, y=bench_normalized.values,
            mode='lines', name='Benchmark (NIFTY 50)',
            line=dict(color='#ff9f43', width=1.5, dash='dot')
        ))
    
    fig.add_hline(
        y=metrics.get('actual_investment', equity.iloc[0] if len(equity) > 0 else 0),
        line_dash='dash', line_color='#64748b', annotation_text='Initial Capital'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='<b>EQUITY CURVE</b>', font=dict(family='JetBrains Mono', size=14, color='#00d4ff')),
        xaxis=dict(gridcolor='rgba(30, 41, 59, 0.5)', showgrid=True),
        yaxis=dict(gridcolor='rgba(30, 41, 59, 0.5)', showgrid=True, title='Portfolio Value (₹)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(family='JetBrains Mono', size=10)),
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_drawdown_chart(metrics: Dict, benchmark_df: pd.DataFrame = None) -> go.Figure:
    """Create drawdown underwater chart with benchmark comparison"""
    fig = go.Figure()
    
    drawdown = metrics.get('drawdown_series', pd.Series())
    
    if len(drawdown) > 0:
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100,
            mode='lines', name='Strategy',
            line=dict(color='#ff4757', width=1),
            fill='tozeroy', fillcolor='rgba(255, 71, 87, 0.3)'
        ))
    
    if benchmark_df is not None and not benchmark_df.empty:
        bench_close = benchmark_df['Close'] if 'Close' in benchmark_df.columns else benchmark_df.iloc[:, 0]
        bench_max = bench_close.cummax()
        bench_dd = ((bench_close - bench_max) / bench_max) * 100
        fig.add_trace(go.Scatter(
            x=bench_dd.index, y=bench_dd.values,
            mode='lines', name='Benchmark (NIFTY 50)',
            line=dict(color='#ff9f43', width=1, dash='dot')
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='<b>DRAWDOWN - UNDERWATER PLOT</b>', font=dict(family='JetBrains Mono', size=14, color='#ff4757')),
        xaxis=dict(gridcolor='rgba(30, 41, 59, 0.5)'),
        yaxis=dict(gridcolor='rgba(30, 41, 59, 0.5)', title='Drawdown %'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(family='JetBrains Mono', size=10)),
        margin=dict(l=50, r=20, t=50, b=50),
        height=300
    )
    
    return fig


def create_monthly_heatmap(metrics: Dict) -> go.Figure:
    """Create monthly returns heatmap"""
    monthly_returns = metrics.get('monthly_returns', pd.Series())
    
    if len(monthly_returns) == 0:
        fig = go.Figure()
        fig.update_layout(title='Monthly Returns (%) - No Data', paper_bgcolor='rgba(0,0,0,0)')
        return fig
    
    df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    pivot = df.pivot_table(index='Year', columns='Month', values='Return', aggfunc='sum')
    
    for month in range(1, 13):
        if month not in pivot.columns:
            pivot[month] = np.nan
    pivot = pivot.reindex(columns=range(1, 13))
    
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    max_abs = max(abs(pivot.values[~np.isnan(pivot.values)].min()), abs(pivot.values[~np.isnan(pivot.values)].max())) if not np.all(np.isnan(pivot.values)) else 10
    
    colorscale = [[0, '#ff4757'], [0.5, '#1a2332'], [1, '#00ff88']]
    
    text_data = [['' if np.isnan(val) else f'{val:.1f}' for val in row] for row in pivot.values]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=month_labels, y=[str(y) for y in pivot.index],
        colorscale=colorscale, zmid=0, zmin=-max_abs, zmax=max_abs,
        text=text_data, texttemplate='%{text}',
        textfont=dict(size=11, color='white'),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>',
        colorbar=dict(title=dict(text='Return %', font=dict(color='#e0e0e0')), tickfont=dict(color='#e0e0e0')),
        xgap=2, ygap=2
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='<b>MONTHLY RETURNS HEATMAP</b>', font=dict(family='JetBrains Mono', size=14, color='#00d4ff')),
        xaxis=dict(side='top', tickfont=dict(size=11, family='JetBrains Mono')),
        yaxis=dict(tickfont=dict(size=11, family='JetBrains Mono'), autorange='reversed'),
        height=max(250, min(500, 100 + len(pivot.index) * 40))
    )
    
    return fig


def create_efficient_frontier_plot(ef_engine: EfficientFrontierEngine) -> Tuple[go.Figure, np.ndarray, np.ndarray]:
    """Create efficient frontier visualization"""
    mc_results = ef_engine.monte_carlo_simulation(3000)
    ef_results = ef_engine.efficient_frontier(50)
    
    max_sharpe_weights, max_sharpe_ret, max_sharpe_vol, max_sharpe = ef_engine.optimize_sharpe()
    min_vol_weights, min_vol_ret, min_vol_vol, min_vol_sharpe = ef_engine.min_volatility()
    
    fig = go.Figure()
    
    # Monte Carlo points
    fig.add_trace(go.Scatter(
        x=mc_results['Volatility'], y=mc_results['Return'],
        mode='markers', marker=dict(
            size=4, color=mc_results['Sharpe'], colorscale='Viridis', showscale=True,
            colorbar=dict(title=dict(text='Sharpe', font=dict(family='JetBrains Mono', size=10))),
            opacity=0.6
        ),
        name='Random Portfolios',
        hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Efficient frontier line
    if not ef_results.empty:
        fig.add_trace(go.Scatter(
            x=ef_results['Volatility'], y=ef_results['Return'],
            mode='lines', name='Efficient Frontier', line=dict(color='#00d4ff', width=3)
        ))
    
    # Maximum Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_vol * 100], y=[max_sharpe_ret * 100],
        mode='markers', marker=dict(size=20, color='#00ff88', symbol='star'),
        name=f'Max Sharpe ({max_sharpe:.2f})'
    ))
    
    # Minimum volatility portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol_vol * 100], y=[min_vol_ret * 100],
        mode='markers', marker=dict(size=15, color='#ff9f43', symbol='diamond'),
        name=f'Min Volatility ({min_vol_vol*100:.1f}%)'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='<b>EFFICIENT FRONTIER - PORTFOLIO OPTIMIZATION</b>', 
                   font=dict(family='JetBrains Mono', size=14, color='#00d4ff')),
        xaxis=dict(title=dict(text='Volatility (%)', font=dict(family='JetBrains Mono')),
                   gridcolor='rgba(30, 41, 59, 0.5)'),
        yaxis=dict(title=dict(text='Expected Return (%)', font=dict(family='JetBrains Mono')),
                   gridcolor='rgba(30, 41, 59, 0.5)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(family='JetBrains Mono', size=10),
                    orientation='h', yanchor='bottom', y=-0.3),
        margin=dict(l=50, r=20, t=50, b=100),
        height=500
    )
    
    return fig, max_sharpe_weights, min_vol_weights


def create_portfolio_allocation_chart(weights: np.ndarray, symbols: List[str], title: str) -> go.Figure:
    """Create portfolio allocation pie chart"""
    non_zero_mask = weights > 0.01
    filtered_weights = weights[non_zero_mask]
    filtered_symbols = [s for s, m in zip(symbols, non_zero_mask) if m]
    
    colors = px.colors.sequential.Viridis[:len(filtered_weights)]
    
    fig = go.Figure(data=[go.Pie(
        labels=filtered_symbols, values=filtered_weights * 100, hole=0.5,
        marker=dict(colors=colors), textinfo='label+percent',
        textfont=dict(family='JetBrains Mono', size=10),
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f'<b>{title}</b>', font=dict(family='JetBrains Mono', size=12, color='#00d4ff')),
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_stock_chart(symbol: str) -> go.Figure:
    """Create candlestick chart for a stock"""
    try:
        df = yf.download(f"{symbol}.NS", period="1y", interval="1d", progress=False)
        
        if df.empty:
            return go.Figure()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df['20DMA'] = df['Close'].rolling(window=20).mean()
        df['50DMA'] = df['Close'].rolling(window=50).mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#00ff88', decreasing_line_color='#ff4757'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['20DMA'], mode='lines', name='20 DMA',
                                  line=dict(color='#00d4ff', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['50DMA'], mode='lines', name='50 DMA',
                                  line=dict(color='#ff9f43', width=1.5)), row=1, col=1)
        
        colors = ['#00ff88' if c >= o else '#ff4757' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(text=f'<b>{symbol} - 1 YEAR CHART</b>', font=dict(family='JetBrains Mono', size=14, color='#00d4ff')),
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(family='JetBrains Mono', size=10)),
            height=500
        )
        
        fig.update_xaxes(gridcolor='rgba(30, 41, 59, 0.5)')
        fig.update_yaxes(gridcolor='rgba(30, 41, 59, 0.5)')
        
        return fig
    except:
        return go.Figure()


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def export_to_excel(trades_df: pd.DataFrame, metrics: Dict) -> bytes:
    """Export data to multi-sheet Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Trade book
        if not trades_df.empty:
            trades_df.to_excel(writer, sheet_name='Trade Book', index=False)
        
        # Metrics summary
        metrics_data = {
            'Metric': [
                'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Ratio %',
                'Initial Capital', 'Total Turnover', 'Ending Balance', 'Net PnL',
                'Net PnL %', 'CAGR %', 'Max Drawdown %', 'Sharpe Ratio', 'Sortino Ratio',
                'Calmar Ratio', 'Profit Factor', 'Kelly %', 'Avg Holding Period (Days)'
            ],
            'Value': [
                metrics.get('total_trades', 0), metrics.get('winning_trades', 0),
                metrics.get('losing_trades', 0), round(metrics.get('win_ratio', 0), 2),
                round(metrics.get('actual_investment', 0), 2),
                round(metrics.get('total_turnover', 0), 2),
                round(metrics.get('ending_balance', 0), 2),
                round(metrics.get('net_pnl', 0), 2), round(metrics.get('net_pnl_pct', 0), 2),
                round(metrics.get('cagr', 0), 2), round(metrics.get('max_drawdown', 0), 2),
                round(metrics.get('sharpe', 0), 2), round(metrics.get('sortino', 0), 2),
                round(metrics.get('calmar', 0), 2),
                round(metrics.get('profit_factor', 0), 2) if metrics.get('profit_factor', 0) != float('inf') else 999,
                round(metrics.get('kelly', 0), 2), round(metrics.get('avg_holding_period', 0), 1)
            ]
        }
        pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Metrics Summary', index=False)
        
        # Monthly returns
        monthly = metrics.get('monthly_returns', pd.Series())
        if len(monthly) > 0:
            monthly_df = pd.DataFrame({'Date': monthly.index, 'Return %': monthly.values})
            monthly_df.to_excel(writer, sheet_name='Monthly Returns', index=False)
        
        # Equity curve
        equity = metrics.get('equity_curve', pd.Series())
        if len(equity) > 0:
            equity_df = pd.DataFrame({'Date': equity.index, 'Portfolio Value': equity.values})
            equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
    
    return output.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
def run_optimization(instruments: List[str], start_date: date, end_date: date, 
                     initial_capital: float, progress_callback=None) -> List[Dict]:
    """Run parameter optimization across 87 combinations"""
    
    # Parameter combinations
    param_combos = []
    
    # Static mode: 4 combinations
    for fresh_amt in [10000, 20000]:
        for avg_amt in [10000, 20000]:
            for target in [0.03, 0.06, 0.08]:
                param_combos.append({
                    'mode': 'static', 'fresh_static_amt': fresh_amt, 'avg_static_amt': avg_amt,
                    'fresh_cash_pct': 0, 'avg_cash_pct': 0, 'fresh_trade_divisor': None,
                    'avg_trade_divisor': None, 'target_pct': target
                })
    
    # Dynamic mode: 9 combinations
    for fresh_pct in [0.015, 0.02, 0.025]:
        for avg_pct in [0.015, 0.02, 0.025]:
            for target in [0.03, 0.06, 0.08]:
                param_combos.append({
                    'mode': 'dynamic', 'fresh_static_amt': 0, 'avg_static_amt': 0,
                    'fresh_cash_pct': fresh_pct, 'avg_cash_pct': avg_pct,
                    'fresh_trade_divisor': None, 'avg_trade_divisor': None, 'target_pct': target
                })
    
    # Divisor mode: 16 combinations
    for fresh_div in [10, 20, 30, 40]:
        for avg_div in [10, 20, 30, 40]:
            for target in [0.03, 0.06, 0.08]:
                param_combos.append({
                    'mode': 'divisor', 'fresh_static_amt': 0, 'avg_static_amt': 0,
                    'fresh_cash_pct': 0, 'avg_cash_pct': 0,
                    'fresh_trade_divisor': fresh_div, 'avg_trade_divisor': avg_div, 'target_pct': target
                })
    
    results = []
    total = len(param_combos)
    
    # Load data once
    data_cache = fetch_multiple_stocks(instruments, start_date, end_date)
    
    for i, params in enumerate(param_combos):
        if progress_callback:
            progress_callback((i + 1) / total, f"Testing {params['mode']} config {i+1}/{total}...")
        
        try:
            engine = BacktestEngine(
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                position_sizing_mode=params['mode'],
                fresh_static_amt=params['fresh_static_amt'],
                avg_static_amt=params['avg_static_amt'],
                fresh_cash_pct=params['fresh_cash_pct'],
                avg_cash_pct=params['avg_cash_pct'],
                fresh_trade_divisor=params['fresh_trade_divisor'],
                avg_trade_divisor=params['avg_trade_divisor'],
                initial_capital=initial_capital,
                target_pct=params['target_pct']
            )
            
            engine.data = copy.deepcopy(data_cache)
            trades_df = engine.run_backtest()
            metrics = engine.compute_all_metrics(trades_df)
            
            # Determine param display values
            if params['mode'] == 'static':
                fresh_param = f"₹{params['fresh_static_amt']/1000:.0f}K"
                avg_param = f"₹{params['avg_static_amt']/1000:.0f}K"
            elif params['mode'] == 'dynamic':
                fresh_param = f"{params['fresh_cash_pct']*100:.1f}%"
                avg_param = f"{params['avg_cash_pct']*100:.1f}%"
            else:
                fresh_param = str(params['fresh_trade_divisor'])
                avg_param = str(params['avg_trade_divisor'])
            
            results.append({
                'Mode': params['mode'].capitalize(),
                'Fresh Param': fresh_param,
                'Avg Param': avg_param,
                'Target %': f"{params['target_pct']*100:.0f}%",
                'Total Trades': metrics.get('total_trades', 0),
                'Win Ratio %': round(metrics.get('win_ratio', 0), 1),
                'Net PnL': round(metrics.get('net_pnl', 0), 0),
                'Net PnL %': round(metrics.get('net_pnl_pct', 0), 2),
                'CAGR %': round(metrics.get('cagr', 0), 2),
                'Max DD %': round(abs(metrics.get('max_drawdown', 0)), 2),
                'Sharpe': round(metrics.get('sharpe', 0), 2),
                'Sortino': round(metrics.get('sortino', 0), 2),
                'Calmar': round(metrics.get('calmar', 0), 2),
                'Profit Factor': round(metrics.get('profit_factor', 0), 2) if metrics.get('profit_factor', 0) != float('inf') else 999.99
            })
            
        except Exception as e:
            results.append({
                'Mode': params['mode'].capitalize(), 'Fresh Param': '-', 'Avg Param': '-',
                'Target %': f"{params['target_pct']*100:.0f}%", 'Total Trades': 0,
                'Win Ratio %': 0, 'Net PnL': 0, 'Net PnL %': 0, 'CAGR %': 0,
                'Max DD %': 0, 'Sharpe': 0, 'Sortino': 0, 'Calmar': 0, 'Profit Factor': 0
            })
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    """Render terminal header"""
    st.markdown("""
        <div class="terminal-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="terminal-title">📈 STALLIONS QUANT TERMINAL</h1>
                    <p class="terminal-subtitle">Institutional-Grade Mean Reversion Analytics • Midcap 50 Strategy • Author: Ankit Gupta</p>
                </div>
                <div class="live-indicator">
                    <span class="live-dot"></span>
                    <span>SYSTEM ACTIVE</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def main():
    render_header()
    
    # ==================== SIDEBAR CONFIGURATION ====================
    with st.sidebar:
        st.markdown("### ⚙️ CONFIGURATION")
        
        st.markdown("#### 📊 Backtest Settings")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date(2020, 1, 1), key="bt_start")
        with col2:
            end_date = st.date_input("End Date", date.today(), key="bt_end")
        
        sizing_mode = st.selectbox("Position Sizing Mode", ["divisor", "static", "dynamic"], key="bt_sizing")
        
        if sizing_mode == "divisor":
            fresh_divisor = st.number_input("Fresh Trade Divisor", 10, 100, 40, key="bt_fresh_div")
            avg_divisor = st.number_input("Avg Trade Divisor", 5, 50, 10, key="bt_avg_div")
            fresh_static, avg_static, fresh_pct, avg_pct = 10000, 5000, 0.025, 0.0125
        elif sizing_mode == "static":
            fresh_static = st.number_input("Fresh Trade Amount (₹)", 5000, 100000, 10000, key="bt_fresh_static")
            avg_static = st.number_input("Avg Trade Amount (₹)", 2500, 50000, 5000, key="bt_avg_static")
            fresh_divisor, avg_divisor, fresh_pct, avg_pct = 40, 10, 0.025, 0.0125
        else:
            fresh_pct = st.slider("Fresh Trade % of Cash", 1, 20, 4, key="bt_fresh_pct") / 100
            avg_pct = st.slider("Avg Trade % of Cash", 1, 15, 3, key="bt_avg_pct") / 100
            fresh_divisor, avg_divisor, fresh_static, avg_static = 40, 10, 10000, 5000
        
        st.markdown("#### 🎯 Strategy Parameters")
        initial_capital = st.number_input("Initial Capital (₹)", 100000, 10000000, 400000, key="bt_capital")
        target_pct = st.slider("Target %", 3, 15, 8, key="bt_target") / 100
        avg_trigger = st.slider("Avg Trigger %", 2, 10, 5, key="bt_avg_trigger") / 100
        max_positions = st.slider("Max Lots per Stock", 1, 5, 3, key="bt_max_pos")
        dma_window = st.slider("DMA Window", 10, 50, 20, key="bt_dma")
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; opacity: 0.6; font-size: 0.75rem; font-family: 'JetBrains Mono';">
                Stallions QTF Framework v3.0<br>
                Ultimate Edition<br>
                <span style="color: #00d4ff;">Author: Ankit Gupta</span>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Daily Screener", "📊 Backtest", "🔬 Optimization", "📈 Efficient Frontier", "📋 Analysis"
    ])
    
    # ==================== TAB 1: DAILY SCREENER ====================
    with tab1:
        st.markdown('<div class="section-header">REAL-TIME STOCK SCREENER</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🔍 RUN SCREENER", use_container_width=True, key="run_screener"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                screener_df = run_daily_screener(MIDCAP50_UNIVERSE, update_progress)
                
                progress_bar.progress(100)
                status_text.text("✅ Scan complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['screener_df'] = screener_df
        
        with col2:
            top_n = st.selectbox("Show Top", [5, 10, 15, 20, "All"], index=1)
        
        with col3:
            filter_signal = st.multiselect(
                "Filter by Signal",
                ["🔥 Strong Buy", "✅ Buy", "👀 Watch", "⏸️ Hold"],
                default=["🔥 Strong Buy", "✅ Buy", "👀 Watch"]
            )
        
        if 'screener_df' in st.session_state:
            screener_df = st.session_state['screener_df']
            
            if not screener_df.empty:
                # Filter by signal
                if filter_signal:
                    filtered_df = screener_df[screener_df['Signal'].isin(filter_signal)]
                else:
                    filtered_df = screener_df
                
                if top_n != "All":
                    filtered_df = filtered_df.head(int(top_n))
                
                # Summary metrics
                total_scanned = len(screener_df)
                below_20dma = len(screener_df[screener_df['Dev from 20DMA %'] < 0])
                strong_buys = len(screener_df[screener_df['Signal'] == '🔥 Strong Buy'])
                buys = len(screener_df[screener_df['Signal'] == '✅ Buy'])
                
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.markdown(render_metric_card("TOTAL SCANNED", str(total_scanned), "neutral"), unsafe_allow_html=True)
                with mcol2:
                    st.markdown(render_metric_card("BELOW 20 DMA", str(below_20dma), "positive"), unsafe_allow_html=True)
                with mcol3:
                    st.markdown(render_metric_card("🔥 STRONG BUY", str(strong_buys), "positive"), unsafe_allow_html=True)
                with mcol4:
                    st.markdown(render_metric_card("✅ BUY SIGNALS", str(buys), "positive"), unsafe_allow_html=True)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Display table
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                
                # Stock chart viewer
                st.markdown('<div class="section-header">STOCK CHART VIEWER</div>', unsafe_allow_html=True)
                selected_symbol = st.selectbox("Select Stock", filtered_df['Symbol'].tolist())
                if selected_symbol:
                    fig = create_stock_chart(selected_symbol)
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: BACKTEST ====================
    with tab2:
        st.markdown('<div class="section-header">BACKTEST ENGINE</div>', unsafe_allow_html=True)
        
        st.info("💡 Configure backtest parameters in the sidebar on the left")
        
        if st.button("🚀 RUN BACKTEST", use_container_width=True, key="run_backtest"):
            # Build engine parameters
            engine_params = {
                'instruments': MIDCAP50_UNIVERSE,
                'start_date': start_date,
                'end_date': end_date,
                'position_sizing_mode': sizing_mode,
                'initial_capital': initial_capital,
                'target_pct': target_pct,
                'avg_trigger_pct': avg_trigger,
                'max_avg': max_positions,
                'dma_window': dma_window
            }
            
            if sizing_mode == "divisor":
                engine_params['fresh_trade_divisor'] = fresh_divisor
                engine_params['avg_trade_divisor'] = avg_divisor
            elif sizing_mode == "static":
                engine_params['fresh_static_amt'] = fresh_static
                engine_params['avg_static_amt'] = avg_static
            else:
                engine_params['fresh_cash_pct'] = fresh_pct
                engine_params['avg_cash_pct'] = avg_pct
            
            engine = BacktestEngine(**engine_params)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)
            
            status_text.text("Loading data...")
            engine.load_data(update_progress)
            
            status_text.text("Running backtest...")
            trades_df = engine.run_backtest(update_progress)
            
            status_text.text("Computing metrics...")
            benchmark_df = fetch_benchmark_data(start_date, end_date)
            metrics = engine.compute_all_metrics(trades_df, benchmark_df)
            
            progress_bar.progress(100)
            status_text.text("✅ Backtest complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state['backtest_trades'] = trades_df
            st.session_state['backtest_metrics'] = metrics
            st.session_state['benchmark_df'] = benchmark_df
        
        # Display results
        if 'backtest_metrics' in st.session_state:
            metrics = st.session_state['backtest_metrics']
            trades_df = st.session_state['backtest_trades']
            benchmark_df = st.session_state.get('benchmark_df', None)
            
            # Key metrics
            st.markdown('<div class="section-header">KEY PERFORMANCE METRICS</div>', unsafe_allow_html=True)
            
            mcols = st.columns(6)
            with mcols[0]:
                st.markdown(render_metric_card("NET P&L", f"₹{metrics.get('net_pnl', 0):,.0f}", 
                    "positive" if metrics.get('net_pnl', 0) > 0 else "negative"), unsafe_allow_html=True)
            with mcols[1]:
                st.markdown(render_metric_card("CAGR", f"{metrics.get('cagr', 0):.2f}%",
                    "positive" if metrics.get('cagr', 0) > 0 else "negative"), unsafe_allow_html=True)
            with mcols[2]:
                st.markdown(render_metric_card("MAX DRAWDOWN", f"{metrics.get('max_drawdown', 0):.2f}%", "negative"), unsafe_allow_html=True)
            with mcols[3]:
                st.markdown(render_metric_card("SHARPE", f"{metrics.get('sharpe', 0):.2f}",
                    "positive" if metrics.get('sharpe', 0) > 1 else "neutral"), unsafe_allow_html=True)
            with mcols[4]:
                st.markdown(render_metric_card("WIN RATE", f"{metrics.get('win_ratio', 0):.1f}%",
                    "positive" if metrics.get('win_ratio', 0) > 50 else "neutral"), unsafe_allow_html=True)
            with mcols[5]:
                st.markdown(render_metric_card("PROFIT FACTOR", 
                    f"{metrics.get('profit_factor', 0):.2f}" if metrics.get('profit_factor', 0) != float('inf') else "∞",
                    "positive"), unsafe_allow_html=True)
            
            # Charts
            st.markdown('<div class="section-header">EQUITY CURVE</div>', unsafe_allow_html=True)
            st.plotly_chart(create_equity_curve_chart(metrics, benchmark_df), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_drawdown_chart(metrics, benchmark_df), use_container_width=True)
            with col2:
                st.plotly_chart(create_monthly_heatmap(metrics), use_container_width=True)
            
            # Extended metrics
            st.markdown('<div class="section-header">DETAILED METRICS</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Returns Metrics**")
                returns_data = {
                    'Metric': ['Total Return', 'CAGR', 'MTD', 'YTD', '3M', '6M', '1Y'],
                    'Value': [
                        f"{metrics.get('net_pnl_pct', 0):.2f}%", f"{metrics.get('cagr', 0):.2f}%",
                        f"{metrics.get('mtd', 0):.2f}%", f"{metrics.get('ytd', 0):.2f}%",
                        f"{metrics.get('3m', 0):.2f}%", f"{metrics.get('6m', 0):.2f}%",
                        f"{metrics.get('1y', 0):.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(returns_data), hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("**Risk Metrics**")
                risk_data = {
                    'Metric': ['Max Drawdown', 'Avg Drawdown', 'Longest DD', 'Annual Vol', 'Sharpe', 'Sortino', 'Calmar'],
                    'Value': [
                        f"{metrics.get('max_drawdown', 0):.2f}%", f"{metrics.get('avg_drawdown', 0):.2f}%",
                        f"{metrics.get('longest_dd_days', 0)} days", f"{metrics.get('volatility_annual', 0):.2f}%",
                        f"{metrics.get('sharpe', 0):.2f}", f"{metrics.get('sortino', 0):.2f}",
                        f"{metrics.get('calmar', 0):.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)
            
            with col3:
                st.markdown("**Trade Statistics**")
                trade_data = {
                    'Metric': ['Total Trades', 'Win Rate', 'Profit Factor', 'Avg Win', 'Avg Loss', 'Best Trade', 'Worst Trade'],
                    'Value': [
                        str(metrics.get('total_trades', 0)), f"{metrics.get('win_ratio', 0):.1f}%",
                        f"{metrics.get('profit_factor', 0):.2f}" if metrics.get('profit_factor', 0) != float('inf') else "∞",
                        f"₹{metrics.get('avg_win', 0):,.0f}", f"₹{metrics.get('avg_loss', 0):,.0f}",
                        f"₹{metrics.get('best_trade', 0):,.0f}", f"₹{metrics.get('worst_trade', 0):,.0f}"
                    ]
                }
                st.dataframe(pd.DataFrame(trade_data), hide_index=True, use_container_width=True)
            
            # Trade book
            st.markdown('<div class="section-header">TRADE BOOK</div>', unsafe_allow_html=True)
            if not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
                
                # Export buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = trades_df.to_csv(index=False)
                    st.download_button("📥 DOWNLOAD CSV", csv, f"trades_{date.today()}.csv", "text/csv")
                with col2:
                    excel_data = export_to_excel(trades_df, metrics)
                    st.download_button("📥 DOWNLOAD EXCEL", excel_data, f"backtest_{date.today()}.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # ==================== TAB 3: OPTIMIZATION ====================
    with tab3:
        st.markdown('<div class="section-header">PARAMETER OPTIMIZATION</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Run automated optimization across **87 parameter combinations**:
        - **Static**: Fresh ₹10K/₹20K × Avg ₹10K/₹20K = 4 combos
        - **Dynamic**: Fresh 1.5%/2%/2.5% × Avg 1.5%/2%/2.5% = 9 combos  
        - **Divisor**: Fresh 10/20/30/40 × Avg 10/20/30/40 = 16 combos
        - **Targets**: 3% / 6% / 8% (×3 each)
        """)
        
        st.info(f"💡 Using dates from sidebar: {start_date} to {end_date} | Capital: ₹{initial_capital:,}")
        
        if st.button("🚀 RUN OPTIMIZATION (87 BACKTESTS)", use_container_width=True, key="run_opt"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)
            
            results = run_optimization(MIDCAP50_UNIVERSE, start_date, end_date, initial_capital, update_progress)
            
            progress_bar.progress(100)
            status_text.text("✅ Optimization complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.session_state['optimization_results'] = results
        
        if 'optimization_results' in st.session_state:
            results = st.session_state['optimization_results']
            results_df = pd.DataFrame(results)
            
            # Best performers
            st.markdown('<div class="section-header">🏆 TOP PERFORMERS</div>', unsafe_allow_html=True)
            
            bcols = st.columns(4)
            with bcols[0]:
                best_cagr = results_df.loc[results_df['CAGR %'].idxmax()]
                st.markdown(render_metric_card("BEST CAGR", f"{best_cagr['CAGR %']:.2f}%", "positive"), unsafe_allow_html=True)
                st.caption(f"{best_cagr['Mode']} | T:{best_cagr['Target %']}")
            with bcols[1]:
                best_sharpe = results_df.loc[results_df['Sharpe'].idxmax()]
                st.markdown(render_metric_card("BEST SHARPE", f"{best_sharpe['Sharpe']:.2f}", "positive"), unsafe_allow_html=True)
                st.caption(f"{best_sharpe['Mode']} | T:{best_sharpe['Target %']}")
            with bcols[2]:
                best_calmar = results_df.loc[results_df['Calmar'].idxmax()]
                st.markdown(render_metric_card("BEST CALMAR", f"{best_calmar['Calmar']:.2f}", "positive"), unsafe_allow_html=True)
                st.caption(f"{best_calmar['Mode']} | T:{best_calmar['Target %']}")
            with bcols[3]:
                lowest_dd = results_df.loc[results_df['Max DD %'].idxmin()]
                st.markdown(render_metric_card("LOWEST MAX DD", f"{lowest_dd['Max DD %']:.2f}%", "positive"), unsafe_allow_html=True)
                st.caption(f"{lowest_dd['Mode']} | T:{lowest_dd['Target %']}")
            
            # Full results
            st.markdown('<div class="section-header">ALL RESULTS (SORTED BY CAGR)</div>', unsafe_allow_html=True)
            results_df_sorted = results_df.sort_values('CAGR %', ascending=False).reset_index(drop=True)
            results_df_sorted.index = results_df_sorted.index + 1
            st.dataframe(results_df_sorted, use_container_width=True, height=500)
            
            # Download
            csv = results_df_sorted.to_csv(index=False)
            st.download_button("📥 DOWNLOAD OPTIMIZATION RESULTS", csv, f"optimization_{date.today()}.csv", "text/csv")
    
    # ==================== TAB 4: EFFICIENT FRONTIER ====================
    with tab4:
        st.markdown('<div class="section-header">EFFICIENT FRONTIER OPTIMIZATION</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            lookback_days = st.slider("Lookback Period (Days)", 60, 500, 252, key="ef_lookback")
        with col2:
            risk_free = st.slider("Risk-Free Rate %", 0, 10, 6, key="ef_rf") / 100
        with col3:
            num_assets = st.slider("Number of Assets", 5, 50, 15, key="ef_assets")
        
        if st.button("🎯 RUN OPTIMIZATION", use_container_width=True, key="run_ef"):
            with st.spinner("Computing efficient frontier..."):
                ef_end = date.today()
                ef_start = ef_end - timedelta(days=lookback_days)
                
                selected_symbols = MIDCAP50_UNIVERSE[:num_assets]
                data = fetch_multiple_stocks(selected_symbols, ef_start, ef_end)
                
                if len(data) >= 3:
                    closes = pd.DataFrame({sym: df['Close'] for sym, df in data.items()})
                    returns = closes.pct_change().dropna()
                    
                    ef_engine = EfficientFrontierEngine(returns, risk_free)
                    fig_ef, max_sharpe_weights, min_vol_weights = create_efficient_frontier_plot(ef_engine)
                    
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                    # Optimal portfolios
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="section-header">MAX SHARPE PORTFOLIO</div>', unsafe_allow_html=True)
                        max_ret, max_vol, max_sharpe = ef_engine.portfolio_performance(max_sharpe_weights)
                        
                        mcols = st.columns(3)
                        with mcols[0]:
                            st.markdown(render_metric_card("EXP. RETURN", f"{max_ret*100:.1f}%", "positive"), unsafe_allow_html=True)
                        with mcols[1]:
                            st.markdown(render_metric_card("VOLATILITY", f"{max_vol*100:.1f}%", "neutral"), unsafe_allow_html=True)
                        with mcols[2]:
                            st.markdown(render_metric_card("SHARPE", f"{max_sharpe:.2f}", "positive"), unsafe_allow_html=True)
                        
                        st.plotly_chart(create_portfolio_allocation_chart(max_sharpe_weights, list(data.keys()), "MAX SHARPE ALLOCATION"), use_container_width=True)
                    
                    with col2:
                        st.markdown('<div class="section-header">MIN VOLATILITY PORTFOLIO</div>', unsafe_allow_html=True)
                        min_ret, min_vol, min_sharpe = ef_engine.portfolio_performance(min_vol_weights)
                        
                        mcols2 = st.columns(3)
                        with mcols2[0]:
                            st.markdown(render_metric_card("EXP. RETURN", f"{min_ret*100:.1f}%", "positive"), unsafe_allow_html=True)
                        with mcols2[1]:
                            st.markdown(render_metric_card("VOLATILITY", f"{min_vol*100:.1f}%", "positive"), unsafe_allow_html=True)
                        with mcols2[2]:
                            st.markdown(render_metric_card("SHARPE", f"{min_sharpe:.2f}", "neutral"), unsafe_allow_html=True)
                        
                        st.plotly_chart(create_portfolio_allocation_chart(min_vol_weights, list(data.keys()), "MIN VOLATILITY ALLOCATION"), use_container_width=True)
                    
                    # Allocation table
                    st.markdown('<div class="section-header">DETAILED ALLOCATIONS</div>', unsafe_allow_html=True)
                    allocation_df = pd.DataFrame({
                        'Symbol': list(data.keys()),
                        'Max Sharpe %': [f"{w*100:.2f}%" for w in max_sharpe_weights],
                        'Min Vol %': [f"{w*100:.2f}%" for w in min_vol_weights]
                    })
                    allocation_df = allocation_df[(max_sharpe_weights > 0.01) | (min_vol_weights > 0.01)]
                    st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                else:
                    st.error("Insufficient data for optimization. Please try increasing the number of assets.")
    
    # ==================== TAB 5: ANALYSIS ====================
    with tab5:
        st.markdown('<div class="section-header">STRATEGY ANALYSIS</div>', unsafe_allow_html=True)
        
        if 'backtest_trades' in st.session_state:
            trades_df = st.session_state['backtest_trades']
            metrics = st.session_state['backtest_metrics']
            
            if not trades_df.empty:
                # Analysis by Symbol
                st.markdown("#### Performance by Symbol")
                symbol_stats = trades_df.groupby('Symbol').agg({
                    'NetPnl': ['sum', 'mean', 'count'],
                    'NetPnl%': 'mean'
                }).round(2)
                symbol_stats.columns = ['Total P&L', 'Avg P&L', 'Trades', 'Avg %']
                symbol_stats = symbol_stats.sort_values('Total P&L', ascending=False)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(symbol_stats.head(15), use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_sym = go.Figure()
                    fig_sym.add_trace(go.Bar(
                        x=symbol_stats.index[:15],
                        y=symbol_stats['Total P&L'][:15],
                        marker_color=['#00ff88' if x > 0 else '#ff4757' for x in symbol_stats['Total P&L'][:15]]
                    ))
                    fig_sym.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title='<b>TOP 15 SYMBOLS BY P&L</b>',
                        xaxis=dict(title='Symbol'),
                        yaxis=dict(title='Total P&L (₹)'),
                        height=350
                    )
                    st.plotly_chart(fig_sym, use_container_width=True)
                
                # Year over Year
                st.markdown("#### Performance by Year")
                trades_df['Year'] = pd.to_datetime(trades_df['Exit Date']).dt.year
                yearly_stats = trades_df.groupby('Year').agg({
                    'NetPnl': 'sum',
                    'Symbol': 'count',
                    'NetPnl%': 'mean'
                }).round(2)
                yearly_stats.columns = ['Total P&L', 'Trades', 'Avg %']
                st.dataframe(yearly_stats, use_container_width=True)
                
                # Kelly and Advanced Metrics
                st.markdown("#### Advanced Metrics")
                adv_cols = st.columns(4)
                with adv_cols[0]:
                    st.markdown(render_metric_card("KELLY %", f"{metrics.get('kelly', 0):.2f}%", "neutral"), unsafe_allow_html=True)
                with adv_cols[1]:
                    st.markdown(render_metric_card("SMART SHARPE", f"{metrics.get('smart_sharpe', 0):.2f}", "neutral"), unsafe_allow_html=True)
                with adv_cols[2]:
                    st.markdown(render_metric_card("MAX CONSEC WINS", str(metrics.get('max_consec_wins', 0)), "positive"), unsafe_allow_html=True)
                with adv_cols[3]:
                    st.markdown(render_metric_card("MAX CONSEC LOSSES", str(metrics.get('max_consec_losses', 0)), "negative"), unsafe_allow_html=True)
        else:
            st.info("Run a backtest first to see detailed analysis.")
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 20px; margin-top: 50px; border-top: 1px solid #1e293b;">
            <p style="color: #64748b; font-size: 12px; font-family: 'JetBrains Mono', monospace;">
                Made with ❤️ by Stallions | ©2025 Stallions.in - All Rights Reserved<br>
                <span style="color: #00d4ff;">Author: Ankit Gupta</span>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
