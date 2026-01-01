"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MCAP50 QUANTITATIVE TRADING TERMINAL                      ║
║                     Institutional-Grade Analytics Engine                      ║
║                                                                              ║
║  Author: Ankit Gupta                                                         ║
║  Stallions Quantitative Research                                             ║
║  QTF Framework: Repeatable, Risk-Controlled Probabilistic Decision Engine   ║
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

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stallions Quant Terminal",
    page_icon="🐎",
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
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NIFTY MIDCAP 50 UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════
MIDCAP50_UNIVERSE = [
    'PERSISTENT', 'PRESTIGE', 'AUBANK', 'GODREJPROP', 'COFORGE', 'PHOENIXLTD', 
    'HDFCAMC', 'OBEROIRLTY', 'PAYTM', 'NHPC', 'TIINDIA', 'INDUSTOWER', 'OIL', 
    'IRCTC', 'SBICARD', 'BHEL', 'MPHASIS', 'MUTHOOTFIN', 'DABUR', 'GMRAIRPORT', 
    'COLPAL', 'SRF', 'HINDPETRO', 'UPL', 'FORTIS', 'SUPREMEIND', 'DIXON', 
    'POLYCAB', 'NMDC', 'BHARATFORG', 'PAGEIND', 'JUBLFOOD', 'FEDERALBNK', 
    'APLAPOLLO', 'CUMMINSIND', 'BSE', 'ASHOKLEY', 'IDFCFIRSTB', 'YESBANK', 
    'LUPIN', 'MANKIND', 'PIIND', 'SUZLON', 'MARICO', 'MFSL', 'HEROMOTOCO', 
    'AUROPHARMA', 'INDUSINDBK', 'POLICYBZR', 'OFSS'
]

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

@st.cache_data(ttl=300)
def fetch_multiple_stocks(symbols: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple stocks"""
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        status_text.text(f"⚡ Fetching {symbol}...")
        df = fetch_stock_data(symbol, start_date, end_date)
        if not df.empty:
            data[symbol] = df
        progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    return data

# ══════════════════════════════════════════════════════════════════════════════
# SCREENER ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_screener(symbols: List[str], dma_window: int = 20) -> pd.DataFrame:
    """Screen stocks based on deviation from DMA"""
    end_date = date.today()
    start_date = end_date - timedelta(days=100)
    
    results = []
    data = fetch_multiple_stocks(symbols, start_date, end_date)
    
    for symbol, df in data.items():
        if len(df) < dma_window:
            continue
        
        df['DMA'] = df['Close'].rolling(window=dma_window).mean()
        df['DMA_Deviation'] = ((df['Close'] - df['DMA']) / df['DMA']) * 100
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Calculate additional metrics
        returns_5d = ((df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
        returns_20d = ((df['Close'].iloc[-1] / df['Close'].iloc[-20]) - 1) * 100 if len(df) >= 20 else 0
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        
        results.append({
            'Symbol': symbol,
            'Close': latest['Close'],
            f'{dma_window}DMA': latest['DMA'],
            'Deviation %': latest['DMA_Deviation'],
            'Signal': 'BUY' if latest['DMA_Deviation'] < 0 else 'HOLD',
            '5D Return %': returns_5d,
            '20D Return %': returns_20d,
            'Volatility %': volatility,
            'Avg Volume': avg_volume,
            'Below DMA': latest['Close'] < latest['DMA']
        })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('Deviation %', ascending=True)
    
    return df_results

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
            self.negative_sharpe, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
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
            self.portfolio_volatility, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        opt_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(opt_weights)
        return opt_weights, ret, vol, sharpe
    
    def efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """Generate efficient frontier"""
        target_returns = np.linspace(
            self.mean_returns.min(), 
            self.mean_returns.max(), 
            n_points
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
                    self.portfolio_volatility, 
                    initial_weights, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints
                )
                if result.success:
                    ret, vol, sharpe = self.portfolio_performance(result.x)
                    efficient_portfolios.append({
                        'Return': ret * 100,
                        'Volatility': vol * 100,
                        'Sharpe': sharpe
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
                'Return': ret * 100,
                'Volatility': vol * 100,
                'Sharpe': sharpe,
                'Weights': weights
            })
        
        return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (LIFO LOT TRACKING)
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


class BacktestEngine:
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
        brokerage_per_order: float = 20.0,
        dma_window: int = 20,
        max_avg: int = 3,
    ):
        self.instruments = instruments
        self.start_date = start_date
        self.end_date = end_date
        self.position_sizing_mode = position_sizing_mode
        
        self.fresh_static_amt = fresh_static_amt
        self.avg_static_amt = avg_static_amt
        self.fresh_cash_pct = fresh_cash_pct
        self.avg_cash_pct = avg_cash_pct
        self.fresh_trade_divisor = fresh_trade_divisor
        self.avg_trade_divisor = avg_trade_divisor
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.target_pct = target_pct
        self.avg_trigger_pct = avg_trigger_pct
        self.brokerage_per_order = brokerage_per_order
        self.dma_window = dma_window
        self.max_avg = max_avg
        
        self.positions: Dict[str, Position] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self._completed_trades: List[dict] = []
        self.cashflow_ledger: List[Tuple] = []
        self.realized_pnl_by_date: Dict[pd.Timestamp, float] = {}
        self.equity_curve: List[dict] = []
        
    def _portfolio_value(self, dt: pd.Timestamp) -> float:
        """Calculate total portfolio value"""
        value = self.cash
        for sym, pos in self.positions.items():
            if pos.total_qty() > 0:
                df = self.data.get(sym)
                if df is not None and dt in df.index:
                    value += pos.total_qty() * float(df.loc[dt, 'Close'])
        return value
    
    def _determine_qty_for_buy(self, trade_type: str, close_price: float, dt: pd.Timestamp) -> int:
        """Determine quantity based on position sizing mode"""
        if self.position_sizing_mode == 'static':
            amount = self.fresh_static_amt if trade_type == 'fresh' else self.avg_static_amt
        elif self.position_sizing_mode == 'dynamic':
            pct = self.fresh_cash_pct if trade_type == 'fresh' else self.avg_cash_pct
            amount = self.cash * pct
        else:  # divisor
            portfolio_val = self._portfolio_value(dt)
            divisor = self.fresh_trade_divisor if trade_type == 'fresh' else self.avg_trade_divisor
            amount = portfolio_val / divisor
        
        return int(amount // close_price)
    
    def run_backtest(self, progress_callback=None) -> pd.DataFrame:
        """Run the complete backtest"""
        # Fetch all data
        self.data = fetch_multiple_stocks(self.instruments, self.start_date, self.end_date)
        
        if not self.data:
            return pd.DataFrame()
        
        # Calculate DMA for all instruments
        for sym, df in self.data.items():
            df['DMA'] = df['Close'].rolling(window=self.dma_window).mean()
            df['Below_DMA'] = df['Close'] < df['DMA']
        
        # Get all trading days
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index.tolist())
        
        trading_days = sorted([d for d in all_dates if self.start_date <= d.date() <= self.end_date])
        
        # Main backtest loop
        for i, dt in enumerate(trading_days):
            if progress_callback:
                progress_callback((i + 1) / len(trading_days))
            
            # Process exits first
            self._process_exits(dt)
            
            # Process fresh entries
            self._process_entries(dt)
            
            # Process averaging
            self._process_averaging(dt)
            
            # Record equity
            self.equity_curve.append({
                'Date': dt,
                'Portfolio_Value': self._portfolio_value(dt),
                'Cash': self.cash
            })
        
        return pd.DataFrame(self._completed_trades)
    
    def _process_exits(self, dt: pd.Timestamp):
        """Process exits for existing positions"""
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            if pos.total_qty() == 0:
                continue
            
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            
            high_price = float(df.loc[dt, 'High'])
            avg_price = pos.avg_price()
            
            if avg_price is None:
                continue
            
            target_price = avg_price * (1 + self.target_pct)
            
            if high_price >= target_price:
                # Exit at target
                total_qty = pos.total_qty()
                total_buy_cost = sum(l.qty * l.price for l in pos.lots)
                total_buy_brokerage = pos.total_buy_brokerage()
                
                sell_value = total_qty * target_price
                sell_brokerage = self.brokerage_per_order
                
                gross_pnl = sell_value - total_buy_cost
                net_pnl = gross_pnl - total_buy_brokerage - sell_brokerage
                pnl_pct = (net_pnl / total_buy_cost) * 100
                
                self.cash += sell_value - sell_brokerage
                
                # Record trade
                self._completed_trades.append({
                    'Symbol': sym,
                    'Entry_Date': pos.lots[0].date,
                    'Exit_Date': dt,
                    'Avg_Price': avg_price,
                    'Exit_Price': target_price,
                    'Qty': total_qty,
                    'Gross_PnL': gross_pnl,
                    'Net_PnL': net_pnl,
                    'PnL_%': pnl_pct,
                    'Lots': len(pos.lots)
                })
                
                # Record cashflow
                self.cashflow_ledger.append((dt.date(), sell_value - sell_brokerage))
                
                if dt not in self.realized_pnl_by_date:
                    self.realized_pnl_by_date[dt] = 0
                self.realized_pnl_by_date[dt] += net_pnl
                
                # Clear position
                pos.lots.clear()
    
    def _process_entries(self, dt: pd.Timestamp):
        """Process fresh entries"""
        candidates = []
        
        for sym in self.instruments:
            if sym in self.positions and self.positions[sym].total_qty() > 0:
                continue
            
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            
            if pd.isna(df.loc[dt, 'DMA']):
                continue
            
            if df.loc[dt, 'Below_DMA']:
                deviation = ((df.loc[dt, 'Close'] - df.loc[dt, 'DMA']) / df.loc[dt, 'DMA']) * 100
                candidates.append((sym, deviation, float(df.loc[dt, 'Close'])))
        
        # Sort by deviation and take top 5
        candidates.sort(key=lambda x: x[1])
        
        for sym, deviation, close_price in candidates[:5]:
            qty = self._determine_qty_for_buy('fresh', close_price, dt)
            if qty <= 0:
                continue
            
            total_cost = qty * close_price + self.brokerage_per_order
            if total_cost > self.cash:
                continue
            
            self.cash -= total_cost
            
            if sym not in self.positions:
                self.positions[sym] = Position(symbol=sym)
            
            lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
            self.positions[sym].lots.append(lot)
            
            self.cashflow_ledger.append((dt.date(), -total_cost))
    
    def _process_averaging(self, dt: pd.Timestamp):
        """Process averaging for existing positions"""
        for sym, pos in self.positions.items():
            if pos.total_qty() == 0:
                continue
            
            if len(pos.lots) >= self.max_avg:
                continue
            
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            
            close_price = float(df.loc[dt, 'Close'])
            last_buy_price = pos.last_buy_price()
            
            if last_buy_price is None:
                continue
            
            pct_drop = (last_buy_price - close_price) / last_buy_price
            
            if pct_drop > self.avg_trigger_pct:
                qty = self._determine_qty_for_buy('avg', close_price, dt)
                if qty <= 0:
                    continue
                
                total_cost = qty * close_price + self.brokerage_per_order
                if total_cost > self.cash:
                    continue
                
                self.cash -= total_cost
                
                lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
                pos.lots.append(lot)
                
                self.cashflow_ledger.append((dt.date(), -total_cost))
                break  # One averaging per day
    
    def get_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""
        if not self._completed_trades:
            return {}
        
        trades_df = pd.DataFrame(self._completed_trades)
        starting_balance = self.initial_capital
        total_net_pnl = trades_df['Net_PnL'].sum()
        ending_balance = starting_balance + total_net_pnl
        
        # Time metrics
        days = (self.end_date - self.start_date).days
        years = days / 365.25
        
        # Returns
        total_return = (ending_balance / starting_balance - 1) * 100
        cagr = ((ending_balance / starting_balance) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Win/Loss metrics
        wins = trades_df[trades_df['Net_PnL'] > 0]
        losses = trades_df[trades_df['Net_PnL'] <= 0]
        win_rate = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        # Risk metrics from equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['Returns'] = equity_df['Portfolio_Value'].pct_change()
            equity_df['Cummax'] = equity_df['Portfolio_Value'].cummax()
            equity_df['Drawdown'] = (equity_df['Portfolio_Value'] - equity_df['Cummax']) / equity_df['Cummax'] * 100
            
            max_drawdown = equity_df['Drawdown'].min()
            daily_returns = equity_df['Returns'].dropna()
            
            sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            
            neg_returns = daily_returns[daily_returns < 0]
            sortino = (daily_returns.mean() / neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0
            
            calmar = (cagr / abs(max_drawdown)) if max_drawdown != 0 else 0
        else:
            max_drawdown = 0
            sharpe = 0
            sortino = 0
            calmar = 0
        
        return {
            'starting_balance': starting_balance,
            'ending_balance': ending_balance,
            'total_net_pnl': total_net_pnl,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'avg_win': wins['Net_PnL'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['Net_PnL'].mean() if len(losses) > 0 else 0,
            'avg_win_pct': wins['PnL_%'].mean() if len(wins) > 0 else 0,
            'avg_loss_pct': losses['PnL_%'].mean() if len(losses) > 0 else 0,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'profit_factor': abs(wins['Net_PnL'].sum() / losses['Net_PnL'].sum()) if len(losses) > 0 and losses['Net_PnL'].sum() != 0 else float('inf'),
            'years': years
        }

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def create_equity_curve(equity_data: List[dict], benchmark_data: pd.DataFrame = None) -> go.Figure:
    """Create equity curve chart"""
    df = pd.DataFrame(equity_data)
    
    fig = go.Figure()
    
    # Portfolio equity curve
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Portfolio_Value'],
        name='Strategy',
        line=dict(color='#00d4ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Initial capital line
    fig.add_hline(
        y=df['Portfolio_Value'].iloc[0],
        line_dash='dash',
        line_color='#64748b',
        annotation_text='Initial Capital'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='<b>EQUITY CURVE</b>',
            font=dict(family='JetBrains Mono', size=14, color='#00d4ff')
        ),
        xaxis=dict(
            gridcolor='rgba(30, 41, 59, 0.5)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(30, 41, 59, 0.5)',
            showgrid=True,
            title='Portfolio Value (₹)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(family='JetBrains Mono', size=10)
        ),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

def create_drawdown_chart(equity_data: List[dict]) -> go.Figure:
    """Create drawdown underwater chart"""
    df = pd.DataFrame(equity_data)
    df['Cummax'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] - df['Cummax']) / df['Cummax'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Drawdown'],
        name='Drawdown',
        line=dict(color='#ff4757', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 71, 87, 0.3)'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='<b>DRAWDOWN - UNDERWATER PLOT</b>',
            font=dict(family='JetBrains Mono', size=14, color='#ff4757')
        ),
        xaxis=dict(gridcolor='rgba(30, 41, 59, 0.5)'),
        yaxis=dict(
            gridcolor='rgba(30, 41, 59, 0.5)',
            title='Drawdown %'
        ),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

def create_monthly_returns_heatmap(trades_df: pd.DataFrame) -> go.Figure:
    """Create monthly returns heatmap"""
    if trades_df.empty:
        return go.Figure()
    
    trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
    trades_df['Year'] = trades_df['Exit_Date'].dt.year
    trades_df['Month'] = trades_df['Exit_Date'].dt.month
    
    monthly_pnl = trades_df.groupby(['Year', 'Month'])['Net_PnL'].sum().unstack(fill_value=0)
    
    # Convert to returns percentage (assuming starting capital)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=monthly_pnl.values,
        x=months[:monthly_pnl.shape[1]] if monthly_pnl.shape[1] <= 12 else list(range(1, monthly_pnl.shape[1]+1)),
        y=monthly_pnl.index.astype(str).tolist(),
        colorscale=[
            [0, '#ff4757'],
            [0.5, '#1a2332'],
            [1, '#00ff88']
        ],
        text=np.round(monthly_pnl.values / 1000, 1),
        texttemplate='%{text}K',
        textfont=dict(family='JetBrains Mono', size=10),
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            title=dict(text='PnL (₹)', font=dict(family='JetBrains Mono', size=10)),
            tickfont=dict(family='JetBrains Mono', size=9)
        )
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='<b>MONTHLY P&L HEATMAP</b>',
            font=dict(family='JetBrains Mono', size=14, color='#00d4ff')
        ),
        xaxis=dict(title='Month', tickfont=dict(family='JetBrains Mono')),
        yaxis=dict(title='Year', tickfont=dict(family='JetBrains Mono')),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

def create_efficient_frontier_plot(ef_engine: EfficientFrontierEngine) -> go.Figure:
    """Create efficient frontier visualization"""
    # Monte Carlo simulation
    mc_results = ef_engine.monte_carlo_simulation(3000)
    
    # Efficient frontier
    ef_results = ef_engine.efficient_frontier(50)
    
    # Optimal portfolios
    max_sharpe_weights, max_sharpe_ret, max_sharpe_vol, max_sharpe = ef_engine.optimize_sharpe()
    min_vol_weights, min_vol_ret, min_vol_vol, min_vol_sharpe = ef_engine.min_volatility()
    
    fig = go.Figure()
    
    # Monte Carlo points
    fig.add_trace(go.Scatter(
        x=mc_results['Volatility'],
        y=mc_results['Return'],
        mode='markers',
        marker=dict(
            size=4,
            color=mc_results['Sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(text='Sharpe', font=dict(family='JetBrains Mono', size=10)),
                tickfont=dict(family='JetBrains Mono', size=9)
            ),
            opacity=0.6
        ),
        name='Random Portfolios',
        hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Efficient frontier line
    if not ef_results.empty:
        fig.add_trace(go.Scatter(
            x=ef_results['Volatility'],
            y=ef_results['Return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#00d4ff', width=3)
        ))
    
    # Maximum Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_vol * 100],
        y=[max_sharpe_ret * 100],
        mode='markers',
        marker=dict(size=20, color='#00ff88', symbol='star'),
        name=f'Max Sharpe ({max_sharpe:.2f})'
    ))
    
    # Minimum volatility portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol_vol * 100],
        y=[min_vol_ret * 100],
        mode='markers',
        marker=dict(size=15, color='#ff9f43', symbol='diamond'),
        name=f'Min Volatility ({min_vol_vol*100:.1f}%)'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='<b>EFFICIENT FRONTIER - PORTFOLIO OPTIMIZATION</b>',
            font=dict(family='JetBrains Mono', size=14, color='#00d4ff')
        ),
        xaxis=dict(
            title=dict(text='Volatility (%)', font=dict(family='JetBrains Mono')),
            gridcolor='rgba(30, 41, 59, 0.5)',
            tickfont=dict(family='JetBrains Mono')
        ),
        yaxis=dict(
            title=dict(text='Expected Return (%)', font=dict(family='JetBrains Mono')),
            gridcolor='rgba(30, 41, 59, 0.5)',
            tickfont=dict(family='JetBrains Mono')
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(family='JetBrains Mono', size=10),
            orientation='h',
            yanchor='bottom',
            y=-0.3
        ),
        margin=dict(l=50, r=20, t=50, b=100)
    )
    
    return fig, max_sharpe_weights, min_vol_weights

def create_portfolio_allocation_chart(weights: np.ndarray, symbols: List[str], title: str) -> go.Figure:
    """Create portfolio allocation pie chart"""
    # Filter out zero weights
    non_zero_mask = weights > 0.01
    filtered_weights = weights[non_zero_mask]
    filtered_symbols = [s for s, m in zip(symbols, non_zero_mask) if m]
    
    colors = px.colors.sequential.Viridis[:len(filtered_weights)]
    
    fig = go.Figure(data=[go.Pie(
        labels=filtered_symbols,
        values=filtered_weights * 100,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(family='JetBrains Mono', size=10),
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(family='JetBrains Mono', size=12, color='#00d4ff')
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    """Render terminal header"""
    st.markdown("""
        <div class="terminal-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="terminal-title">◆ STALLIONS QUANT TERMINAL</h1>
                    <p class="terminal-subtitle">Institutional-Grade Mean Reversion Analytics • QTF Framework • Author: Ankit Gupta</p>
                </div>
                <div class="live-indicator">
                    <span class="live-dot"></span>
                    <span>SYSTEM ACTIVE</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, status: str = "neutral"):
    """Render a metric card"""
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {status}">{value}</div>
        </div>
    """

def main():
    render_header()
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ CONFIGURATION")
        
        mode = st.selectbox(
            "Operation Mode",
            ["📊 Screener", "🔄 Backtest", "📈 Efficient Frontier", "📋 Analysis"],
            index=0
        )
        
        st.markdown("---")
        
        if mode == "📊 Screener":
            st.markdown("#### Screener Settings")
            dma_window = st.slider("DMA Window", 5, 50, 20)
            top_n = st.slider("Top N Stocks", 3, 15, 5)
            
        elif mode == "🔄 Backtest":
            st.markdown("#### Backtest Period")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", date(2020, 1, 1))
            with col2:
                end_date = st.date_input("End Date", date.today())
            
            st.markdown("#### Position Sizing")
            sizing_mode = st.selectbox(
                "Sizing Mode",
                ["divisor", "static", "dynamic"]
            )
            
            if sizing_mode == "divisor":
                fresh_divisor = st.number_input("Fresh Trade Divisor", 10, 100, 40)
                avg_divisor = st.number_input("Avg Trade Divisor", 5, 50, 10)
            elif sizing_mode == "static":
                fresh_static = st.number_input("Fresh Trade Amount (₹)", 5000, 100000, 10000)
                avg_static = st.number_input("Avg Trade Amount (₹)", 2500, 50000, 5000)
            else:
                fresh_pct = st.slider("Fresh Trade % of Cash", 1, 20, 4) / 100
                avg_pct = st.slider("Avg Trade % of Cash", 1, 15, 3) / 100
            
            st.markdown("#### Strategy Parameters")
            initial_capital = st.number_input("Initial Capital (₹)", 100000, 10000000, 400000)
            target_pct = st.slider("Target %", 3, 15, 8) / 100
            avg_trigger = st.slider("Avg Trigger %", 2, 10, 5) / 100
            max_positions = st.slider("Max Positions per Stock", 1, 5, 3)
            dma_window = st.slider("DMA Window", 10, 50, 20)
            
        elif mode == "📈 Efficient Frontier":
            st.markdown("#### Optimization Settings")
            lookback_days = st.slider("Lookback Period (Days)", 60, 500, 252)
            risk_free = st.slider("Risk-Free Rate %", 0, 10, 6) / 100
            num_assets = st.slider("Number of Assets", 5, 50, 15)
            
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; opacity: 0.6; font-size: 0.75rem; font-family: 'JetBrains Mono';">
                Stallions QTF Framework v2.0<br>
                Risk-Controlled Probabilistic Engine<br>
                <span style="color: #00d4ff;">Author: Ankit Gupta</span>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    if mode == "📊 Screener":
        st.markdown('<div class="section-header">REAL-TIME STOCK SCREENER</div>', unsafe_allow_html=True)
        
        if st.button("🔍 RUN SCREENER", use_container_width=True):
            with st.spinner("Scanning NIFTY Midcap 50 universe..."):
                screener_df = run_screener(MIDCAP50_UNIVERSE, dma_window)
            
            if not screener_df.empty:
                # Top metrics
                below_dma_count = screener_df['Below DMA'].sum()
                avg_deviation = screener_df['Deviation %'].mean()
                buy_signals = len(screener_df[screener_df['Signal'] == 'BUY'])
                
                cols = st.columns(4)
                with cols[0]:
                    st.markdown(render_metric_card("STOCKS SCANNED", str(len(screener_df)), "neutral"), unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(render_metric_card("BELOW DMA", str(int(below_dma_count)), "positive" if below_dma_count > 0 else "neutral"), unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(render_metric_card("AVG DEVIATION", f"{avg_deviation:.2f}%", "negative" if avg_deviation < 0 else "positive"), unsafe_allow_html=True)
                with cols[3]:
                    st.markdown(render_metric_card("BUY SIGNALS", str(buy_signals), "positive"), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Top picks
                st.markdown('<div class="section-header">TOP PICKS - MAXIMUM DEVIATION FROM DMA</div>', unsafe_allow_html=True)
                top_picks = screener_df[screener_df['Below DMA']].head(top_n)
                
                if not top_picks.empty:
                    for idx, row in top_picks.iterrows():
                        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
                        with col1:
                            st.markdown(f"**{row['Symbol']}**")
                        with col2:
                            st.markdown(f"₹{row['Close']:.2f}")
                        with col3:
                            st.markdown(f"DMA: ₹{row[f'{dma_window}DMA']:.2f}")
                        with col4:
                            color = "🔴" if row['Deviation %'] < -5 else "🟡" if row['Deviation %'] < 0 else "🟢"
                            st.markdown(f"{color} {row['Deviation %']:.2f}%")
                        with col5:
                            st.markdown(f"<span class='badge badge-success'>BUY</span>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Full screener table
                st.markdown('<div class="section-header">COMPLETE SCREENER RESULTS</div>', unsafe_allow_html=True)
                
                display_df = screener_df.copy()
                display_df['Close'] = display_df['Close'].apply(lambda x: f"₹{x:,.2f}")
                display_df[f'{dma_window}DMA'] = display_df[f'{dma_window}DMA'].apply(lambda x: f"₹{x:,.2f}")
                display_df['Deviation %'] = display_df['Deviation %'].apply(lambda x: f"{x:.2f}%")
                display_df['5D Return %'] = display_df['5D Return %'].apply(lambda x: f"{x:.2f}%")
                display_df['20D Return %'] = display_df['20D Return %'].apply(lambda x: f"{x:.2f}%")
                display_df['Volatility %'] = display_df['Volatility %'].apply(lambda x: f"{x:.1f}%")
                display_df['Avg Volume'] = display_df['Avg Volume'].apply(lambda x: f"{x/1e6:.2f}M")
                
                st.dataframe(
                    display_df.drop(columns=['Below DMA']),
                    use_container_width=True,
                    hide_index=True
                )
                
    elif mode == "🔄 Backtest":
        st.markdown('<div class="section-header">BACKTEST ENGINE</div>', unsafe_allow_html=True)
        
        if st.button("🚀 RUN BACKTEST", use_container_width=True):
            # Initialize backtest
            if sizing_mode == "divisor":
                engine = BacktestEngine(
                    instruments=MIDCAP50_UNIVERSE,
                    start_date=start_date,
                    end_date=end_date,
                    position_sizing_mode="divisor",
                    fresh_trade_divisor=fresh_divisor,
                    avg_trade_divisor=avg_divisor,
                    initial_capital=initial_capital,
                    target_pct=target_pct,
                    avg_trigger_pct=avg_trigger,
                    max_avg=max_positions,
                    dma_window=dma_window
                )
            elif sizing_mode == "static":
                engine = BacktestEngine(
                    instruments=MIDCAP50_UNIVERSE,
                    start_date=start_date,
                    end_date=end_date,
                    position_sizing_mode="static",
                    fresh_static_amt=fresh_static,
                    avg_static_amt=avg_static,
                    initial_capital=initial_capital,
                    target_pct=target_pct,
                    avg_trigger_pct=avg_trigger,
                    max_avg=max_positions,
                    dma_window=dma_window
                )
            else:
                engine = BacktestEngine(
                    instruments=MIDCAP50_UNIVERSE,
                    start_date=start_date,
                    end_date=end_date,
                    position_sizing_mode="dynamic",
                    fresh_cash_pct=fresh_pct,
                    avg_cash_pct=avg_pct,
                    initial_capital=initial_capital,
                    target_pct=target_pct,
                    avg_trigger_pct=avg_trigger,
                    max_avg=max_positions,
                    dma_window=dma_window
                )
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            def update_progress(p):
                progress_bar.progress(p)
                status.text(f"Processing... {p*100:.0f}%")
            
            trades_df = engine.run_backtest(progress_callback=update_progress)
            progress_bar.empty()
            status.empty()
            
            if not trades_df.empty:
                metrics = engine.get_performance_metrics()
                
                # Store in session state
                st.session_state['backtest_trades'] = trades_df
                st.session_state['backtest_metrics'] = metrics
                st.session_state['backtest_equity'] = engine.equity_curve
                
                # Key Metrics Row 1
                cols = st.columns(5)
                with cols[0]:
                    st.markdown(render_metric_card("TOTAL RETURN", f"{metrics['total_return']:.1f}%", "positive" if metrics['total_return'] > 0 else "negative"), unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(render_metric_card("CAGR", f"{metrics['cagr']:.1f}%", "positive" if metrics['cagr'] > 0 else "negative"), unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(render_metric_card("WIN RATE", f"{metrics['win_rate']:.1f}%", "positive" if metrics['win_rate'] > 50 else "negative"), unsafe_allow_html=True)
                with cols[3]:
                    st.markdown(render_metric_card("MAX DRAWDOWN", f"{metrics['max_drawdown']:.2f}%", "negative" if metrics['max_drawdown'] < -10 else "neutral"), unsafe_allow_html=True)
                with cols[4]:
                    st.markdown(render_metric_card("SHARPE", f"{metrics['sharpe']:.2f}", "positive" if metrics['sharpe'] > 1 else "neutral"), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Key Metrics Row 2
                cols2 = st.columns(5)
                with cols2[0]:
                    st.markdown(render_metric_card("TOTAL TRADES", str(metrics['total_trades']), "neutral"), unsafe_allow_html=True)
                with cols2[1]:
                    st.markdown(render_metric_card("NET P&L", f"₹{metrics['total_net_pnl']/1e5:.2f}L", "positive" if metrics['total_net_pnl'] > 0 else "negative"), unsafe_allow_html=True)
                with cols2[2]:
                    st.markdown(render_metric_card("SORTINO", f"{metrics['sortino']:.2f}", "positive" if metrics['sortino'] > 1 else "neutral"), unsafe_allow_html=True)
                with cols2[3]:
                    st.markdown(render_metric_card("CALMAR", f"{metrics['calmar']:.2f}", "positive" if metrics['calmar'] > 1 else "neutral"), unsafe_allow_html=True)
                with cols2[4]:
                    pf = metrics['profit_factor']
                    pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
                    st.markdown(render_metric_card("PROFIT FACTOR", pf_str, "positive" if pf > 1.5 else "neutral"), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Charts
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.plotly_chart(create_equity_curve(engine.equity_curve), use_container_width=True)
                    st.plotly_chart(create_monthly_returns_heatmap(trades_df), use_container_width=True)
                
                with col_right:
                    st.plotly_chart(create_drawdown_chart(engine.equity_curve), use_container_width=True)
                    
                    # Win/Loss distribution
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=trades_df['PnL_%'],
                        nbinsx=50,
                        marker_color='#00d4ff',
                        opacity=0.7,
                        name='P&L Distribution'
                    ))
                    fig_dist.add_vline(x=0, line_dash='dash', line_color='#ff4757')
                    fig_dist.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title=dict(text='<b>P&L DISTRIBUTION</b>', font=dict(family='JetBrains Mono', size=14, color='#00d4ff')),
                        xaxis=dict(title='P&L %', gridcolor='rgba(30, 41, 59, 0.5)'),
                        yaxis=dict(title='Frequency', gridcolor='rgba(30, 41, 59, 0.5)'),
                        margin=dict(l=50, r=20, t=50, b=50)
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Trade Book
                st.markdown('<div class="section-header">TRADE BOOK</div>', unsafe_allow_html=True)
                
                display_trades = trades_df.copy()
                display_trades['Entry_Date'] = pd.to_datetime(display_trades['Entry_Date']).dt.strftime('%Y-%m-%d')
                display_trades['Exit_Date'] = pd.to_datetime(display_trades['Exit_Date']).dt.strftime('%Y-%m-%d')
                display_trades['Avg_Price'] = display_trades['Avg_Price'].apply(lambda x: f"₹{x:.2f}")
                display_trades['Exit_Price'] = display_trades['Exit_Price'].apply(lambda x: f"₹{x:.2f}")
                display_trades['Net_PnL'] = display_trades['Net_PnL'].apply(lambda x: f"₹{x:,.0f}")
                display_trades['PnL_%'] = display_trades['PnL_%'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(display_trades, use_container_width=True, hide_index=True)
                
                # Download button
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    label="📥 DOWNLOAD TRADE BOOK",
                    data=csv,
                    file_name=f"mcap50_backtest_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No trades were executed during the backtest period.")
    
    elif mode == "📈 Efficient Frontier":
        st.markdown('<div class="section-header">EFFICIENT FRONTIER OPTIMIZATION</div>', unsafe_allow_html=True)
        
        if st.button("🎯 RUN OPTIMIZATION", use_container_width=True):
            with st.spinner("Fetching data and computing efficient frontier..."):
                end_date = date.today()
                start_date = end_date - timedelta(days=lookback_days)
                
                # Fetch data for optimization
                selected_symbols = MIDCAP50_UNIVERSE[:num_assets]
                data = fetch_multiple_stocks(selected_symbols, start_date, end_date)
                
                if len(data) >= 3:
                    # Build returns dataframe
                    closes = pd.DataFrame({sym: df['Close'] for sym, df in data.items()})
                    returns = closes.pct_change().dropna()
                    
                    # Run optimization
                    ef_engine = EfficientFrontierEngine(returns, risk_free)
                    
                    fig_ef, max_sharpe_weights, min_vol_weights = create_efficient_frontier_plot(ef_engine)
                    
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                    # Display optimal portfolios
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
                        
                        st.plotly_chart(
                            create_portfolio_allocation_chart(max_sharpe_weights, list(data.keys()), "MAX SHARPE ALLOCATION"),
                            use_container_width=True
                        )
                    
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
                        
                        st.plotly_chart(
                            create_portfolio_allocation_chart(min_vol_weights, list(data.keys()), "MIN VOLATILITY ALLOCATION"),
                            use_container_width=True
                        )
                    
                    # Allocation table
                    st.markdown('<div class="section-header">DETAILED ALLOCATIONS</div>', unsafe_allow_html=True)
                    
                    allocation_df = pd.DataFrame({
                        'Symbol': list(data.keys()),
                        'Max Sharpe %': [f"{w*100:.2f}%" for w in max_sharpe_weights],
                        'Min Vol %': [f"{w*100:.2f}%" for w in min_vol_weights]
                    })
                    allocation_df = allocation_df[
                        (max_sharpe_weights > 0.01) | (min_vol_weights > 0.01)
                    ]
                    
                    st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                else:
                    st.error("Insufficient data for optimization. Please try increasing the number of assets.")
    
    elif mode == "📋 Analysis":
        st.markdown('<div class="section-header">STRATEGY ANALYSIS</div>', unsafe_allow_html=True)
        
        if 'backtest_trades' in st.session_state:
            trades_df = st.session_state['backtest_trades']
            metrics = st.session_state['backtest_metrics']
            
            # Detailed analysis
            tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🎯 By Symbol", "📅 By Period"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Returns Statistics")
                    stats_data = {
                        'Metric': ['Total Return', 'CAGR', 'Best Month', 'Worst Month', 'Avg Monthly', 'Volatility (Ann.)'],
                        'Value': [
                            f"{metrics['total_return']:.2f}%",
                            f"{metrics['cagr']:.2f}%",
                            f"{trades_df.groupby(pd.to_datetime(trades_df['Exit_Date']).dt.to_period('M'))['Net_PnL'].sum().max()/1000:.1f}K",
                            f"{trades_df.groupby(pd.to_datetime(trades_df['Exit_Date']).dt.to_period('M'))['Net_PnL'].sum().min()/1000:.1f}K",
                            f"{trades_df.groupby(pd.to_datetime(trades_df['Exit_Date']).dt.to_period('M'))['Net_PnL'].sum().mean()/1000:.1f}K",
                            f"{trades_df['PnL_%'].std() * np.sqrt(12):.2f}%"
                        ]
                    }
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### Risk Statistics")
                    risk_data = {
                        'Metric': ['Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Win Rate', 'Profit Factor'],
                        'Value': [
                            f"{metrics['max_drawdown']:.2f}%",
                            f"{metrics['sharpe']:.2f}",
                            f"{metrics['sortino']:.2f}",
                            f"{metrics['calmar']:.2f}",
                            f"{metrics['win_rate']:.1f}%",
                            f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞"
                        ]
                    }
                    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)
            
            with tab2:
                st.markdown("#### Performance by Symbol")
                symbol_stats = trades_df.groupby('Symbol').agg({
                    'Net_PnL': ['sum', 'mean', 'count'],
                    'PnL_%': 'mean'
                }).round(2)
                symbol_stats.columns = ['Total P&L', 'Avg P&L', 'Trades', 'Avg %']
                symbol_stats = symbol_stats.sort_values('Total P&L', ascending=False)
                
                st.dataframe(symbol_stats, use_container_width=True)
                
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
                    yaxis=dict(title='Total P&L (₹)')
                )
                st.plotly_chart(fig_sym, use_container_width=True)
            
            with tab3:
                st.markdown("#### Performance by Year")
                trades_df['Year'] = pd.to_datetime(trades_df['Exit_Date']).dt.year
                yearly_stats = trades_df.groupby('Year').agg({
                    'Net_PnL': 'sum',
                    'Symbol': 'count',
                    'PnL_%': 'mean'
                }).round(2)
                yearly_stats.columns = ['Total P&L', 'Trades', 'Avg %']
                
                st.dataframe(yearly_stats, use_container_width=True)
        else:
            st.info("Run a backtest first to see detailed analysis.")

if __name__ == "__main__":
    main()
