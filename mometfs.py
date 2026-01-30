"""
Quant ETFs Momentum Dashboard
=============================

Portfolio Optimization Methods (PyPortfolioOpt):
- Mean-Variance Optimization (MVO) - Maximize Sharpe
- Tangency Portfolio (CAL) - Optimal Risky + Risk-Free allocation
- Global Minimum Variance (GMVP) - Minimize volatility
- Hierarchical Risk Parity (HRP) - Clustering-based
- Risk Parity - Equal risk contribution
- Black-Litterman - Incorporate views
- Mean-Semivariance - Downside risk focus
- Inverse Volatility (IVP) - Weight by 1/vol
- Most Diversified Portfolio (MDP) - Max diversification ratio
- Equal Weight (EWP) - Simple 1/N
- Ulcer Performance Index (UPI) - Drawdown-based optimization

Rebalance Frequencies:
- Buy & Hold - No rebalancing
- Fortnightly - Every 2 weeks
- Monthly - End of month
- Quarterly - End of quarter

Install: pip install streamlit yfinance pandas numpy plotly quantstats pypfopt scikit-learn
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import quantstats as qs
import matplotlib
matplotlib.use('Agg')
import warnings
from datetime import datetime, timedelta, date
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize

# PyPortfolioOpt imports
try:
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import HRPOpt
    from pypfopt import black_litterman
    from pypfopt import BlackLittermanModel
    from pypfopt import objective_functions
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Constants
MOMENTUM_LOOKBACK = 126
MIN_MOMENTUM_THRESHOLD = 0.03
RISK_FREE_RATE_DEFAULT = 0.065

# --- ENUMS ---
class AssetSelectionModel(Enum):
    HIGH_CONVICTION = "High Conviction (Top Sharpe + Momentum)"
    LOW_VOLATILITY = "Low Volatility (< 25% Annual)"
    MOMENTUM_126D = "Momentum (126-Day Returns > 3%)"
    QUALITY_MOMENTUM = "Quality Momentum (Combined Score)"
    RELATIVE_STRENGTH = "Relative Strength vs Benchmark"
    RISK_ADJUSTED = "Risk-Adjusted (Sharpe > 0.5)"
    TREND_FOLLOWING = "Trend Following (MA Crossover)"

class RebalanceFrequency(Enum):
    BUY_HOLD = "Buy & Hold"
    FORTNIGHTLY = "Fortnightly"
    MONTHLY = "Monthly"
    QUARTERLY = "Quarterly"

# All Portfolio Optimization Methods (Alphabetically sorted)
OPTIMIZATION_METHODS = [
    "Black-Litterman",
    "Equal Weight (EWP)",
    "Global Minimum Variance (GMVP)",
    "Hierarchical Risk Parity (HRP)",
    "Inverse Volatility (IVP)",
    "Mean-Semivariance (Downside Risk)",
    "Mean-Variance (Max Sharpe)",
    "Most Diversified Portfolio (MDP)",
    "Risk Parity (Equal Risk Contribution)",
    "Tangency Portfolio (CAL)",
    "Ulcer Performance Index (UPI)"
]

@dataclass
class AssetScore:
    ticker: str
    momentum_126d: float
    momentum_63d: float
    volatility: float
    sharpe_ratio: float
    trend_score: float
    relative_strength: float
    composite_score: float
    ulcer_index: float = 0.0
    upi: float = 0.0
    signal: str = ""

@dataclass
class OpportunitySignal:
    ticker: str
    signal_type: str
    strength: float
    momentum_126d: float
    sharpe_ratio: float
    trend_status: str
    support_level: float
    resistance_level: float
    current_price: float
    target_price: float
    stop_loss: float
    rationale: str
    weight: float = 0.0
    ulcer_index: float = 0.0
    upi: float = 0.0
    max_drawdown: float = 0.0

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Quant ETFs Momentum Dashboard", 
    layout="wide", 
    page_icon="⚡"
)

# CSS with colorful 3D button theme, enhanced sliders, and multi-color tables
st.markdown("""
<style>
    .stApp { background-color: #0a0d12; }
    [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #1e2530; }
    
    /* Metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #151922, #1a1f2c);
        border: 1px solid #2a3545; padding: 15px; border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.15);
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00ff88 !important; font-family: 'SF Mono', monospace;
    }
    
    /* Headers */
    h1, h2, h3, h4 { color: #f0f4f8; font-weight: 600; }
    h1 { background: linear-gradient(90deg, #00ff88, #00d4ff);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Colorful 3D Tab Buttons */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
        background: linear-gradient(180deg, #0d1117, #161b22);
        padding: 10px 15px; 
        border-radius: 15px;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
    }
    
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        min-width: 140px;
        background: linear-gradient(180deg, #1a1f2c, #0d1117);
        border-radius: 12px; 
        color: #8892a0;
        border: 2px solid transparent;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Tab 1 - Green glow */
    .stTabs [data-baseweb="tab"]:nth-child(1) {
        border-color: #00ff88;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4), 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
        background: linear-gradient(180deg, #0d3320, #051510);
        color: #00ff88;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.6), inset 0 0 20px rgba(0, 255, 136, 0.1);
    }
    
    /* Tab 2 - Orange glow */
    .stTabs [data-baseweb="tab"]:nth-child(2) {
        border-color: #ff9500;
        box-shadow: 0 0 20px rgba(255, 149, 0, 0.4), 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
        background: linear-gradient(180deg, #332200, #1a1100);
        color: #ff9500;
        box-shadow: 0 0 30px rgba(255, 149, 0, 0.6), inset 0 0 20px rgba(255, 149, 0, 0.1);
    }
    
    /* Tab 3 - Purple glow */
    .stTabs [data-baseweb="tab"]:nth-child(3) {
        border-color: #a855f7;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.4), 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
        background: linear-gradient(180deg, #2d1f4a, #1a1030);
        color: #a855f7;
        box-shadow: 0 0 30px rgba(168, 85, 247, 0.6), inset 0 0 20px rgba(168, 85, 247, 0.1);
    }
    
    /* Tab 4 - Cyan glow */
    .stTabs [data-baseweb="tab"]:nth-child(4) {
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4), 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {
        background: linear-gradient(180deg, #0d2833, #051518);
        color: #00d4ff;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6), inset 0 0 20px rgba(0, 212, 255, 0.1);
    }
    
    /* Tab 5 - Pink/Magenta glow */
    .stTabs [data-baseweb="tab"]:nth-child(5) {
        border-color: #ff0080;
        box-shadow: 0 0 20px rgba(255, 0, 128, 0.4), 0 4px 15px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab"]:nth-child(5)[aria-selected="true"] {
        background: linear-gradient(180deg, #330d20, #1a0510);
        color: #ff0080;
        box-shadow: 0 0 30px rgba(255, 0, 128, 0.6), inset 0 0 20px rgba(255, 0, 128, 0.1);
    }
    
    /* Enhanced Slider Styling - Better visibility and bolder text */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #ff0080, #00ff88, #00d4ff) !important;
    }
    
    .stSlider label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, #ff0080, #a855f7, #00d4ff, #00ff88) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    .stSlider [data-testid="stTickBarMin"], 
    .stSlider [data-testid="stTickBarMax"] {
        color: #00ff88 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Slider thumb */
    .stSlider [role="slider"] {
        background: #00ff88 !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.8) !important;
    }
    
    /* Slider value display - FIXED visibility */
    .stSlider [data-testid="stThumbValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        background: rgba(0, 0, 0, 0.8) !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
        text-shadow: none !important;
    }
    
    /* Current value shown on slider */
    .stSlider div[data-baseweb="slider"] div[role="slider"]::after {
        color: #ffffff !important;
    }
    
    /* Min/Max values at ends */
    .stSlider p {
        color: #00ff88 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    
    /* Slider current value text */
    .stSlider > div > div > div > div:last-child {
        color: #ffffff !important;
        font-weight: 700 !important;
        background: rgba(0, 0, 0, 0.7) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(145deg, #1a2535, #0d1520);
        border: 2px solid #00ff88;
        color: #00ff88;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.5);
        transform: translateY(-2px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1a2535, #0d1520);
        border: 1px solid #2a3545;
        border-radius: 8px;
    }
    
    /* ========== PLOTLY CHART HOVER EFFECT ========== */
    .js-plotly-plot .plotly .scatterlayer .trace path.js-line {
        transition: stroke-width 0.15s ease, opacity 0.15s ease !important;
    }
</style>
""", unsafe_allow_html=True)

# --- BENCHMARKS ---
BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 100": "^CNX100",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 100": "^NSEMDCP50",
    "Bank Nifty": "^NSEBANK"
}

# --- ETF SYMBOLS ---
@st.cache_data(ttl=3600)
def load_etf_symbols() -> Tuple[List[str], Dict[str, str]]:
    symbols = [
        'SILVERBEES', 'GOLDBEES', 'PSUBNKBEES', 'METALIETF', 'ITBEES',
        'HNGSNGBEES', 'DIVOPPBEES', 'MASPTOP50', 'BANKBEES', 'COMMOIETF',
        'LIQUIDBEES', 'ICICIB22', 'HDFCGROWTH', 'LTGILTBEES', 'GILT5YBEES',
        'MNC', 'PVTBANIETF', 'AUTOBEES', 'FINIETF', 'CPSEETF',
        'LOWVOLIETF', 'BFSI', 'PHARMABEES', 'ABSLPSE', 'MAHKTECH',
        'TNIDETF', 'ESG', 'MAKEINDIA', 'NIFTYBEES', 'MOCAPITAL',
        'MONQ50', 'MSCIINDIA', 'OILIETF', 'JUNIORBEES', 'TOP10ADD',
        'MID150BEES', 'AONETOTAL', 'MON100', 'MOM30IETF', 'MULTICAP',
        'INFRAIETF', 'MAFANG', 'GROWWEV', 'GROWWRAIL', 'ALPHA',
        'HEALTHIETF', 'MODEFENCE', 'MIDSMALL', 'MOMENTUM50', 'CONSUMBEES',
        'FMCGIETF', 'HDFCSML250', 'CONSUMER', 'SELECTIPO', 'MOREALTY'
    ]
    return symbols, {s: f"{s}.NS" for s in symbols}

def clean_ticker(ticker: str) -> str:
    return ticker.replace('.NS', '').replace('.BO', '').replace('^', '')

# --- DATA FETCHING ---
def fetch_data(tickers: List[str], start: str, end: str) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch price data for given tickers. No caching to ensure fresh data."""
    if not tickers:
        return pd.DataFrame(), []
    
    validated = list(set([t.strip().upper() for t in tickers if t]))
    failed = []
    successful = {}
    limited_history = []  # Track ETFs with limited history
    
    try:
        logger.info(f"Fetching {len(validated)} tickers from {start} to {end}")
        raw = yf.download(validated, start=start, end=end, progress=False, threads=True)
        if raw.empty:
            logger.warning("yfinance returned empty data")
            return pd.DataFrame(), validated
        
        close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']].rename(columns={'Close': validated[0]})
        
        # Parse start date for comparison
        start_date = pd.to_datetime(start)
        
        for ticker in validated:
            try:
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    if len(series) >= 30:
                        successful[ticker] = series
                        # Check if data starts significantly later than requested
                        if series.index.min() > start_date + pd.Timedelta(days=90):
                            limited_history.append(f"{ticker} (from {series.index.min().strftime('%Y-%m')})")
                    else:
                        failed.append(f"{ticker} ({len(series)}d)")
                else:
                    failed.append(ticker)
            except:
                failed.append(ticker)
        
        if successful:
            df = pd.DataFrame(successful)
            
            logger.info(f"Fetched {len(successful)} assets, date range: {df.index.min()} to {df.index.max()}")
            
            # CRITICAL: Don't drop any rows! Keep NaNs for dynamic backtest
            # Only forward fill small gaps (weekends/holidays)
            df = df.ffill(limit=5)
            
            # Log limited history ETFs
            if limited_history:
                logger.info(f"ETFs with limited history: {limited_history}")
            
            return df, failed
        return pd.DataFrame(), failed
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return pd.DataFrame(), validated

def fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch benchmark data. No caching to ensure fresh data."""
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False)
        return raw['Close'].dropna() if not raw.empty else pd.Series()
    except:
        return pd.Series()

# --- CUSTOM OPTIMIZATION FUNCTIONS ---
def calculate_ulcer_index(returns: pd.Series) -> float:
    """Calculate Ulcer Index (drawdown-based risk measure)."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ulcer_index = np.sqrt((drawdown ** 2).mean())
    return ulcer_index

def ulcer_performance_index(returns: pd.Series, rf: float = 0.065) -> float:
    """Calculate Ulcer Performance Index (return / ulcer index)."""
    ann_return = returns.mean() * 252
    ui = calculate_ulcer_index(returns)
    return (ann_return - rf) / ui if ui > 0 else 0

def optimize_upi(returns: pd.DataFrame, rf: float = 0.065) -> Dict[str, float]:
    """
    Optimize portfolio to maximize Ulcer Performance Index.
    
    Uses a two-stage approach:
    1. Calculate individual asset UPI scores as quality weights
    2. Optimize portfolio UPI while respecting asset quality rankings
    
    This prevents low-quality assets from dominating due to low volatility alone.
    """
    n_assets = len(returns.columns)
    assets = returns.columns.tolist()
    
    # Stage 1: Calculate individual asset quality scores
    asset_upi = {}
    asset_sharpe = {}
    for col in assets:
        rets = returns[col]
        ann_ret = rets.mean() * 252
        vol = rets.std() * np.sqrt(252)
        ui = calculate_ulcer_index(rets)
        
        # Individual UPI
        asset_upi[col] = (ann_ret - rf) / ui if ui > 0 else 0
        # Individual Sharpe
        asset_sharpe[col] = (ann_ret - rf) / vol if vol > 0 else 0
    
    # Create quality score (blend of UPI and Sharpe)
    # Normalize scores
    upi_vals = np.array(list(asset_upi.values()))
    sharpe_vals = np.array(list(asset_sharpe.values()))
    
    # Handle edge cases
    upi_min, upi_max = upi_vals.min(), upi_vals.max()
    sharpe_min, sharpe_max = sharpe_vals.min(), sharpe_vals.max()
    
    upi_norm = (upi_vals - upi_min) / (upi_max - upi_min + 1e-6)
    sharpe_norm = (sharpe_vals - sharpe_min) / (sharpe_max - sharpe_min + 1e-6)
    
    # Combined quality score: 50% UPI + 50% Sharpe
    quality_scores = 0.5 * upi_norm + 0.5 * sharpe_norm
    quality_dict = {assets[i]: quality_scores[i] for i in range(n_assets)}
    
    # Stage 2: Optimize with quality-aware objective
    def quality_weighted_neg_upi(weights):
        port_returns = returns.dot(weights)
        port_upi = ulcer_performance_index(port_returns, rf)
        
        # Quality penalty: penalize allocation to low-quality assets
        quality_alignment = sum(w * quality_dict[assets[i]] for i, w in enumerate(weights))
        
        # Combined objective: 70% portfolio UPI + 30% quality alignment
        combined = 0.7 * port_upi + 0.3 * quality_alignment * 10  # Scale quality component
        return -combined
    
    # Set bounds based on quality - high quality assets get higher max allocation
    bounds = []
    for i, col in enumerate(assets):
        q = quality_dict[col]
        if q >= 0.7:  # High quality
            bounds.append((0, 0.5))
        elif q >= 0.4:  # Medium quality
            bounds.append((0, 0.35))
        else:  # Low quality
            bounds.append((0, 0.20))  # Cap low-quality at 20%
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    init = np.array([1/n_assets] * n_assets)
    
    result = minimize(quality_weighted_neg_upi, init, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = {assets[i]: max(0, result.x[i]) for i in range(n_assets)}
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    # Fallback: quality-weighted allocation
    total_q = sum(max(0.1, q) for q in quality_scores)
    return {assets[i]: max(0.1, quality_scores[i]) / total_q for i in range(n_assets)}

def optimize_risk_parity(cov_matrix: pd.DataFrame) -> Dict[str, float]:
    """Risk Parity: Equal risk contribution from each asset."""
    n = len(cov_matrix)
    assets = cov_matrix.columns.tolist()
    
    def risk_budget_objective(weights, cov):
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / port_vol
        target_risk = port_vol / n
        return np.sum((risk_contrib - target_risk) ** 2)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0.01, 0.4) for _ in range(n)]
    init = np.array([1/n] * n)
    
    result = minimize(risk_budget_objective, init, args=(cov_matrix.values,),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = {assets[i]: max(0, result.x[i]) for i in range(n)}
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    return {a: 1/n for a in assets}

def optimize_inverse_volatility(returns: pd.DataFrame) -> Dict[str, float]:
    """Inverse Volatility Portfolio: Weight by 1/volatility."""
    vols = returns.std() * np.sqrt(252)
    inv_vols = 1 / vols
    weights = inv_vols / inv_vols.sum()
    return weights.to_dict()

def optimize_mdp(returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> Dict[str, float]:
    """Most Diversified Portfolio: Maximize diversification ratio."""
    n = len(returns.columns)
    assets = returns.columns.tolist()
    vols = returns.std().values * np.sqrt(252)
    
    def neg_div_ratio(weights):
        weighted_vol_sum = weights @ vols
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        return -weighted_vol_sum / port_vol if port_vol > 0 else 0
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 0.4) for _ in range(n)]
    init = np.array([1/n] * n)
    
    result = minimize(neg_div_ratio, init, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = {assets[i]: max(0, result.x[i]) for i in range(n)}
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    return {a: 1/n for a in assets}

def optimize_tangency_cal(mu: pd.Series, S: pd.DataFrame, rf: float, 
                          risk_tolerance: float = 1.0) -> Tuple[Dict[str, float], float]:
    """
    Tangency Portfolio with Capital Allocation Line (CAL).
    
    Finds the optimal risky portfolio (tangency portfolio) and allocates
    between risk-free asset and risky portfolio based on risk tolerance.
    
    Args:
        mu: Expected returns
        S: Covariance matrix
        rf: Risk-free rate
        risk_tolerance: 0 = 100% risk-free, 1 = 100% risky (tangency)
    
    Returns:
        weights: Asset weights including risk-free allocation
        risky_weight: Proportion allocated to risky portfolio
    """
    try:
        # Step 1: Find Tangency Portfolio (Max Sharpe on efficient frontier)
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w <= 0.40)
        tangency_weights = ef.max_sharpe(risk_free_rate=rf)
        tangency_weights = ef.clean_weights()
        
        # Calculate tangency portfolio expected return and volatility
        w_arr = np.array([tangency_weights.get(a, 0) for a in mu.index])
        tangency_return = w_arr @ mu.values
        tangency_vol = np.sqrt(w_arr @ S.values @ w_arr)
        tangency_sharpe = (tangency_return - rf) / tangency_vol if tangency_vol > 0 else 0
        
        # Step 2: Apply CAL allocation based on risk tolerance
        # risk_tolerance = 1.0 means 100% in tangency portfolio
        # risk_tolerance = 0.5 means 50% in tangency, 50% in risk-free
        risky_weight = max(0.0, min(1.0, risk_tolerance))
        
        # Final weights: scale tangency weights by risky allocation
        final_weights = {k: v * risky_weight for k, v in tangency_weights.items()}
        
        # Add risk-free allocation info
        risk_free_alloc = 1.0 - risky_weight
        
        return final_weights, risky_weight, risk_free_alloc, tangency_return, tangency_vol, tangency_sharpe
        
    except Exception as e:
        logger.error(f"Tangency CAL optimization error: {e}")
        n = len(mu)
        return {a: 1.0/n for a in mu.index}, 1.0, 0.0, 0.0, 0.0, 0.0

# --- MAIN OPTIMIZATION ENGINE ---
def optimize_portfolio(
    data: pd.DataFrame,
    rf: float,
    method: str,
    selected: List[str],
    risk_tolerance: float = 1.0
) -> Tuple[Dict[str, float], Tuple[float, float, float], pd.DataFrame, any, any, Dict]:
    """
    Portfolio optimization using multiple methods.
    Returns additional info dict for CAL details.
    """
    if not PYPFOPT_AVAILABLE:
        st.error("PyPortfolioOpt not installed!")
        return {}, (0, 0, 0), pd.DataFrame(), None, None, {}
    
    extra_info = {}
    
    try:
        if data.empty or len(data) < 63:
            raise ValueError("Insufficient data")
        
        avail = [a for a in selected if a in data.columns]
        if len(avail) < 2:
            raise ValueError("Need at least 2 assets")
        
        subset = data[avail].copy()
        assets = subset.columns.tolist()
        returns = subset.pct_change().dropna()
        
        # Calculate covariance and expected returns
        if SKLEARN_AVAILABLE:
            S = risk_models.CovarianceShrinkage(subset).ledoit_wolf()
        else:
            S = risk_models.sample_cov(subset)
        
        mu = expected_returns.capm_return(subset, risk_free_rate=rf, frequency=252)
        
        weights = {}
        
        # === OPTIMIZATION METHODS ===
        
        if method == "Mean-Variance (Max Sharpe)":
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= 0.40)
            weights = ef.max_sharpe(risk_free_rate=rf)
            weights = ef.clean_weights()
            
        elif method == "Tangency Portfolio (CAL)":
            weights, risky_w, rf_alloc, tang_ret, tang_vol, tang_sharpe = optimize_tangency_cal(
                mu, S, rf, risk_tolerance
            )
            extra_info = {
                'risky_weight': risky_w,
                'risk_free_alloc': rf_alloc,
                'tangency_return': tang_ret,
                'tangency_vol': tang_vol,
                'tangency_sharpe': tang_sharpe
            }
            
        elif method == "Global Minimum Variance (GMVP)":
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= 0.40)
            weights = ef.min_volatility()
            weights = ef.clean_weights()
            
        elif method == "Hierarchical Risk Parity (HRP)":
            hrp = HRPOpt(returns)
            weights = hrp.optimize()
            weights = hrp.clean_weights()
            
        elif method == "Risk Parity (Equal Risk Contribution)":
            weights = optimize_risk_parity(S)
            
        elif method == "Black-Litterman":
            delta = black_litterman.market_implied_risk_aversion(subset)
            delta = 2.5 if delta <= 0 or np.isnan(delta) else delta
            pi = black_litterman.market_implied_prior_returns(len(assets), delta, S)
            views = {t: 0.05 for t in assets}
            bl = BlackLittermanModel(S, pi=pi, absolute_views=views, 
                                    omega="idzorek", view_confidences=[0.6]*len(assets))
            bl_mu = bl.bl_returns()
            ef = EfficientFrontier(bl_mu, S)
            ef.add_constraint(lambda w: w <= 0.40)
            weights = ef.max_sharpe(risk_free_rate=rf)
            weights = ef.clean_weights()
            mu = bl_mu
            
        elif method == "Mean-Semivariance (Downside Risk)":
            semi_cov = risk_models.semicovariance(subset, benchmark=0)
            ef = EfficientFrontier(mu, semi_cov)
            ef.add_constraint(lambda w: w <= 0.40)
            weights = ef.min_volatility()
            weights = ef.clean_weights()
            S = semi_cov
            
        elif method == "Inverse Volatility (IVP)":
            weights = optimize_inverse_volatility(returns)
            
        elif method == "Most Diversified Portfolio (MDP)":
            weights = optimize_mdp(returns, S)
            
        elif method == "Equal Weight (EWP)":
            weights = {a: 1.0/len(assets) for a in assets}
            
        elif method == "Ulcer Performance Index (UPI)":
            weights = optimize_upi(returns, rf)
        
        else:
            weights = {a: 1.0/len(assets) for a in assets}
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # Calculate performance
        w_series = pd.Series(weights)
        port_returns = returns.dot(w_series)
        ann_ret = port_returns.mean() * 252
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
        perf = (ann_ret, ann_vol, sharpe)
        
        return weights, perf, subset, S, mu, extra_info
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        if selected:
            n = len([a for a in selected if a in data.columns])
            weights = {a: 1.0/n for a in selected if a in data.columns}
        return weights, (0, 0, 0), data, None, None, {}

# --- ASSET SCORING ---
def calculate_scores(data: pd.DataFrame, benchmark: pd.Series, min_mom: float, rf: float) -> List[AssetScore]:
    scores = []
    returns = data.pct_change().dropna()
    
    for ticker in data.columns:
        try:
            prices = data[ticker].dropna()
            rets = returns[ticker].dropna()
            if len(rets) < 63:
                continue
            
            lb126 = min(126, len(prices) - 1)
            mom_126d = (prices.iloc[-1] / prices.iloc[-lb126] - 1) if lb126 > 0 else 0
            
            lb63 = min(63, len(prices) - 1)
            mom_63d = (prices.iloc[-1] / prices.iloc[-lb63] - 1) if lb63 > 0 else 0
            
            vol = rets.std() * np.sqrt(252)
            ann_ret = mom_126d * (252 / lb126) if lb126 > 0 else 0
            sharpe = (ann_ret - rf) / (vol + 0.001) if vol > 0 else 0
            
            # Calculate Ulcer Index and UPI for individual asset
            ui = calculate_ulcer_index(rets)
            asset_upi = ulcer_performance_index(rets, rf)
            
            curr = prices.iloc[-1]
            ma50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else curr
            ma200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else curr
            trend = (1 if curr > ma50 else 0) + (1 if curr > ma200 else 0)
            
            rs = 0
            if benchmark is not None and len(benchmark) >= lb126:
                try:
                    common = prices.index.intersection(benchmark.index)
                    if len(common) >= lb126:
                        idx = max(0, len(common) - lb126)
                        t_perf = prices.loc[common].iloc[-1] / prices.loc[common].iloc[idx] - 1
                        b_perf = benchmark.loc[common].iloc[-1] / benchmark.loc[common].iloc[idx] - 1
                        rs = t_perf - b_perf if pd.notna(t_perf) and pd.notna(b_perf) else 0
                except:
                    pass
            
            if mom_126d >= min_mom and sharpe > 0.5 and trend >= 1:
                signal = "STRONG BUY"
            elif mom_126d >= min_mom and trend >= 1:
                signal = "BUY"
            elif mom_126d >= 0 and trend >= 1:
                signal = "HOLD"
            elif mom_126d < 0 and trend == 0:
                signal = "AVOID"
            else:
                signal = "WATCH"
            
            scores.append(AssetScore(
                ticker=ticker, momentum_126d=mom_126d, momentum_63d=mom_63d,
                volatility=vol, sharpe_ratio=sharpe, trend_score=trend,
                relative_strength=rs, composite_score=0, 
                ulcer_index=ui, upi=asset_upi, signal=signal
            ))
        except Exception as e:
            logger.warning(f"Score error {ticker}: {e}")
    
    return scores

def select_assets(scores: List[AssetScore], model: AssetSelectionModel, 
                  min_assets: int, max_assets: int, min_mom: float) -> List[str]:
    if not scores:
        return []
    
    df = pd.DataFrame([vars(s) for s in scores])
    
    if model != AssetSelectionModel.LOW_VOLATILITY:
        filtered = df[df['momentum_126d'] >= min_mom]
        if len(filtered) >= min_assets:
            df = filtered
    
    for col in ['momentum_126d', 'momentum_63d', 'sharpe_ratio', 'relative_strength']:
        mn, mx = df[col].min(), df[col].max()
        df[f'{col}_n'] = (df[col] - mn) / (mx - mn + 1e-6)
    
    mn, mx = df['volatility'].min(), df['volatility'].max()
    df['volatility_n'] = 1 - (df['volatility'] - mn) / (mx - mn + 1e-6)
    
    if model == AssetSelectionModel.MOMENTUM_126D:
        df['composite'] = df['momentum_126d_n']
    elif model == AssetSelectionModel.RISK_ADJUSTED:
        df['composite'] = df['sharpe_ratio_n']
    elif model == AssetSelectionModel.LOW_VOLATILITY:
        df['composite'] = df['volatility_n']
    elif model == AssetSelectionModel.QUALITY_MOMENTUM:
        df['composite'] = 0.4*df['momentum_126d_n'] + 0.3*df['sharpe_ratio_n'] + 0.2*(df['trend_score']/2) + 0.1*df['volatility_n']
    elif model == AssetSelectionModel.RELATIVE_STRENGTH:
        df['composite'] = df['relative_strength'] + 0.3*df['momentum_126d_n']
    elif model == AssetSelectionModel.TREND_FOLLOWING:
        df = df[df['trend_score'] >= 1] if len(df[df['trend_score'] >= 1]) >= min_assets else df
        df['composite'] = df['momentum_126d_n']
    elif model == AssetSelectionModel.HIGH_CONVICTION:
        df['composite'] = 0.5*df['sharpe_ratio_n'] + 0.5*df['momentum_126d_n']
    
    df = df.sort_values('composite', ascending=False)
    return df.head(min(max_assets, max(min_assets, len(df))))['ticker'].tolist()

# --- OPPORTUNITY SIGNALS ---
def generate_signals(data: pd.DataFrame, scores: List[AssetScore], benchmark: pd.Series, 
                    weights: Dict[str, float] = None) -> List[OpportunitySignal]:
    signals = []
    weights = weights or {}
    returns = data.pct_change().dropna()
    
    for score in scores:
        ticker = score.ticker
        if ticker not in data.columns:
            continue
        
        try:
            prices = data[ticker].dropna()
            if len(prices) < 50:
                continue
            
            # Calculate max drawdown
            cumulative = (1 + returns[ticker]).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            curr = prices.iloc[-1]
            support = prices.tail(20).min()
            resistance = prices.tail(20).max()
            
            if score.signal in ["STRONG BUY", "BUY"]:
                target = curr * (1 + abs(score.momentum_126d) * 0.5)
                stop = support * 0.98
            else:
                target = resistance
                stop = support * 0.95
            
            strength = 0
            if score.momentum_126d >= 0.03: strength += 30
            if score.sharpe_ratio > 0.5: strength += 25
            if score.trend_score >= 2: strength += 25
            elif score.trend_score >= 1: strength += 15
            if score.relative_strength > 0: strength += 20
            strength = min(100, strength)
            
            trend_status = "Strong Uptrend" if score.trend_score == 2 else "Uptrend" if score.trend_score == 1 else "Downtrend"
            
            rationale = []
            if score.momentum_126d >= 0.03:
                rationale.append(f"126D: {score.momentum_126d:.1%}")
            if score.sharpe_ratio > 0.5:
                rationale.append(f"Sharpe: {score.sharpe_ratio:.2f}")
            if score.upi > 1.0:
                rationale.append(f"UPI: {score.upi:.2f}")
            if max_dd < 0.10:
                rationale.append(f"Low DD: {max_dd:.1%}")
            if score.trend_score >= 1:
                rationale.append("Above MAs")
            
            signals.append(OpportunitySignal(
                ticker=ticker, signal_type=score.signal, strength=strength,
                momentum_126d=score.momentum_126d, sharpe_ratio=score.sharpe_ratio,
                trend_status=trend_status, support_level=support, resistance_level=resistance,
                current_price=curr, target_price=target, stop_loss=stop,
                rationale=". ".join(rationale) or "Monitoring",
                weight=weights.get(ticker, 0),
                ulcer_index=score.ulcer_index,
                upi=score.upi,
                max_drawdown=max_dd
            ))
        except Exception as e:
            logger.warning(f"Signal error {ticker}: {e}")
    
    # Sort by weight (descending), then by strength (descending)
    signals.sort(key=lambda x: (-x.weight, -x.strength))
    return signals

# --- REBALANCING FUNCTIONS ---
def get_rebalance_dates(data: pd.DataFrame, frequency: RebalanceFrequency) -> List[pd.Timestamp]:
    """Generate rebalance dates based on frequency."""
    dates = data.index.tolist()
    
    if frequency == RebalanceFrequency.BUY_HOLD:
        return [dates[0]]  # Only initial allocation
    
    elif frequency == RebalanceFrequency.FORTNIGHTLY:
        # Every 2 weeks (10 trading days approximately)
        rebalance_dates = [dates[0]]
        last_rebal = dates[0]
        for d in dates[1:]:
            days_diff = len(data.loc[last_rebal:d]) - 1
            if days_diff >= 10:  # ~2 weeks of trading days
                rebalance_dates.append(d)
                last_rebal = d
        return rebalance_dates
    
    elif frequency == RebalanceFrequency.MONTHLY:
        # End of each month
        monthly = data.resample('ME').last().index.tolist()
        return [d for d in monthly if d in data.index]
    
    elif frequency == RebalanceFrequency.QUARTERLY:
        # End of each quarter
        quarterly = data.resample('QE').last().index.tolist()
        return [d for d in quarterly if d in data.index]
    
    return [dates[0]]

def calculate_exit_rank(num_etfs: int, exit_rank_pct: float = 50.0) -> int:
    """
    Calculate exit rank based on number of ETFs and percentage buffer.
    
    Formula: exit_rank = ceil(num_etfs * (1 + exit_rank_pct/100))
    
    Examples with 50% buffer:
    - 5 ETFs → 5 * 1.5 = 7.5 → 8
    - 8 ETFs → 8 * 1.5 = 12
    - 10 ETFs → 10 * 1.5 = 15
    
    Args:
        num_etfs: Number of ETFs in current portfolio
        exit_rank_pct: Percentage buffer (e.g., 50 means exit rank is 50% above num_etfs)
    
    Returns:
        Exit rank threshold (integer)
    """
    import math
    multiplier = 1 + (exit_rank_pct / 100.0)
    exit_rank = math.ceil(num_etfs * multiplier)
    return exit_rank

def check_exit_rank_breach(
    holdings: Dict[str, float],
    scores: List[AssetScore],
    exit_rank: int
) -> Tuple[bool, Dict[str, int], List[str]]:
    """
    Check if any holding has breached the exit rank threshold.
    
    Args:
        holdings: Current holdings {ticker: shares}
        scores: List of AssetScore objects
        exit_rank: Exit rank threshold
    
    Returns:
        Tuple of:
        - breach_detected: True if any holding rank > exit_rank
        - holding_ranks: Dict of {ticker: current_rank}
        - breached_tickers: List of tickers that breached
    """
    if not holdings or not scores:
        return True, {}, []  # Force rebalance if no holdings
    
    # Create ranking from scores (1 = best)
    scores_df = pd.DataFrame([vars(s) for s in scores])
    
    # Sort by composite score descending to get rankings
    # First calculate composite if not present
    for col in ['momentum_126d', 'sharpe_ratio']:
        if col in scores_df.columns:
            mn, mx = scores_df[col].min(), scores_df[col].max()
            scores_df[f'{col}_n'] = (scores_df[col] - mn) / (mx - mn + 1e-6)
    
    # Use quality momentum composite
    if 'momentum_126d_n' in scores_df.columns and 'sharpe_ratio_n' in scores_df.columns:
        scores_df['composite'] = 0.5 * scores_df['momentum_126d_n'] + 0.5 * scores_df['sharpe_ratio_n']
    else:
        scores_df['composite'] = scores_df.get('momentum_126d', 0)
    
    scores_df = scores_df.sort_values('composite', ascending=False).reset_index(drop=True)
    scores_df['rank'] = scores_df.index + 1  # Rank 1 = best
    
    # Get current holdings' ranks
    holding_ranks = {}
    breached_tickers = []
    
    for ticker in holdings.keys():
        matching = scores_df[scores_df['ticker'] == ticker]
        if len(matching) > 0:
            rank = matching['rank'].iloc[0]
            holding_ranks[ticker] = rank
            if rank > exit_rank:
                breached_tickers.append(ticker)
        else:
            # Ticker not in scores - consider it breached (delisted or no data)
            holding_ranks[ticker] = 999
            breached_tickers.append(ticker)
    
    breach_detected = len(breached_tickers) > 0
    return breach_detected, holding_ranks, breached_tickers

def run_backtest_with_rebalancing(
    weights_func,
    data: pd.DataFrame,
    benchmark: pd.Series,
    frequency: RebalanceFrequency,
    initial: float = 100,
    rf: float = 0.065,
    opt_method: str = "Mean-Variance (Max Sharpe)",
    selection_model: AssetSelectionModel = AssetSelectionModel.QUALITY_MOMENTUM,
    min_assets: int = 5,
    max_assets: int = 10,
    min_mom: float = 0.03,
    risk_tolerance: float = 1.0,
    exit_rank_mode: str = "auto",  # "auto", "manual", or "disabled"
    exit_rank_pct: float = 50.0,   # For auto mode: 50% means exit rank = 1.5x num_etfs
    manual_exit_rank: int = 10    # For manual mode: specific exit rank
) -> Tuple[pd.DataFrame, pd.Series, List[Dict]]:
    """
    Run backtest with periodic rebalancing and EXIT RANK logic.
    
    EXIT RANK MECHANISM:
    - At each potential rebalance date, check current holdings' ranks
    - If ANY holding's rank > exit_rank → REBALANCE (exit rank breached)
    - If ALL holdings' ranks <= exit_rank → SKIP rebalance (no trigger)
    
    EXIT RANK CALCULATION:
    - Auto mode: exit_rank = ceil(num_etfs * (1 + exit_rank_pct/100))
      Example with 50%: 5 ETFs → 8, 8 ETFs → 12, 10 ETFs → 15
    - Manual mode: Use specific exit_rank value
    - Disabled: Always rebalance on schedule (old behavior)
    
    Returns:
        equity_df: Equity curve dataframe
        port_rets: Portfolio returns series
        rebalance_log: List of rebalancing events (with exit rank details)
    """
    try:
        if data.empty:
            return pd.DataFrame(), pd.Series(), []
        
        rebalance_dates = get_rebalance_dates(data, frequency)
        
        logger.info(f"Backtest: {len(data)} days, {len(data.columns)} total assets, {len(rebalance_dates)} rebalance dates")
        
        # Initialize
        portfolio_value = initial
        holdings = {}  # ticker -> shares
        cash = initial
        current_weights = {}
        current_exit_rank = manual_exit_rank  # Will be updated in auto mode
        
        equity_curve = []
        port_returns = []
        rebalance_log = []
        skipped_rebalances = []  # Track skipped rebalances for logging
        
        prev_value = initial
        first_rebalance_done = False
        
        # Minimum lookback before first rebalance (126 days for momentum calculation)
        min_lookback = 126
        
        for i, date in enumerate(data.index):
            # Check if rebalancing day
            is_rebalance = date in rebalance_dates
            
            if is_rebalance and i >= min_lookback:
                # Get historical data up to this date
                hist_data = data.loc[:date].copy()
                
                # DYNAMIC ASSET SELECTION: Find assets with sufficient data at THIS point in time
                # Require at least 126 days of valid data for momentum calculation
                available_assets = []
                for col in hist_data.columns:
                    # Count non-null values in the lookback period
                    lookback_data = hist_data[col].tail(min_lookback)
                    valid_count = lookback_data.notna().sum()
                    if valid_count >= min_lookback * 0.8:  # 80% data coverage in lookback
                        available_assets.append(col)
                
                if len(available_assets) >= min_assets:
                    # Use only available assets for scoring
                    hist_data_filtered = hist_data[available_assets].copy()
                    
                    # Forward fill any gaps within available assets
                    hist_data_filtered = hist_data_filtered.ffill()
                    
                    # Calculate scores using historical data
                    hist_bench = None
                    if benchmark is not None and not benchmark.empty:
                        try:
                            hist_bench = benchmark.loc[:date]
                        except:
                            pass
                    
                    scores = calculate_scores(hist_data_filtered, hist_bench, min_mom, rf)
                    
                    if scores and len(scores) >= min_assets:
                        # ========== EXIT RANK CHECK ==========
                        should_rebalance = True
                        breach_detected = False
                        holding_ranks = {}
                        breached_tickers = []
                        
                        if first_rebalance_done and exit_rank_mode != "disabled":
                            # Calculate current exit rank based on portfolio size
                            num_etfs_in_portfolio = len([h for h in holdings.keys() if holdings[h] > 0])
                            
                            if exit_rank_mode == "auto":
                                current_exit_rank = calculate_exit_rank(num_etfs_in_portfolio, exit_rank_pct)
                            else:  # manual mode
                                current_exit_rank = manual_exit_rank
                            
                            # Cap exit rank at number of available assets
                            current_exit_rank = min(current_exit_rank, len(available_assets))
                            
                            # Check if any holding has breached exit rank
                            breach_detected, holding_ranks, breached_tickers = check_exit_rank_breach(
                                holdings, scores, current_exit_rank
                            )
                            
                            should_rebalance = breach_detected
                            
                            if not should_rebalance:
                                # Log skipped rebalance
                                skipped_rebalances.append({
                                    'date': date,
                                    'reason': 'Exit rank not breached',
                                    'exit_rank': current_exit_rank,
                                    'holding_ranks': holding_ranks.copy(),
                                    'num_holdings': num_etfs_in_portfolio
                                })
                                logger.info(f"SKIP rebalance {date}: Exit rank {current_exit_rank} not breached. Holdings ranks: {holding_ranks}")
                        
                        # ========== REBALANCE IF TRIGGERED ==========
                        if should_rebalance:
                            # Select assets based on model (from available assets only)
                            selected = select_assets(scores, selection_model, min_assets, max_assets, min_mom)
                            
                            if len(selected) >= 2:
                                # Optimize with selected assets
                                weights, perf, _, _, _, extra = optimize_portfolio(
                                    hist_data_filtered, rf, opt_method, selected, risk_tolerance
                                )
                                
                                if weights:
                                    # Calculate current portfolio value
                                    current_prices = data.loc[date]
                                    if holdings:
                                        portfolio_value = sum(
                                            holdings.get(t, 0) * current_prices.get(t, 0) 
                                            for t in holdings if t in current_prices.index and pd.notna(current_prices.get(t, 0))
                                        ) + cash
                                    
                                    # Rebalance to new weights
                                    new_holdings = {}
                                    for ticker, weight in weights.items():
                                        if weight > 0.001 and ticker in current_prices.index:
                                            price = current_prices[ticker]
                                            if pd.notna(price) and price > 0:
                                                new_holdings[ticker] = (portfolio_value * weight) / price
                                    
                                    if new_holdings:
                                        old_holdings = list(holdings.keys()) if holdings else []
                                        holdings = new_holdings
                                        current_weights = weights.copy()
                                        cash = 0  # Fully invested
                                        
                                        # Update exit rank for new portfolio size
                                        num_new_etfs = len([h for h in new_holdings.keys() if new_holdings[h] > 0])
                                        if exit_rank_mode == "auto":
                                            current_exit_rank = calculate_exit_rank(num_new_etfs, exit_rank_pct)
                                        
                                        first_rebalance_done = True
                                        
                                        # Log rebalance with exit rank details
                                        rebalance_log.append({
                                            'date': date,
                                            'weights': weights.copy(),
                                            'selected': selected.copy(),
                                            'available': len(available_assets),
                                            'perf': perf,
                                            'exit_rank_mode': exit_rank_mode,
                                            'exit_rank': current_exit_rank,
                                            'breach_detected': breach_detected,
                                            'breached_tickers': breached_tickers.copy() if breached_tickers else [],
                                            'holding_ranks_before': holding_ranks.copy() if holding_ranks else {},
                                            'trigger_reason': 'Exit rank breached' if breach_detected else 'Initial rebalance'
                                        })
                                        
                                        logger.info(f"REBALANCE {date}: Exit rank {current_exit_rank}, Breached: {breached_tickers}, New: {len(selected)} ETFs")
            
            # Calculate daily portfolio value
            current_prices = data.loc[date]
            
            if holdings and first_rebalance_done:
                # After first rebalance - use actual holdings
                new_value = 0
                for t in holdings:
                    if t in current_prices.index and pd.notna(current_prices[t]):
                        new_value += holdings[t] * current_prices[t]
                portfolio_value = new_value + cash if new_value > 0 else prev_value
            elif i > 0:
                # Before first rebalance - use equal weight portfolio of ALL available assets as proxy
                prev_prices = data.iloc[i-1]
                daily_ret = 0
                valid_count = 0
                for asset in data.columns:
                    if (asset in current_prices.index and asset in prev_prices.index and
                        pd.notna(prev_prices[asset]) and pd.notna(current_prices[asset]) and 
                        prev_prices[asset] > 0):
                        asset_ret = (current_prices[asset] / prev_prices[asset] - 1)
                        daily_ret += asset_ret
                        valid_count += 1
                if valid_count > 0:
                    daily_ret /= valid_count
                    portfolio_value = prev_value * (1 + daily_ret)
            
            # Calculate return
            daily_return = (portfolio_value / prev_value - 1) if prev_value > 0 else 0
            port_returns.append(daily_return)
            
            equity_curve.append({
                'Date': date,
                'Portfolio': portfolio_value
            })
            
            prev_value = portfolio_value
        
        # Create equity dataframe
        eq_df = pd.DataFrame(equity_curve).set_index('Date')
        
        # Add individual asset performance (only for assets in final weights)
        final_assets = list(holdings.keys()) if holdings else []
        for asset in final_assets[:5]:  # Limit to top 5 for display
            if asset in data.columns:
                asset_data = data[asset].dropna()
                if len(asset_data) > 0:
                    first_valid = asset_data.first_valid_index()
                    if first_valid is not None:
                        eq_df[clean_ticker(asset)] = data[asset] / data[asset].loc[first_valid] * initial
        
        # Add benchmark
        if benchmark is not None and not benchmark.empty:
            common = eq_df.index.intersection(benchmark.index)
            if len(common) > 0:
                b = benchmark.loc[common]
                eq_df = eq_df.loc[common]
                eq_df['Benchmark'] = (b / b.iloc[0] * initial).values
        
        port_rets = pd.Series(port_returns, index=data.index)
        
        logger.info(f"Backtest complete: {len(rebalance_log)} rebalances, {len(skipped_rebalances)} skipped (exit rank not breached)")
        
        # Add skipped rebalances info to the last log entry for UI display
        if rebalance_log and skipped_rebalances:
            rebalance_log[-1]['total_skipped'] = len(skipped_rebalances)
            rebalance_log[-1]['skipped_details'] = skipped_rebalances
        
        return eq_df, port_rets, rebalance_log
        
    except Exception as e:
        logger.error(f"Backtest with rebalancing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.Series(), []

# --- SIMPLE BACKTEST (for Buy & Hold) ---
def run_backtest(weights: Dict, data: pd.DataFrame, benchmark: pd.Series, initial: float = 100):
    try:
        if not weights:
            return pd.DataFrame(), pd.Series()
        
        selected = [a for a in weights if a in data.columns]
        if not selected:
            return pd.DataFrame(), pd.Series()
        
        prices = data[selected].copy()
        rets = prices.pct_change().dropna()
        w = pd.Series({k: weights[k] for k in selected})
        port_rets = rets.dot(w)
        
        equity = (1 + port_rets).cumprod() * initial
        eq_df = pd.DataFrame({'Portfolio': equity})
        
        for asset in selected:
            eq_df[clean_ticker(asset)] = prices[asset] / prices[asset].iloc[0] * initial
        
        if benchmark is not None and not benchmark.empty:
            try:
                common = eq_df.index.intersection(benchmark.index)
                if len(common) > 0:
                    b = benchmark.loc[common]
                    eq_df = eq_df.loc[common]
                    eq_df['Benchmark'] = (b / b.iloc[0] * initial).values
            except:
                pass
        
        return eq_df, port_rets
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return pd.DataFrame(), pd.Series()

# --- CHARTS ---
def create_equity_chart(eq: pd.DataFrame) -> go.Figure:
    """
    Create equity chart with BOLD HIGHLIGHT on hover.
    Uses Plotly's native hoveron and hoverinfo for highlighting effect.
    """
    fig = go.Figure()
    
    # Neon color palette
    neon_colors = {
        'Portfolio': '#00ff88',      # Neon Green
        'Benchmark': '#ff6b6b',      # Neon Red/Coral
    }
    
    # Extended neon palette for individual assets
    asset_neon_colors = [
        '#00bfff',  # Cyan Blue
        '#ff00ff',  # Magenta  
        '#ffff00',  # Yellow
        '#ff8c00',  # Orange
        '#00ffff',  # Cyan
        '#ff1493',  # Pink
        '#7fff00',  # Chartreuse
        '#9400d3',  # Purple
        '#00fa9a',  # Spring Green
        '#ff4500',  # Orange Red
        '#1e90ff',  # Dodger Blue
        '#ffd700',  # Gold
        '#adff2f',  # Green Yellow
        '#da70d6',  # Orchid
        '#87ceeb',  # Sky Blue
    ]
    
    for i, col in enumerate(eq.columns):
        if col == 'Portfolio':
            color = neon_colors['Portfolio']
            width = 2
            dash = 'solid'
        elif col == 'Benchmark':
            color = neon_colors['Benchmark']
            width = 1
            dash = 'dash'
        else:
            color = asset_neon_colors[i % len(asset_neon_colors)]
            width = 1
            dash = 'solid'
        
        fig.add_trace(go.Scatter(
            x=eq.index, 
            y=eq[col], 
            mode='lines', 
            name=col,
            line=dict(color=color, width=width, dash=dash),
            hoverinfo='text+name',
            hovertext=[f"{col}<br>{eq.index[j].strftime('%Y-%m-%d')}<br>Equity: {eq[col].iloc[j]:.1f}%" for j in range(len(eq))],
            hoverlabel=dict(
                bgcolor='rgba(20,25,35,0.95)',
                bordercolor=color,
                font=dict(size=13, color='white', family='SF Mono, Consolas, monospace')
            ),
            # Key for highlighting - set selected/unselected states
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.1))
        ))
    
    # Update layout with hover mode that enables trace highlighting
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,15,25,0.95)',
        title=dict(
            text="Price Performance (Indexed to 100)",
            font=dict(size=18, color='#e0e0e0', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Date",
            gridcolor='rgba(50,60,80,0.3)',
            showgrid=True,
            zeroline=False,
            tickfont=dict(color='#888')
        ),
        yaxis=dict(
            title="Equity (%, Log scale)",
            gridcolor='rgba(50,60,80,0.3)',
            showgrid=True,
            zeroline=False,
            tickfont=dict(color='#888'),
            type='log'
        ),
        hovermode='closest',
        hoverdistance=20,  # Increase hover detection distance
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10, color='#aaa'),
            itemsizing='constant'
        ),
        margin=dict(r=50, t=80, b=100, l=70),
    )
    
    # Add baseline at 100
    fig.add_hline(
        y=100, 
        line_dash="dot", 
        line_color="rgba(255,255,255,0.2)",
        line_width=1
    )
    
    return fig


def create_equity_chart_with_highlight(eq: pd.DataFrame) -> str:
    """
    Create equity chart with TRUE hover highlighting using Plotly.js directly.
    Returns HTML string that can be rendered with st.components.v1.html()
    """
    
    # Neon color palette
    neon_colors = {
        'Portfolio': '#00ff88',
        'Benchmark': '#ff6b6b',
    }
    
    asset_neon_colors = [
        '#00bfff', '#ff00ff', '#ffff00', '#ff8c00', '#00ffff',
        '#ff1493', '#7fff00', '#9400d3', '#00fa9a', '#ff4500',
        '#1e90ff', '#ffd700', '#adff2f', '#da70d6', '#87ceeb'
    ]
    
    # Build traces data for JavaScript
    traces_js = []
    for i, col in enumerate(eq.columns):
        if col == 'Portfolio':
            color = neon_colors['Portfolio']
            width = 2
            dash = 'solid'
        elif col == 'Benchmark':
            color = neon_colors['Benchmark']
            width = 1
            dash = 'dash'
        else:
            color = asset_neon_colors[i % len(asset_neon_colors)]
            width = 1
            dash = 'solid'
        
        x_vals = [d.strftime('%Y-%m-%d') for d in eq.index]
        y_vals = eq[col].tolist()
        
        trace = {
            'x': x_vals,
            'y': y_vals,
            'mode': 'lines',
            'name': col,
            'line': {'color': color, 'width': width, 'dash': dash},
            'hovertemplate': f'<b>{col}</b><br>%{{x}}<br>Equity: %{{y:.1f}}%<extra></extra>',
            'hoverlabel': {
                'bgcolor': 'rgba(20,25,35,0.95)',
                'bordercolor': color,
                'font': {'size': 13, 'color': 'white', 'family': 'SF Mono, Consolas, monospace'}
            }
        }
        traces_js.append(trace)
    
    import json
    traces_json = json.dumps(traces_js)
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background: transparent; }}
            #chart {{ width: 100%; height: 500px; }}
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <script>
            var traces = {traces_json};
            var originalWidths = traces.map(t => t.line.width);
            
            var layout = {{
                template: 'plotly_dark',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(10,15,25,0.95)',
                title: {{
                    text: 'Price Performance (Indexed to 100)',
                    font: {{size: 18, color: '#e0e0e0'}},
                    x: 0.5
                }},
                xaxis: {{
                    title: 'Date',
                    gridcolor: 'rgba(50,60,80,0.3)',
                    tickfont: {{color: '#888'}}
                }},
                yaxis: {{
                    title: 'Equity (%, Log scale)',
                    gridcolor: 'rgba(50,60,80,0.3)',
                    tickfont: {{color: '#888'}},
                    type: 'log'
                }},
                hovermode: 'closest',
                legend: {{
                    orientation: 'h',
                    y: -0.15,
                    x: 0.5,
                    xanchor: 'center',
                    font: {{size: 10, color: '#aaa'}}
                }},
                margin: {{r: 50, t: 60, b: 80, l: 70}},
                shapes: [{{
                    type: 'line',
                    x0: 0, x1: 1, xref: 'paper',
                    y0: 100, y1: 100,
                    line: {{color: 'rgba(255,255,255,0.2)', width: 1, dash: 'dot'}}
                }}]
            }};
            
            Plotly.newPlot('chart', traces, layout, {{responsive: true}});
            
            var chart = document.getElementById('chart');
            
            // Hover highlight effect
            chart.on('plotly_hover', function(data) {{
                var hoveredTrace = data.points[0].curveNumber;
                var update = {{}};
                
                var widths = [];
                var opacities = [];
                
                for (var i = 0; i < traces.length; i++) {{
                    if (i === hoveredTrace) {{
                        widths.push(originalWidths[i] * 2.5);  // Bold
                        opacities.push(1);
                    }} else {{
                        widths.push(originalWidths[i] * 0.7);  // Slightly thinner
                        opacities.push(0.5);  // 50% opacity - still visible for comparison
                    }}
                }}
                
                Plotly.restyle(chart, {{
                    'line.width': widths,
                    'opacity': opacities
                }});
            }});
            
            // Restore on unhover
            chart.on('plotly_unhover', function() {{
                var widths = originalWidths.slice();
                var opacities = traces.map(() => 0.85);
                
                Plotly.restyle(chart, {{
                    'line.width': widths,
                    'opacity': opacities
                }});
            }});
        </script>
    </body>
    </html>
    '''
    
    return html

def create_monthly_heatmap(rets: pd.Series) -> go.Figure:
    """Create monthly returns heatmap with NEON colors, bold text, Total and Max DD columns."""
    try:
        if rets.empty or len(rets) < 20:
            return go.Figure()
        
        # Resample to monthly returns
        monthly = rets.resample('ME').apply(lambda x: (1+x).prod()-1 if len(x) > 0 else 0)
        
        if monthly.empty:
            return go.Figure()
        
        # Get unique years
        years = sorted(monthly.index.year.unique())
        
        if len(years) == 0:
            return go.Figure()
        
        # Create column names: 12 months + Total + Max DD
        column_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Total', 'Max DD']
        
        # Build the data matrix with Total and Max DD
        data_matrix = []
        for year in years:
            row = []
            year_returns = []
            
            # Get monthly returns for this year
            for month in range(1, 13):
                matching = monthly[(monthly.index.year == year) & (monthly.index.month == month)]
                if len(matching) > 0:
                    val = matching.iloc[0] * 100  # Convert to percentage
                    row.append(val)
                    year_returns.append(matching.iloc[0])
                else:
                    row.append(np.nan)
            
            # Calculate Total (yearly return) - compound the monthly returns
            if year_returns:
                total_return = ((1 + pd.Series(year_returns)).prod() - 1) * 100
            else:
                total_return = np.nan
            row.append(total_return)
            
            # Calculate Max Drawdown for this year
            year_data = rets[rets.index.year == year]
            if len(year_data) > 0:
                cum_returns = (1 + year_data).cumprod()
                rolling_max = cum_returns.cummax()
                drawdowns = (cum_returns - rolling_max) / rolling_max
                max_dd = drawdowns.min() * 100  # Convert to percentage (will be negative)
            else:
                max_dd = np.nan
            row.append(max_dd)
            
            data_matrix.append(row)
        
        # Convert to numpy array
        z_values = np.array(data_matrix)
        
        # Create text annotations with bold formatting
        text_values = []
        for row in z_values:
            text_row = []
            for i, v in enumerate(row):
                if pd.notna(v):
                    if i == 13:  # Max DD column - always negative
                        text_row.append(f'<b>{v:.2f}%</b>')
                    else:
                        text_row.append(f'<b>{v:.2f}%</b>')
                else:
                    text_row.append('')
            text_values.append(text_row)
        
        # NEON Color Scale: Magenta (negative) -> Dark -> Cyan/Green (positive)
        neon_colorscale = [
            [0.0, '#ff0066'],      # Neon Magenta/Pink (very negative)
            [0.15, '#ff2244'],     # Neon Red
            [0.3, '#ff6600'],      # Neon Orange
            [0.45, '#1a1a2e'],     # Dark (near zero negative)
            [0.5, '#0d1117'],      # Dark center (zero)
            [0.55, '#1a2a1a'],     # Dark (near zero positive)
            [0.7, '#00cc44'],      # Neon Green
            [0.85, '#00ff88'],     # Bright Neon Green
            [1.0, '#00ffff'],      # Neon Cyan (very positive)
        ]
        
        fig = go.Figure(go.Heatmap(
            z=z_values, 
            x=column_names, 
            y=years,
            colorscale=neon_colorscale, 
            zmid=0,
            text=text_values,
            texttemplate='%{text}', 
            textfont=dict(
                size=12, 
                color='white',
                family='SF Mono, Consolas, monospace'
            ),
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                title=dict(text='Return %', font=dict(color='#00ff88', size=12)),
                tickfont=dict(color='#aaa', size=10),
                tickformat='.1f',
                outlinecolor='#333',
                outlinewidth=1,
                bgcolor='rgba(20,25,35,0.8)'
            ),
            hovertemplate='<b>%{y} %{x}</b><br>Value: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,15,25,0.95)',
            title=dict(
                text=f"Monthly Returns (%) - {years[0]} to {years[-1]}",
                font=dict(size=16, color='#00ff88'),
                x=0.5,
                xanchor='center'
            ),
            height=max(300, 70 * len(years)),
            yaxis=dict(
                tickmode='array',
                tickvals=years,
                ticktext=[f'<b>{y}</b>' for y in years],
                tickfont=dict(color='#00ffff', size=13, family='SF Mono'),
                autorange='reversed',  # Latest year at top
                gridcolor='rgba(50,60,80,0.3)'
            ),
            xaxis=dict(
                side='top',
                tickangle=0,
                tickfont=dict(color='#a855f7', size=11, family='SF Mono'),
                gridcolor='rgba(50,60,80,0.3)'
            ),
            margin=dict(t=100, b=30, l=60, r=120)
        )
        
        # Add cell borders for better visibility
        fig.update_traces(
            xgap=2,  # Gap between cells
            ygap=2
        )
        
        # Add vertical lines to separate Total and Max DD columns
        fig.add_vline(x=11.5, line_width=2, line_color='rgba(168,85,247,0.5)')  # Before Total
        fig.add_vline(x=12.5, line_width=2, line_color='rgba(168,85,247,0.5)')  # Before Max DD
        
        return fig
    except Exception as e:
        logger.error(f"Monthly heatmap error: {e}")
        return go.Figure()

def create_yearly_returns_chart(rets: pd.Series) -> go.Figure:
    """Create yearly returns bar chart with 2 decimal percentages."""
    try:
        if rets.empty:
            return go.Figure()
        
        yearly = rets.resample('YE').apply(lambda x: (1+x).prod()-1)
        
        if yearly.empty:
            return go.Figure()
        
        yearly_pct = yearly * 100
        years = yearly.index.year.tolist()
        
        colors = ['#00ff88' if v >= 0 else '#ff6b6b' for v in yearly_pct.values]
        
        fig = go.Figure(go.Bar(
            x=years,
            y=yearly_pct.values,
            marker_color=colors,
            text=[f'{v:.2f}%' for v in yearly_pct.values],
            textposition='outside',
            textfont=dict(size=12)
        ))
        
        # Calculate proper y-axis range to fit all bars and labels
        max_val = max(yearly_pct.values)
        min_val = min(yearly_pct.values)
        
        # Add padding for text labels (15% extra space)
        y_max = max_val * 1.2 if max_val > 0 else max_val * 0.8
        y_min = min_val * 1.2 if min_val < 0 else min_val * 0.8
        
        # Ensure some minimum range
        if y_max < 10:
            y_max = max(10, max_val + 5)
        if y_min > -10:
            y_min = min(-10, min_val - 5) if min_val < 0 else 0
        
        fig.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)',
            title="Yearly Returns (%)",
            xaxis_title="Year",
            yaxis_title="Return (%)",
            height=350,
            xaxis=dict(
                tickmode='array',
                tickvals=years,
                ticktext=[str(y) for y in years],
                dtick=1
            ),
            yaxis=dict(
                range=[y_min, y_max],
                autorange=False
            )
        )
        return fig
    except Exception as e:
        logger.error(f"Yearly returns chart error: {e}")
        return go.Figure()

def safe_fmt(val, fmt, fallback="N/A"):
    try:
        if val is None or np.isnan(val) or np.isinf(val): return fallback
        return fmt.format(val)
    except: return fallback

def create_styled_table_html(df: pd.DataFrame, signal_col: str = 'signal') -> str:
    """
    Create styled HTML table with multi-color based on signal type.
    - Blue Neon for STRONG BUY
    - Green Neon for BUY
    - Yellow Neon for HOLD/WATCH
    - Red Neon for AVOID/SELL
    - Purple Neon header text
    - All headers and values centered except Asset (left aligned)
    """
    signal_colors = {
        'STRONG BUY': '#00bfff',  # Blue Neon
        'BUY': '#00ff88',          # Green Neon
        'HOLD': '#ffff00',         # Yellow Neon
        'WATCH': '#ffa500',        # Orange
        'AVOID': '#ff4444',        # Red Neon
        'SELL': '#ff0000'          # Red
    }
    
    html = """
    <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 14px;
            background: linear-gradient(180deg, #151922, #0d1117);
            border-radius: 10px;
            overflow: hidden;
        }
        .styled-table th {
            background: linear-gradient(180deg, #1a1530, #0d0a1a);
            color: #a855f7 !important;
            font-weight: 800;
            padding: 14px 12px;
            text-align: center;
            border-bottom: 2px solid #a855f7;
            text-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
        }
        .styled-table th:first-child {
            text-align: left;
        }
        .styled-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #2a3545;
            color: #e0e0e0;
        }
        .styled-table td:first-child {
            text-align: left;
            font-weight: 600;
        }
        .styled-table tr:hover td {
            background: rgba(168, 85, 247, 0.1);
        }
        .signal-strong-buy { 
            color: #00bfff !important; 
            font-weight: 800; 
            text-shadow: 0 0 8px #00bfff, 0 0 15px #00bfff, 0 0 25px rgba(0, 191, 255, 0.6);
            letter-spacing: 0.5px;
        }
        .signal-buy { 
            color: #00ff88 !important; 
            font-weight: 800; 
            text-shadow: 0 0 8px #00ff88, 0 0 15px #00ff88, 0 0 25px rgba(0, 255, 136, 0.6);
            letter-spacing: 0.5px;
        }
        .signal-hold { 
            color: #ffff00 !important; 
            font-weight: 800; 
            text-shadow: 0 0 8px #ffff00, 0 0 15px #ffff00, 0 0 25px rgba(255, 255, 0, 0.6);
            letter-spacing: 0.5px;
        }
        .signal-watch { 
            color: #ffa500 !important; 
            font-weight: 800; 
            text-shadow: 0 0 8px #ffa500, 0 0 15px #ffa500, 0 0 25px rgba(255, 165, 0, 0.6);
            letter-spacing: 0.5px;
        }
        .signal-avoid { 
            color: #ff4444 !important; 
            font-weight: 800; 
            text-shadow: 0 0 8px #ff4444, 0 0 15px #ff4444, 0 0 25px rgba(255, 68, 68, 0.6);
            letter-spacing: 0.5px;
        }
        .selected-yes { color: #00ff88; }
        .selected-no { color: #666666; }
    </style>
    <table class="styled-table">
    <thead><tr>
    """
    
    # Headers
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    # Rows
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            val = row[col]
            cell_class = ""
            
            if col == signal_col or col == 'signal':
                signal_val = str(val).upper()
                if 'STRONG' in signal_val:
                    cell_class = "signal-strong-buy"
                elif signal_val == 'BUY':
                    cell_class = "signal-buy"
                elif signal_val == 'HOLD':
                    cell_class = "signal-hold"
                elif signal_val == 'WATCH':
                    cell_class = "signal-watch"
                elif signal_val in ['AVOID', 'SELL']:
                    cell_class = "signal-avoid"
            elif col == 'Selected':
                cell_class = "selected-yes" if val else "selected-no"
                val = "✓" if val else "✗"
            
            html += f'<td class="{cell_class}">{val}</td>'
        html += "</tr>"
    
    html += "</tbody></table>"
    return html

# === MAIN UI ===
st.markdown("# ⚡ Quant ETFs Momentum Dashboard")
#st.markdown("*Advanced Portfolio Optimization with 126-Day Momentum & Rebalancing*")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    etf_symbols, symbol_map = load_etf_symbols()
    
    st.markdown("### 📈 Benchmark")
    bench_name = st.selectbox("Benchmark", list(BENCHMARKS.keys()))
    bench_ticker = BENCHMARKS[bench_name]
    
    st.markdown("### 📅 Backtest Period")
    from_date = st.date_input("From", value=date.today() - timedelta(days=730),
                              min_value=date(2010,1,1), max_value=date.today()-timedelta(days=60))
    to_date = st.date_input("To", value=date.today(),
                            min_value=from_date+timedelta(days=30), max_value=date.today())
    
    if from_date >= to_date:
        st.error("❌ Invalid date range!")
        st.stop()
    
    period_days = (to_date - from_date).days
    st.info(f"📊 **{period_days} days** ({from_date} → {to_date})")
    
    st.markdown("### 🎯 Asset Selection")
    sel_model = st.selectbox("Selection Model", [m.value for m in AssetSelectionModel], index=3)
    sel_model_enum = AssetSelectionModel(sel_model)
    
    st.markdown("### 🌐 Asset Universe")
    
    # Universe selection mode
    universe_mode = st.radio(
        "Universe Mode",
        ["Auto-Select by Model", "Manual Selection"],
        index=0,
        horizontal=True,
        help="Auto-Select: Model picks best ETFs from full universe. Manual: You choose specific ETFs."
    )
    
    if universe_mode == "Auto-Select by Model":
        # Use ALL available ETFs as the universe
        st.caption("📊 Model will select from ALL available ETFs based on criteria")
        all_tickers = [symbol_map.get(t, f"{t}.NS") for t in etf_symbols]
        tickers = all_tickers
        selected_display = etf_symbols  # For display purposes
        
        # Show what the model will filter by
        model_criteria = {
            "High Conviction (Top Sharpe + Momentum)": "Top Sharpe + Momentum combined",
            "Low Volatility (< 25% Annual)": "Volatility < 25% annual",
            "Momentum (126-Day Returns > 3%)": "126D returns > 3%",
            "Quality Momentum (Combined Score)": "Combined: 40% Mom + 30% Sharpe + 20% Trend + 10% Low Vol",
            "Relative Strength vs Benchmark": "Outperforming benchmark",
            "Risk-Adjusted (Sharpe > 0.5)": "Sharpe ratio > 0.5",
            "Trend Following (MA Crossover)": "Price above moving averages"
        }
        st.info(f"🎯 **{sel_model}**\n\nCriteria: {model_criteria.get(sel_model, 'Combined scoring')}")
        
    else:
        # Manual selection mode
        default_etfs = ["NIFTYBEES", "GOLDBEES", "SILVERBEES", "MON100", "MAFANG", "MID150BEES", "BANKBEES", "ITBEES",
                        "PHARMABEES", "MOMENTUM50", "FMCGIETF", "LIQUIDBEES"]
        selected_display = st.multiselect("Select ETFs", etf_symbols, default=default_etfs)
        tickers = [symbol_map.get(t, f"{t}.NS") for t in selected_display]
        st.caption(f"📊 {len(selected_display)} ETFs selected → Model will filter to top performers")
    
    st.markdown("### 🔧 Portfolio Optimization")
    opt_model = st.selectbox("Method", OPTIMIZATION_METHODS, index=6)  # Mean-Variance (Max Sharpe)
    
    # Risk tolerance for Tangency/CAL
    risk_tolerance = 1.0
    if opt_model == "Tangency Portfolio (CAL)":
        st.markdown("#### Capital Allocation Line")
        risk_tolerance = st.slider(
            "Risk Tolerance (Risky Asset %)", 
            min_value=0.0, max_value=1.0, value=1.0, step=0.05,
            help="1.0 = 100% Tangency Portfolio, 0.5 = 50% Tangency + 50% Risk-Free"
        )
        rf_alloc = 1.0 - risk_tolerance
        st.caption(f"📊 Risky: {risk_tolerance:.0%} | Risk-Free: {rf_alloc:.0%}")
    
    st.markdown("### 🔄 Rebalance Frequency")
    rebal_freq = st.selectbox("Frequency", [f.value for f in RebalanceFrequency], index=2)
    rebal_freq_enum = RebalanceFrequency(rebal_freq)
    
    # Exit Rank Controls (only show if not Buy & Hold)
    exit_rank_mode = "disabled"
    exit_rank_pct = 50.0
    manual_exit_rank = 10
    
    if rebal_freq_enum != RebalanceFrequency.BUY_HOLD:
        st.markdown("### 🚪 Exit Rank Trigger")
        st.caption("Rebalance only when holdings breach exit rank threshold")
        
        exit_rank_mode = st.radio(
            "Exit Rank Mode",
            ["auto", "manual", "disabled"],
            index=0,
            horizontal=True,
            help="""
            **Auto**: Exit rank = ETFs × (1 + Buffer%). E.g., 5 ETFs with 50% buffer → Exit rank 8
            **Manual**: Set specific exit rank (1-55)
            **Disabled**: Always rebalance on schedule (old behavior)
            """
        )
        
        if exit_rank_mode == "auto":
            exit_rank_pct = st.slider(
                "Buffer %", 
                min_value=25, max_value=100, value=50, step=5,
                help="Exit rank = ETFs × (1 + Buffer/100). 50% means 1.5× multiplier"
            )
            # Show example calculation
            example_etfs = 5
            example_exit = int(np.ceil(example_etfs * (1 + exit_rank_pct/100)))
            st.caption(f"📊 Example: {example_etfs} ETFs → Exit Rank **{example_exit}** | 8 ETFs → **{int(np.ceil(8 * (1 + exit_rank_pct/100)))}** | 10 ETFs → **{int(np.ceil(10 * (1 + exit_rank_pct/100)))}**")
            
        elif exit_rank_mode == "manual":
            manual_exit_rank = st.number_input(
                "Exit Rank", 
                min_value=1, max_value=55, value=10,
                help="Fixed exit rank threshold. Holdings ranked > this will trigger rebalance."
            )
            st.caption(f"📊 If any holding falls to rank **>{manual_exit_rank}**, rebalance is triggered")
        
        else:  # disabled
            st.info("⚡ **Disabled**: Rebalancing will occur on every scheduled date regardless of holdings' ranks")
    
    st.markdown("### 📊 Parameters")
    rf_rate = st.number_input("Risk-Free Rate", value=0.065, step=0.005, format="%.3f")
    
    with st.expander("🔬 Advanced"):
        min_assets = st.number_input("Min Assets", value=5, min_value=4, max_value=10)
        max_assets = st.number_input("Max Assets", value=10, min_value=5, max_value=15)
        min_mom = st.number_input("Min 126D Mom %", value=3.0, min_value=0.0, max_value=8.0, step=0.5) / 100
    
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Main Content
if tickers:
    start_str = from_date.strftime('%Y-%m-%d')
    end_str = to_date.strftime('%Y-%m-%d')
    
    tab1, tab2 = st.tabs(["📊 Portfolio Analysis", "🔍 Opportunity Scanner"])
    
    with tab1:
        try:
            with st.spinner(f"📡 Fetching data for {start_str} to {end_str}..."):
                data, failed = fetch_data(tickers, start_str, end_str)
                benchmark = fetch_benchmark(bench_ticker, start_str, end_str)
            
            if failed:
                with st.expander(f"⚠️ {len(failed)} failed"):
                    st.warning(", ".join(failed))
            
            if data.empty:
                st.error("❌ No data!")
                st.stop()
            
            # Show ACTUAL data range (may differ from requested range)
            actual_start = data.index.min().strftime('%Y-%m-%d')
            actual_end = data.index.max().strftime('%Y-%m-%d')
            actual_days = len(data)
            
            # Count how many assets have data from the start
            start_date = pd.to_datetime(start_str)
            assets_from_start = sum(1 for col in data.columns if data[col].first_valid_index() is not None and data[col].first_valid_index() <= start_date + pd.Timedelta(days=30))
            
            st.markdown(f"### 📊 Data: {actual_start} → {actual_end}")
            
            # Warn if data range differs significantly from requested
            requested_start = pd.to_datetime(start_str)
            actual_start_dt = data.index.min()
            days_difference = (actual_start_dt - requested_start).days
            
            if days_difference > 30:
                st.warning(f"""
                ⚠️ **Data starts later than requested!**  
                - Requested: {start_str}  
                - Actual data starts: {actual_start}  
                - Gap: {days_difference} days  
                
                **{assets_from_start}/{len(data.columns)} ETFs** have data from your requested start date.  
                The backtest will use whatever ETFs are available at each point in time.
                """)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Assets", len(data.columns))
            c2.metric("Trading Days", actual_days)
            c3.metric("From Start", f"{assets_from_start}/{len(data.columns)}")
            c4.metric("Benchmark", bench_name)
            
            with st.spinner("🧮 Scoring assets..."):
                scores = calculate_scores(data, benchmark, min_mom, rf_rate)
            
            if not scores:
                st.error("❌ No scores!")
                st.stop()
            
            # Select assets using the chosen selection model
            selected_assets = select_assets(scores, sel_model_enum, min_assets, max_assets, min_mom)
            
            if len(selected_assets) < 2:
                st.error("❌ Not enough assets selected by the model!")
                st.stop()
            
            with st.expander("📋 Asset Selection", expanded=True):
                # Show universe vs selected info
                if universe_mode == "Auto-Select by Model":
                    st.success(f"🌐 **Universe:** {len(data.columns)} ETFs (all available) → 🎯 **Model Selected:** {len(selected_assets)} assets")
                else:
                    st.info(f"🌐 **Universe:** {len(data.columns)} ETFs (manually chosen) → 🎯 **Model Selected:** {len(selected_assets)} assets")
                
                st.markdown(f"**Selection Model:** {sel_model}")
                
                # Show selected assets prominently
                selected_names = [clean_ticker(a) for a in selected_assets]
                st.markdown(f"**✅ Selected for Optimization:** `{', '.join(selected_names)}`")
                
                scores_df = pd.DataFrame([vars(s) for s in scores])
                scores_df['ticker'] = scores_df['ticker'].apply(clean_ticker)
                scores_df['Selected'] = scores_df['ticker'].isin([clean_ticker(a) for a in selected_assets])
                scores_df = scores_df.sort_values('momentum_126d', ascending=False)
                
                # Format the display dataframe - now with UPI
                display_df = scores_df[['ticker','momentum_126d','sharpe_ratio','upi','ulcer_index','volatility','trend_score','signal','Selected']].copy()
                display_df.columns = ['Asset', '126D Mom', 'Sharpe', 'UPI', 'Ulcer', 'Vol', 'Trend', 'Signal', 'Selected']
                display_df['126D Mom'] = display_df['126D Mom'].apply(lambda x: f"{x:.1%}")
                display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
                display_df['UPI'] = display_df['UPI'].apply(lambda x: f"{x:.2f}")
                display_df['Ulcer'] = display_df['Ulcer'].apply(lambda x: f"{x:.4f}")
                display_df['Vol'] = display_df['Vol'].apply(lambda x: f"{x:.1%}")
                display_df['Trend'] = display_df['Trend'].astype(int)
                
                # Use styled HTML table
                st.markdown(create_styled_table_html(display_df, 'Signal'), unsafe_allow_html=True)
            
            with st.spinner(f"🎯 Optimizing with {opt_model}..."):
                # Use selected_assets from Asset Selection model
                weights, perf, subset, S, mu, extra_info = optimize_portfolio(
                    data, rf_rate, opt_model, selected_assets, risk_tolerance
                )
            
            if not weights:
                st.error("❌ Optimization failed!")
                st.stop()
            
            st.markdown("---")
            st.markdown(f"### 📈 {opt_model}")
            
            # Show CAL details if using Tangency Portfolio
            if opt_model == "Tangency Portfolio (CAL)" and extra_info:
                st.info(f"""
                **Capital Allocation Line:**  
                🎯 Risky Portfolio: {extra_info.get('risky_weight', 1):.0%} | 
                🏦 Risk-Free: {extra_info.get('risk_free_alloc', 0):.0%}  
                📈 Tangency Return: {extra_info.get('tangency_return', 0):.1%} | 
                📉 Tangency Vol: {extra_info.get('tangency_vol', 0):.1%} | 
                ⚡ Tangency Sharpe: {extra_info.get('tangency_sharpe', 0):.2f}
                """)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected Return", safe_fmt(perf[0], "{:.1%}"))
            c2.metric("Volatility", safe_fmt(perf[1], "{:.1%}"))
            c3.metric("Sharpe Ratio", safe_fmt(perf[2], "{:.2f}"))
            c4.metric("Assets", len([w for w in weights.values() if w > 0.001]))
            
            # Allocation Configuration Sliders
            st.markdown("#### ⚙️ Allocation Configuration")
            st.caption("Adjust parameters to explore different portfolio combinations")
            
            alloc_sl1, alloc_sl2, alloc_sl3, alloc_sl4 = st.columns(4)
            
            with alloc_sl1:
                alloc_min_assets = st.slider(
                    "Min Assets", 
                    min_value=2, max_value=8, value=min_assets,
                    help="Minimum assets in portfolio",
                    key="alloc_min_assets"
                )
            
            with alloc_sl2:
                alloc_max_assets = st.slider(
                    "Max Assets", 
                    min_value=4, max_value=15, value=max_assets,
                    help="Maximum assets in portfolio",
                    key="alloc_max_assets"
                )
            
            with alloc_sl3:
                alloc_min_mom_pct = st.slider(
                    "Min Momentum %", 
                    min_value=0.0, max_value=10.0, value=min_mom*100, step=0.5,
                    help="Minimum 126D momentum threshold",
                    key="alloc_min_mom"
                )
                alloc_min_mom = alloc_min_mom_pct / 100
            
            with alloc_sl4:
                alloc_risk_tol = st.slider(
                    "Risk Tolerance", 
                    min_value=0.0, max_value=1.0, value=risk_tolerance, step=0.05,
                    help="Risk tolerance for CAL",
                    key="alloc_risk_tol"
                )
            
            # Re-optimize if parameters changed
            alloc_selected = select_assets(scores, sel_model_enum, alloc_min_assets, alloc_max_assets, alloc_min_mom)
            if len(alloc_selected) >= 2:
                alloc_weights, alloc_perf, alloc_subset, _, _, alloc_extra = optimize_portfolio(
                    data, rf_rate, opt_model, alloc_selected, alloc_risk_tol
                )
            else:
                alloc_weights, alloc_perf, alloc_subset = weights, perf, subset
            
            # Show updated metrics if changed
            if alloc_weights != weights:
                st.markdown("##### 📊 Updated Portfolio Metrics")
                uc1, uc2, uc3, uc4 = st.columns(4)
                uc1.metric("Expected Return", safe_fmt(alloc_perf[0], "{:.1%}"), 
                          delta=f"{(alloc_perf[0]-perf[0])*100:.1f}pp" if perf[0] != 0 else None)
                uc2.metric("Volatility", safe_fmt(alloc_perf[1], "{:.1%}"),
                          delta=f"{(alloc_perf[1]-perf[1])*100:.1f}pp" if perf[1] != 0 else None, delta_color="inverse")
                uc3.metric("Sharpe Ratio", safe_fmt(alloc_perf[2], "{:.2f}"),
                          delta=f"{alloc_perf[2]-perf[2]:.2f}" if perf[2] != 0 else None)
                uc4.metric("Assets", len([w for w in alloc_weights.values() if w > 0.001]))
                
                # Use updated weights for display
                weights = alloc_weights
                perf = alloc_perf
                subset = alloc_subset
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### Allocation")
                wdf = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                wdf.index = wdf.index.map(clean_ticker)
                wdf = wdf[wdf['Weight'] > 0.001].sort_values('Weight', ascending=True)
                fig_alloc = px.bar(wdf, x='Weight', y=wdf.index, orientation='h',
                                   color='Weight', color_continuous_scale=['#004d26','#00ff88'], text_auto='.1%')
                fig_alloc.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_alloc, use_container_width=True)
            
            with col2:
                st.markdown("#### Correlation")
                if len(subset.columns) > 1:
                    corr = subset.corr()
                    corr.index = [clean_ticker(c) for c in corr.index]
                    corr.columns = [clean_ticker(c) for c in corr.columns]
                    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    fig_corr.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("---")
            st.markdown(f"### 📈 Backtest: {from_date} → {to_date}")
            
            # Backtest Configuration with Sliders
            st.markdown("#### ⚙️ Backtest Configuration")
            sl1, sl2, sl3, sl4 = st.columns(4)
            
            with sl1:
                bt_min_assets = st.slider(
                    "Min Assets", 
                    min_value=2, max_value=8, value=min_assets,
                    help="Minimum number of assets in portfolio",
                    key="bt_min_assets"
                )
            
            with sl2:
                bt_max_assets = st.slider(
                    "Max Assets", 
                    min_value=4, max_value=15, value=max_assets,
                    help="Maximum number of assets in portfolio",
                    key="bt_max_assets"
                )
            
            with sl3:
                bt_min_mom_pct = st.slider(
                    "Min Momentum %", 
                    min_value=0.0, max_value=10.0, value=min_mom*100, step=0.5,
                    help="Minimum 126-day momentum threshold",
                    key="bt_min_mom"
                )
                bt_min_mom = bt_min_mom_pct / 100
            
            with sl4:
                bt_risk_tol = st.slider(
                    "Risk Tolerance", 
                    min_value=0.0, max_value=1.0, value=risk_tolerance, step=0.05,
                    help="1.0 = Fully risky, 0.0 = Fully risk-free (for CAL)",
                    key="bt_risk_tol"
                )
            
            st.markdown(f"**Rebalance Frequency:** {rebal_freq}")
            
            # Show exit rank settings if enabled
            if rebal_freq_enum != RebalanceFrequency.BUY_HOLD and exit_rank_mode != "disabled":
                if exit_rank_mode == "auto":
                    example_exit = int(np.ceil(bt_min_assets * (1 + exit_rank_pct/100)))
                    st.markdown(f"**🚪 Exit Rank:** Auto ({exit_rank_pct:.0f}% buffer) → ~{example_exit} for {bt_min_assets} ETFs")
                else:
                    st.markdown(f"**🚪 Exit Rank:** Manual → Threshold: {manual_exit_rank}")
            
            # Show data diagnostics
            data_start = data.index[0].strftime('%Y-%m-%d')
            data_end = data.index[-1].strftime('%Y-%m-%d')
            
            # Count assets available at start vs end
            start_dt = data.index[0]
            assets_at_start = sum(1 for col in data.columns if pd.notna(data[col].iloc[0]))
            assets_at_end = sum(1 for col in data.columns if pd.notna(data[col].iloc[-1]))
            
            st.info(f"""
            📈 **Backtest Data:**  
            - Period: {data_start} → {data_end} ({len(data)} trading days)  
            - Universe: {len(data.columns)} total ETFs  
            - Available at start: {assets_at_start} ETFs | At end: {assets_at_end} ETFs
            """)
            
            # Run Backtest button
            run_backtest_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
            
            if run_backtest_btn or 'backtest_run' not in st.session_state:
                st.session_state.backtest_run = True
                
                # Run backtest based on rebalance frequency
                with st.spinner("🧮 Running backtest..."):
                    if rebal_freq_enum == RebalanceFrequency.BUY_HOLD:
                        equity, port_rets = run_backtest(weights, data, benchmark, 100)
                        rebalance_log = [{'date': data.index[0], 'weights': weights, 'selected': selected_assets, 'perf': perf, 'available': assets_at_start}]
                    else:
                        equity, port_rets, rebalance_log = run_backtest_with_rebalancing(
                            weights_func=None,
                            data=data,
                            benchmark=benchmark,
                            frequency=rebal_freq_enum,
                            initial=100,
                            rf=rf_rate,
                            opt_method=opt_model,
                            selection_model=sel_model_enum,
                            min_assets=bt_min_assets,
                            max_assets=bt_max_assets,
                            min_mom=bt_min_mom,
                            risk_tolerance=bt_risk_tol,
                            exit_rank_mode=exit_rank_mode,
                            exit_rank_pct=exit_rank_pct,
                            manual_exit_rank=manual_exit_rank
                        )
            
                if not equity.empty and len(port_rets) >= 20:
                    # Show backtest summary
                    port_start = port_rets.index[0].strftime('%Y-%m-%d')
                    port_end = port_rets.index[-1].strftime('%Y-%m-%d')
                    
                    # Show rebalance summary with first/last dates
                    if rebalance_log:
                        first_rebal = rebalance_log[0]['date']
                        last_rebal = rebalance_log[-1]['date']
                        first_str = first_rebal.strftime('%Y-%m-%d') if hasattr(first_rebal, 'strftime') else str(first_rebal)
                        last_str = last_rebal.strftime('%Y-%m-%d') if hasattr(last_rebal, 'strftime') else str(last_rebal)
                        
                        # Get available asset counts over time
                        first_available = rebalance_log[0].get('available', '?')
                        last_available = rebalance_log[-1].get('available', '?')
                        
                        # Get exit rank info
                        total_skipped = rebalance_log[-1].get('total_skipped', 0)
                        exit_rank_used = rebalance_log[-1].get('exit_rank', 'N/A')
                        exit_mode_used = rebalance_log[-1].get('exit_rank_mode', 'disabled')
                        
                        # Build summary message
                        summary_msg = f"""
                        ✅ **Backtest Complete:** {port_start} → {port_end} ({len(port_rets)} days)  
                        📊 **{len(rebalance_log)} Rebalances** | First: {first_str} | Last: {last_str}  
                        🌐 **Dynamic Universe:** {first_available} ETFs available at start → {last_available} at end
                        """
                        
                        if exit_mode_used != "disabled":
                            summary_msg += f"""
                        🚪 **Exit Rank Mode:** {exit_mode_used.upper()} | Threshold: {exit_rank_used} | Skipped: {total_skipped} rebalances
                            """
                        
                        st.success(summary_msg)
                        
                        # Show exit rank details if enabled
                        if exit_mode_used != "disabled" and total_skipped > 0:
                            with st.expander(f"🚪 Exit Rank Details ({total_skipped} rebalances skipped)"):
                                skipped_details = rebalance_log[-1].get('skipped_details', [])
                                if skipped_details:
                                    skip_df = pd.DataFrame([{
                                        'Date': s['date'].strftime('%Y-%m-%d') if hasattr(s['date'], 'strftime') else str(s['date']),
                                        'Exit Rank': s['exit_rank'],
                                        'Holdings': s['num_holdings'],
                                        'Max Rank': max(s['holding_ranks'].values()) if s['holding_ranks'] else 'N/A',
                                        'Reason': s['reason']
                                    } for s in skipped_details[-20:]])  # Show last 20
                                    st.dataframe(skip_df, use_container_width=True, hide_index=True)
                                    st.caption("Showing last 20 skipped rebalances. Holdings stayed within exit rank threshold.")
                        
                        # Warn if first rebalance is much later than data start
                        data_start = data.index[0]
                        days_to_first = (first_rebal - data_start).days if hasattr(first_rebal, '__sub__') else 0
                        if days_to_first > 200:
                            st.warning(f"⚠️ First rebalance at {first_str} ({days_to_first} days after data start). Strategy needs 126+ days of historical data for momentum calculation. Pre-rebalance period uses equal-weight proxy of available assets.")
                    else:
                        st.caption(f"📊 Total Rebalances: 0")
                    
                    t1, t2, t3, t4, t5 = st.tabs(["📈 Equity", "📉 Drawdown", "📊 Metrics", "📅 Monthly", "🔄 Rebalances"])
                    
                    with t1:
                        # Use HTML component for true hover highlighting
                        import streamlit.components.v1 as components
                        chart_html = create_equity_chart_with_highlight(equity)
                        components.html(chart_html, height=550, scrolling=False)
                    
                    with t2:
                        cum = (1 + port_rets).cumprod()
                        dd = (cum - cum.cummax()) / cum.cummax() * 100
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy',
                                                    fillcolor='rgba(255,100,100,0.3)', line=dict(color='#ff6464')))
                        fig_dd.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                             title="Drawdown (%)", yaxis_title="DD%")
                        st.plotly_chart(fig_dd, use_container_width=True)
                    
                    with t3:
                        qs.extend_pandas()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("CAGR", safe_fmt(port_rets.cagr(), "{:.2%}"))
                        c2.metric("Max DD", safe_fmt(port_rets.max_drawdown(), "{:.2%}"))
                        c3.metric("Sortino", safe_fmt(port_rets.sortino(), "{:.2f}"))
                        c4.metric("Win Rate", safe_fmt(port_rets.win_rate(), "{:.2%}"))
                        
                        c5, c6, c7, c8 = st.columns(4)
                        c5.metric("Calmar", safe_fmt(port_rets.calmar(), "{:.2f}"))
                        c6.metric("Ulcer Index", safe_fmt(calculate_ulcer_index(port_rets), "{:.4f}"))
                        c7.metric("UPI", safe_fmt(ulcer_performance_index(port_rets, rf_rate), "{:.2f}"))
                        c8.metric("Best Day", safe_fmt(port_rets.max(), "{:.2%}"))
                        
                        # Yearly returns chart
                        st.plotly_chart(create_yearly_returns_chart(port_rets), use_container_width=True)
                    
                    with t4:
                        st.plotly_chart(create_monthly_heatmap(port_rets), use_container_width=True)
                    
                    with t5:
                        st.markdown("#### Rebalance History")
                        if rebalance_log:
                            # Summary table of all rebalances
                            rebal_summary = []
                            for log in rebalance_log:
                                date_str = log['date'].strftime('%Y-%m-%d') if hasattr(log['date'], 'strftime') else str(log['date'])
                                top_assets = [clean_ticker(k) for k, v in sorted(log.get('weights', {}).items(), key=lambda x: -x[1])[:3] if v > 0.001]
                                rebal_summary.append({
                                    'Date': date_str,
                                    'Available': log.get('available', log.get('available_assets', '-')),
                                    'Selected': len(log.get('selected', [])),
                                    'Exit Rank': log.get('exit_rank', '-'),
                                    'Trigger': log.get('trigger_reason', 'Scheduled')[:20],
                                    'Top Holdings': ', '.join(top_assets) if top_assets else '-'
                                })
                            
                            st.dataframe(
                                pd.DataFrame(rebal_summary), 
                                use_container_width=True, 
                                hide_index=True,
                                height=min(400, 35 * len(rebal_summary) + 40)
                            )
                            
                            st.caption("📊 **Available** = ETFs with data | **Selected** = ETFs chosen | **Exit Rank** = Threshold | **Trigger** = Why rebalanced")
                            
                            # Show last 5 in detail
                            st.markdown("##### Last 5 Rebalances (Details)")
                            for i, log in enumerate(rebalance_log[-5:]):
                                date_label = log['date'].strftime('%Y-%m-%d') if hasattr(log['date'], 'strftime') else log['date']
                                exit_info = f" | Exit Rank: {log.get('exit_rank', 'N/A')}" if log.get('exit_rank') else ""
                                trigger_info = f" | {log.get('trigger_reason', '')}" if log.get('trigger_reason') else ""
                                
                                with st.expander(f"📅 {date_label} | Available: {log.get('available', '?')} → Selected: {len(log.get('selected', []))}{exit_info}{trigger_info}"):
                                    st.markdown(f"**Available ETFs:** {log.get('available', '?')}")
                                    st.markdown(f"**Selected Assets:** {len(log.get('selected', []))}")
                                    
                                    # Show exit rank info if present
                                    if log.get('exit_rank_mode') and log.get('exit_rank_mode') != 'disabled':
                                        st.markdown(f"**Exit Rank Mode:** {log.get('exit_rank_mode', 'N/A').upper()}")
                                        st.markdown(f"**Exit Rank Threshold:** {log.get('exit_rank', 'N/A')}")
                                        
                                        if log.get('breached_tickers'):
                                            breached = [clean_ticker(t) for t in log['breached_tickers']]
                                            st.warning(f"🚪 **Breached Tickers:** {', '.join(breached)}")
                                        
                                        if log.get('holding_ranks_before'):
                                            ranks_str = ', '.join([f"{clean_ticker(k)}: #{v}" for k, v in log['holding_ranks_before'].items()])
                                            st.caption(f"📊 Holdings ranks before: {ranks_str}")
                                    
                                    if log.get('weights'):
                                        wdf = pd.DataFrame([
                                            {'Asset': clean_ticker(k), 'Weight': f"{v:.1%}"}
                                            for k, v in sorted(log['weights'].items(), key=lambda x: -x[1])
                                            if v > 0.001
                                        ])
                                        st.dataframe(wdf, use_container_width=True, hide_index=True)
                        else:
                            st.info("No rebalancing (Buy & Hold)")
                else:
                    st.warning("⚠️ Insufficient data for backtest. Need at least 20 days of returns.")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Opportunity Scanner
    with tab2:
        st.markdown("### 🔍 Opportunity Scanner")
        st.markdown("*Select models and scan for opportunities with optimal weights*")
        
        st.markdown("#### ⚙️ Scanner Configuration")
        
        # Show universe mode info
        if universe_mode == "Auto-Select by Model":
            st.caption(f"🌐 Scanning **ALL {len(tickers)} ETFs** in universe")
        else:
            st.caption(f"🌐 Scanning **{len(tickers)} manually selected ETFs**")
        
        sc1, sc2 = st.columns(2)
        
        with sc1:
            scan_sel_model = st.selectbox("🎯 Selection Model", [m.value for m in AssetSelectionModel],
                                          index=3, key="scan_sel")
            scan_sel_enum = AssetSelectionModel(scan_sel_model)
        
        with sc2:
            scan_opt_model = st.selectbox("🔧 Optimization", OPTIMIZATION_METHODS, index=6, key="scan_opt")  # Mean-Variance
        
        # Sliders for scanner parameters
        st.markdown("#### 📊 Scanner Parameters")
        sl1, sl2, sl3, sl4 = st.columns(4)
        
        with sl1:
            scan_min_assets = st.slider(
                "Min Assets", 
                min_value=2, max_value=8, value=2,
                help="Minimum number of assets",
                key="scan_min_assets"
            )
        
        with sl2:
            scan_max_assets = st.slider(
                "Max Assets", 
                min_value=4, max_value=15, value=6,
                help="Maximum number of assets",
                key="scan_max_assets"
            )
        
        with sl3:
            scan_min_mom_pct = st.slider(
                "Min Momentum %", 
                min_value=0.0, max_value=10.0, value=3.0, step=0.5,
                help="Minimum 126-day momentum threshold",
                key="scan_min_mom"
            )
            scan_min_mom = scan_min_mom_pct / 100
        
        with sl4:
            scan_risk_tolerance = st.slider(
                "Risk Tolerance", 
                min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                help="1.0 = Fully risky, 0.0 = Fully risk-free (for CAL)",
                key="scan_risk_tol"
            )
        
        st.markdown("---")
        
        if st.button("🔄 Scan Now", use_container_width=True, type="primary"):
            with st.spinner(f"🔍 Scanning with {scan_sel_model} + {scan_opt_model}..."):
                scan_start = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                scan_end = date.today().strftime('%Y-%m-%d')
                scan_data, _ = fetch_data(tickers, scan_start, scan_end)
                scan_bench = fetch_benchmark(bench_ticker, scan_start, scan_end)
                
                if not scan_data.empty:
                    scan_scores = calculate_scores(scan_data, scan_bench, scan_min_mom, rf_rate)
                    selected_scan = select_assets(scan_scores, scan_sel_enum, scan_min_assets, scan_max_assets, scan_min_mom)
                    
                    scan_weights, scan_perf, scan_extra = {}, (0, 0, 0), {}
                    if len(selected_scan) >= 2:
                        scan_weights, scan_perf, _, _, _, scan_extra = optimize_portfolio(
                            scan_data, rf_rate, scan_opt_model, selected_scan, scan_risk_tolerance
                        )
                    
                    signals = generate_signals(scan_data, scan_scores, scan_bench, scan_weights)
                    
                    st.session_state['signals'] = signals
                    st.session_state['scan_weights'] = scan_weights
                    st.session_state['scan_perf'] = scan_perf
                    st.session_state['scan_selected'] = selected_scan
                    st.session_state['scan_time'] = datetime.now()
                    st.session_state['scan_sel_name'] = scan_sel_model
                    st.session_state['scan_opt_name'] = scan_opt_model
                    st.session_state['scan_extra'] = scan_extra
                else:
                    st.error("❌ No data for scanning")
        
        if 'signals' in st.session_state:
            signals = st.session_state['signals']
            scan_weights = st.session_state.get('scan_weights', {})
            scan_perf = st.session_state.get('scan_perf', (0, 0, 0))
            scan_selected = st.session_state.get('scan_selected', [])
            scan_extra = st.session_state.get('scan_extra', {})
            
            st.success(f"""
            ✅ **Scanned:** {st.session_state.get('scan_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}  
            🌐 **Universe:** {len(signals)} ETFs analyzed  
            🎯 **Selection Model:** {st.session_state.get('scan_sel_name', '')}  
            🔧 **Optimization:** {st.session_state.get('scan_opt_name', '')}  
            ✅ **Model Selected:** {len(scan_selected)} assets → `{', '.join([clean_ticker(t) for t in scan_selected])}`
            """)
            
            st.caption("ℹ️ Selection model picks candidates from universe → Optimization model determines weights (some may get 0%)")
            
            buy = [s for s in signals if s.signal_type in ['STRONG BUY', 'BUY']]
            watch = [s for s in signals if s.signal_type in ['HOLD', 'WATCH']]
            avoid = [s for s in signals if s.signal_type == 'AVOID']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 Buy", len(buy))
            c2.metric("🟡 Watch", len(watch))
            c3.metric("🔴 Avoid", len(avoid))
            c4.metric("✅ Portfolio", len(scan_selected))
            
            if scan_weights and scan_perf[0] != 0:
                st.markdown("---")
                st.markdown("### 📊 Optimized Portfolio")
                
                # Show CAL info if applicable
                if st.session_state.get('scan_opt_name') == "Tangency Portfolio (CAL)" and scan_extra:
                    st.info(f"""
                    **CAL Allocation:** Risky {scan_extra.get('risky_weight', 1):.0%} | Risk-Free {scan_extra.get('risk_free_alloc', 0):.0%}
                    """)
                
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("📈 Expected Return", safe_fmt(scan_perf[0], "{:.1%}"))
                pc2.metric("📉 Volatility", safe_fmt(scan_perf[1], "{:.1%}"))
                pc3.metric("⚡ Sharpe", safe_fmt(scan_perf[2], "{:.2f}"))
                
                # Count assets with allocation vs 0% allocation
                allocated = [k for k, v in scan_weights.items() if v > 0.001]
                zero_alloc = [clean_ticker(t) for t in scan_selected if t not in allocated]
                
                wdf = pd.DataFrame([
                    {'Asset': clean_ticker(k), 'Weight': v}
                    for k, v in sorted(scan_weights.items(), key=lambda x: -x[1]) if v > 0.001
                ])
                if not wdf.empty:
                    col_c, col_t = st.columns([2, 1])
                    with col_c:
                        fig_w = px.bar(wdf, x='Weight', y='Asset', orientation='h',
                                      color='Weight', color_continuous_scale=['#1a472a','#00ff88'],
                                      text=wdf['Weight'].apply(lambda x: f"{x:.1%}"))
                        fig_w.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                           height=300, showlegend=False, title="Allocation")
                        st.plotly_chart(fig_w, use_container_width=True)
                    with col_t:
                        st.dataframe(wdf.assign(**{'Weight %': wdf['Weight'].apply(lambda x: f"{x:.1%}")})[['Asset','Weight %']],
                                    use_container_width=True, hide_index=True)
                        if zero_alloc:
                            st.caption(f"⚪ 0% allocated: {', '.join(zero_alloc)}")
                
                # Risk Metrics Comparison Table - shows WHY weights were assigned
                st.markdown("#### 📊 Risk Metrics Comparison (Why These Weights?)")
                st.caption("Lower Ulcer Index = Smaller drawdowns = Higher weight in UPI optimization")
                
                # Build comparison dataframe from signals for selected assets
                selected_signals = [s for s in signals if s.ticker in scan_selected]
                if selected_signals:
                    comparison_df = pd.DataFrame([{
                        'Asset': clean_ticker(s.ticker),
                        'Weight': scan_weights.get(s.ticker, 0),
                        '126D Mom': s.momentum_126d,
                        'Sharpe': s.sharpe_ratio,
                        'Ulcer Idx': s.ulcer_index,
                        'UPI': s.upi,
                        'Max DD': s.max_drawdown
                    } for s in selected_signals])
                    
                    # Sort by weight descending
                    comparison_df = comparison_df.sort_values('Weight', ascending=False)
                    
                    # Format for display
                    display_comp = comparison_df.copy()
                    display_comp['Weight'] = display_comp['Weight'].apply(lambda x: f"{x:.1%}")
                    display_comp['126D Mom'] = display_comp['126D Mom'].apply(lambda x: f"{x:.1%}")
                    display_comp['Sharpe'] = display_comp['Sharpe'].apply(lambda x: f"{x:.2f}")
                    display_comp['Ulcer Idx'] = display_comp['Ulcer Idx'].apply(lambda x: f"{x:.4f}")
                    display_comp['UPI'] = display_comp['UPI'].apply(lambda x: f"{x:.2f}")
                    display_comp['Max DD'] = display_comp['Max DD'].apply(lambda x: f"{x:.1%}")
                    
                    # Create styled HTML table
                    st.markdown(create_styled_table_html(display_comp, 'Signal'), unsafe_allow_html=True)
                    
                    # Explanation based on optimization method
                    opt_name = st.session_state.get('scan_opt_name', '')
                    if 'UPI' in opt_name or 'Ulcer' in opt_name:
                        st.info("""
                        **UPI Optimization Logic:**  
                        • Assets with **lower Ulcer Index** (smaller drawdowns) get **higher weights**  
                        • High momentum with big drawdowns → penalized  
                        • Stable returns with small drawdowns → rewarded  
                        • The optimizer finds the combination that maximizes (Return - RF) / Ulcer Index
                        """)
                    elif 'Sharpe' in opt_name or 'Mean-Variance' in opt_name:
                        st.info("""
                        **Mean-Variance (Sharpe) Logic:**  
                        • Maximizes (Expected Return - RF) / Volatility  
                        • Assets with higher Sharpe ratios tend to get more weight  
                        • Correlation between assets also matters for diversification
                        """)
                    elif 'HRP' in opt_name:
                        st.info("""
                        **Hierarchical Risk Parity Logic:**  
                        • Uses clustering to group similar assets  
                        • Allocates inversely to cluster variance  
                        • Doesn't rely on expected returns, only on covariance
                        """)
            
            st.markdown("---")
            
            fc1, fc2, fc3 = st.columns([2, 1, 1])
            with fc1:
                sig_filter = st.multiselect("Filter", ["STRONG BUY", "BUY", "HOLD", "WATCH", "AVOID"],
                                            default=["STRONG BUY", "BUY"], key="sig_f")
            with fc2:
                min_str = st.slider("Min Strength", 0, 100, 0, key="min_s")
            with fc3:
                only_portfolio = st.checkbox("Portfolio Only", value=False, key="only_p")
            
            filtered = [s for s in signals if s.signal_type in sig_filter and s.strength >= min_str]
            if only_portfolio:
                filtered = [s for s in filtered if s.ticker in scan_selected]
            
            st.markdown(f"### 📋 Opportunities ({len(filtered)} shown)")
            st.caption("Sorted by Weight (descending), then Strength")
            
            for sig in filtered:
                emoji = "🟢" if sig.signal_type in ["STRONG BUY", "BUY"] else "🟡" if sig.signal_type in ["HOLD", "WATCH"] else "🔴"
                in_port = sig.ticker in scan_selected
                
                with st.container():
                    c1, c2, c3, c4 = st.columns([1.2, 1.8, 1.2, 1.1])
                    
                    with c1:
                        st.markdown(f"### {emoji} {clean_ticker(sig.ticker)}")
                        st.markdown(f"**{sig.signal_type}**")
                        st.progress(sig.strength / 100)
                        st.caption(f"Strength: {sig.strength}%")
                        if in_port:
                            st.success("✅ Portfolio")
                    
                    with c2:
                        st.markdown(f"""
                        | Metric | Value |
                        |--------|-------|
                        | 126D Mom | {sig.momentum_126d:.1%} |
                        | Sharpe | {sig.sharpe_ratio:.2f} |
                        | **Ulcer Idx** | **{sig.ulcer_index:.4f}** |
                        | **UPI** | **{sig.upi:.2f}** |
                        | Max DD | {sig.max_drawdown:.1%} |
                        | Trend | {sig.trend_status} |
                        """)
                    
                    with c3:
                        st.markdown(f"""
                        **Price:** ₹{sig.current_price:.2f}
                        
                        **Levels:**
                        - 🎯 Target: ₹{sig.target_price:.2f}
                        - 🛑 Stop: ₹{sig.stop_loss:.2f}
                        - Support: ₹{sig.support_level:.2f}
                        - Resist: ₹{sig.resistance_level:.2f}
                        """)
                    
                    with c4:
                        st.markdown("**Weight:**")
                        if sig.weight > 0.001:  # More than 0.1%
                            st.markdown(f"### 📊 {sig.weight:.1%}")
                            st.progress(min(sig.weight * 2.5, 1.0))
                        elif in_port:
                            # In portfolio but optimizer gave ~0% weight
                            st.markdown("### ⚪ 0.0%")
                            st.caption("_In portfolio, 0% allocated_")
                        else:
                            st.markdown("*Not in portfolio*")
                    
                    st.caption(f"💡 {sig.rationale}")
                    st.markdown("---")
            
            if filtered:
                exp_df = pd.DataFrame([{
                    'Ticker': clean_ticker(s.ticker), 'Signal': s.signal_type,
                    'Weight': f"{s.weight:.1%}" if s.weight > 0 else "-",
                    'Strength': s.strength, '126D Mom': f"{s.momentum_126d:.1%}",
                    'Sharpe': f"{s.sharpe_ratio:.2f}", 
                    'Ulcer Index': f"{s.ulcer_index:.4f}",
                    'UPI': f"{s.upi:.2f}",
                    'Max DD': f"{s.max_drawdown:.1%}",
                    'Price': s.current_price,
                    'Target': s.target_price, 'Stop': s.stop_loss
                } for s in filtered])
                st.download_button("📥 Download CSV", exp_df.to_csv(index=False),
                                   "opportunities.csv", use_container_width=True)
        else:
            st.info("👆 Configure and click 'Scan Now'")
            st.markdown("""
            ### 📊 Optimization Methods Available
            
            | Method | Description |
            |--------|-------------|
            | **Mean-Variance (Max Sharpe)** | Maximize risk-adjusted returns |
            | **Tangency Portfolio (CAL)** | Optimal mix of Risky + Risk-Free assets |
            | **Global Min Variance** | Minimize portfolio volatility |
            | **HRP** | Hierarchical clustering allocation |
            | **Risk Parity** | Equal risk contribution |
            | **Black-Litterman** | Incorporate market views |
            | **Mean-Semivariance** | Focus on downside risk |
            | **Inverse Volatility** | Weight by 1/volatility |
            | **Most Diversified** | Maximize diversification ratio |
            | **Equal Weight** | Simple 1/N allocation |
            | **Ulcer Performance** | Optimize for drawdown-adjusted returns |
            """)

else:
    st.info("👈 Select assets in sidebar")
    st.markdown("""
    ### ⚡ Quant ETFs Momentum Dashboard
    
    **Features:**
    - 11 Portfolio Optimization Methods
    - 126-Day Momentum Scoring
    - **Rebalancing:** Buy & Hold, Fortnightly, Monthly, Quarterly
    - **Tangency/CAL:** Optimal Risky + Risk-Free allocation
    - Real-time Opportunity Scanner
    - Custom Date Range Backtesting
    
    **Get Started:**
    1. Select benchmark & date range
    2. Choose ETFs from universe
    3. Pick asset selection model
    4. Choose optimization method
    5. Set rebalance frequency
    6. Run analysis or scan opportunities
    """)
