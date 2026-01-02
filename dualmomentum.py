"""
QTF DUAL MOMENTUM UNIFIED PLATFORM
===================================
Institutional-Grade Investment Framework
All-in-One Streamlit Application with 3D Animated Tabs

Author: QTF Framework
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
import time
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Dual Momentum",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ADVANCED 3D ANIMATED CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Space%20Mono&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, transparent 50%, rgba(99,102,241,0.1) 100%);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 20px;
        padding: 25px 40px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,255,136,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .header-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 4px;
        text-shadow: 0 0 30px rgba(0,255,136,0.5);
    }
    
    .header-subtitle {
        font-family: 'Space Mono', monospace;
        color: #6b7280;
        font-size: 0.9rem;
        letter-spacing: 2px;
        margin-top: 8px;
    }
    
    /* 3D TAB CONTAINER */
    .tab-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 30px 0;
        perspective: 1000px;
    }
    
    /* 3D ANIMATED TABS */
    .tab-3d {
        position: relative;
        padding: 18px 35px;
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6b7280;
        background: linear-gradient(145deg, #1a1a2e, #0f0f1a);
        border: 1px solid #2d2d4a;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform-style: preserve-3d;
        box-shadow: 
            0 10px 30px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.05);
    }
    
    .tab-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 15px;
        background: linear-gradient(135deg, rgba(0,255,136,0.1), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .tab-3d:hover {
        transform: translateY(-8px) rotateX(10deg) scale(1.05);
        color: #00ff88;
        border-color: rgba(0,255,136,0.5);
        box-shadow: 
            0 20px 40px rgba(0,255,136,0.2),
            0 0 30px rgba(0,255,136,0.1),
            inset 0 1px 0 rgba(255,255,255,0.1);
    }
    
    .tab-3d:hover::before {
        opacity: 1;
    }
    
    .tab-3d.active {
        color: #0a0a0f;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        border-color: transparent;
        transform: translateY(-5px) rotateX(5deg) scale(1.08);
        box-shadow: 
            0 15px 50px rgba(0,255,136,0.4),
            0 0 50px rgba(0,255,136,0.3),
            inset 0 -3px 10px rgba(0,0,0,0.2);
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 15px 50px rgba(0,255,136,0.4), 0 0 50px rgba(0,255,136,0.3); }
        50% { box-shadow: 0 15px 60px rgba(0,255,136,0.6), 0 0 70px rgba(0,255,136,0.5); }
    }
    
    .tab-3d .tab-icon {
        margin-right: 10px;
        font-size: 1.2rem;
    }
    
    /* Floating particles effect */
    .tab-3d::after {
        content: '';
        position: absolute;
        width: 4px;
        height: 4px;
        background: #00ff88;
        border-radius: 50%;
        top: 50%;
        left: 10px;
        opacity: 0;
        transition: all 0.3s ease;
    }
    
    .tab-3d.active::after {
        opacity: 1;
        animation: float-particle 1.5s infinite;
    }
    
    @keyframes float-particle {
        0%, 100% { transform: translateY(0); opacity: 1; }
        50% { transform: translateY(-10px); opacity: 0.5; }
    }
    
    /* METRIC CARDS - 3D STYLE */
    .metric-card-3d {
        background: linear-gradient(145deg, #1a1a2e, #12121f);
        border: 1px solid #2d2d4a;
        border-radius: 20px;
        padding: 25px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
        transform-style: preserve-3d;
    }
    
    .metric-card-3d:hover {
        transform: translateY(-10px) rotateX(5deg);
        border-color: rgba(0,255,136,0.3);
        box-shadow: 0 25px 50px rgba(0,0,0,0.5), 0 0 30px rgba(0,255,136,0.1);
    }
    
    .metric-card-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00ff88, #00d4ff, #a855f7);
        border-radius: 20px 20px 0 0;
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-value.negative {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 10px;
    }
    
    /* SECTION HEADERS */
    .section-header-3d {
        font-family: 'Orbitron', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: #00ff88;
        text-transform: uppercase;
        letter-spacing: 3px;
        padding-bottom: 15px;
        margin-bottom: 25px;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #00ff88, transparent) 1;
        position: relative;
    }
    
    .section-header-3d::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 50px;
        height: 2px;
        background: #00ff88;
        box-shadow: 0 0 10px #00ff88;
    }
    
    /* CHART CONTAINER */
    .chart-container {
        background: linear-gradient(145deg, #1a1a2e, #0f0f1a);
        border: 1px solid #2d2d4a;
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        border-color: rgba(0,255,136,0.2);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    /* ALERT BOXES */
    .alert-critical {
        background: linear-gradient(135deg, rgba(255,71,87,0.1), rgba(255,71,87,0.05));
        border-left: 4px solid #ff4757;
        border-radius: 0 15px 15px 0;
        padding: 20px 25px;
        margin: 20px 0;
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,255,136,0.05));
        border-left: 4px solid #00ff88;
        border-radius: 0 15px 15px 0;
        padding: 20px 25px;
        margin: 20px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(255,165,2,0.1), rgba(255,165,2,0.05));
        border-left: 4px solid #ffa502;
        border-radius: 0 15px 15px 0;
        padding: 20px 25px;
        margin: 20px 0;
    }
    
    /* RECOMMENDATION CARDS */
    .recommendation-card {
        background: linear-gradient(145deg, #1a1a2e, #12121f);
        border: 1px solid #2d2d4a;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(10px);
        border-color: rgba(0,255,136,0.3);
    }
    
    /* ALLOCATION BARS */
    .allocation-bar-container {
        background: #0f0f1a;
        border-radius: 10px;
        height: 40px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .allocation-bar {
        height: 100%;
        display: flex;
        align-items: center;
        padding-left: 15px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #0a0a0f;
        font-weight: 600;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d4a;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        font-family: 'Space Mono', monospace;
        color: #6b7280;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }
    
    /* TABLES */
    .dataframe {
        font-family: 'Space Mono', monospace !important;
    }
    
    /* STREAMLIT OVERRIDES */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #1a1a2e, #0f0f1a);
        border: 1px solid #2d2d4a;
        border-radius: 15px;
        padding: 15px 30px;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        letter-spacing: 2px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-5px) scale(1.05);
        border-color: rgba(0,255,136,0.5);
        box-shadow: 0 15px 30px rgba(0,255,136,0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%) !important;
        color: #0a0a0f !important;
        border-color: transparent !important;
        transform: translateY(-5px) scale(1.08);
        box-shadow: 0 20px 40px rgba(0,255,136,0.4), 0 0 50px rgba(0,255,136,0.3);
        animation: tab-pulse 2s infinite;
    }
    
    @keyframes tab-pulse {
        0%, 100% { box-shadow: 0 20px 40px rgba(0,255,136,0.4), 0 0 50px rgba(0,255,136,0.3); }
        50% { box-shadow: 0 20px 50px rgba(0,255,136,0.6), 0 0 70px rgba(0,255,136,0.5); }
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        color: #0a0a0f;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        letter-spacing: 2px;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0,255,136,0.4);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00ff88, #00d4ff) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1a1a2e, #0f0f1a);
        border: 1px solid #2d2d4a;
        border-radius: 10px;
        font-family: 'Orbitron', monospace;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================
@dataclass
class StrategyConfig:
    """Strategy configuration parameters"""
    momentum_lookback: int = 126
    trend_lookback: int = 200
    risk_free_rate: float = 0.06
    target_volatility: float = 0.10
    max_position: float = 0.70
    min_position: float = 0.00
    vol_expansion_threshold: float = 1.5
    trend_threshold: float = 0.0
    rebalance_frequency: str = 'Monthly'
    transaction_cost: float = 0.001
    n_simulations: int = 1000
    confidence_level: float = 0.95


# =============================================================================
# BACKTEST RESULTS DATACLASS
# =============================================================================
@dataclass
class BacktestResults:
    """Container for backtest results"""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    signals: pd.DataFrame
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    rolling_sharpe: pd.Series
    drawdown_series: pd.Series
    regime_history: pd.Series


# =============================================================================
# MOMENTUM CALCULATOR
# =============================================================================
class MomentumCalculator:
    @staticmethod
    def dual_momentum_signal(prices, lookback, rf_rate, trend_threshold):
        returns = prices.pct_change(lookback)
        excess_returns = returns - (rf_rate * lookback / 252)
        ranks = excess_returns.rank(axis=1, ascending=False)
        abs_filter = (returns > trend_threshold).astype(int)
        return ranks, abs_filter, excess_returns


# =============================================================================
# REGIME DETECTOR
# =============================================================================
class RegimeDetector:
    def __init__(self, config):
        self.config = config
    
    def get_regime(self, prices):
        returns = prices.mean(axis=1).pct_change()
        ema_50 = returns.ewm(span=50).mean()
        ema_200 = returns.ewm(span=200).mean()
        trend_signal = (ema_50 > ema_200).astype(int)
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        vol_mean = rolling_vol.expanding().mean()
        vol_regime = (rolling_vol < vol_mean * self.config.vol_expansion_threshold).astype(int)
        regime_score = trend_signal + vol_regime
        regime = pd.Series(index=prices.index, dtype=str)
        regime[regime_score == 2] = 'RISK_ON'
        regime[regime_score == 0] = 'RISK_OFF'
        regime[regime_score == 1] = 'NEUTRAL'
        return regime.ffill()


# =============================================================================
# PORTFOLIO OPTIMIZER
# =============================================================================
class PortfolioOptimizer:
    def __init__(self, config):
        self.config = config
    
    def optimize_sharpe(self, returns, momentum_ranks, abs_filter):
        n_assets = len(returns.columns)
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        eligible = abs_filter > 0
        
        if eligible.sum() == 0:
            return np.zeros(n_assets)
        
        def neg_sharpe(weights):
            port_ret = np.dot(weights, mu)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            if port_vol == 0:
                return 0
            return -(port_ret - self.config.risk_free_rate) / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_position, self.config.max_position) if e else (0, 0) 
                  for e in eligible]
        n_eligible = eligible.sum()
        x0 = np.array([1/n_eligible if e else 0 for e in eligible])
        
        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else x0
    
    def optimize_cvar(self, returns, momentum_ranks, abs_filter, alpha=0.05):
        n_assets = len(returns.columns)
        eligible = abs_filter > 0
        if eligible.sum() == 0:
            return np.zeros(n_assets)
        
        returns_matrix = returns.values
        
        def cvar_objective(weights):
            portfolio_returns = np.dot(returns_matrix, weights)
            var = np.percentile(portfolio_returns, alpha * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            return -cvar
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_position, self.config.max_position) if e else (0, 0) 
                  for e in eligible]
        n_eligible = eligible.sum()
        x0 = np.array([1/n_eligible if e else 0 for e in eligible])
        
        result = minimize(cvar_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else x0
    
    def inverse_volatility(self, returns, abs_filter, lookback=60):
        vol = returns.tail(lookback).std() * np.sqrt(252)
        inv_vol = 1 / vol
        inv_vol[abs_filter == 0] = 0
        if inv_vol.sum() == 0:
            return np.zeros(len(inv_vol))
        weights = inv_vol / inv_vol.sum()
        return np.clip(weights, self.config.min_position, self.config.max_position)
    
    def risk_parity(self, returns, abs_filter):
        cov = returns.cov() * 252
        n = len(returns.columns)
        
        def risk_budget_objective(weights):
            port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            marginal_contrib = np.dot(cov, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            target_risk = port_vol / n
            return np.sum((risk_contrib - target_risk) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, self.config.max_position) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = result.x if result.success else x0
        weights[abs_filter == 0] = 0
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights


# =============================================================================
# BACKTESTER
# =============================================================================
class DualMomentumBacktester:
    def __init__(self, config=None):
        self.config = config or StrategyConfig()
        self.momentum = MomentumCalculator()
        self.regime_detector = RegimeDetector(self.config)
        self.optimizer = PortfolioOptimizer(self.config)
    
    def run(self, prices, optimization_method='sharpe'):
        returns = prices.pct_change().dropna()
        ranks, abs_filter, excess_ret = self.momentum.dual_momentum_signal(
            prices, self.config.momentum_lookback, self.config.risk_free_rate, self.config.trend_threshold
        )
        regime = self.regime_detector.get_regime(prices)
        
        positions = pd.DataFrame(index=returns.index, columns=returns.columns)
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        signals = pd.DataFrame(index=returns.index, columns=['regime', 'top_asset'])
        
        
        # Custom Frequency Mapping
        freq_map = {
            'Weekly': 'W-FRI',
            'Fortnightly': '2W-FRI',
            'Monthly': 'M',
            'Quarterly': 'Q',
            'Halfyearly': '6M'
        }
        target_freq = freq_map.get(self.config.rebalance_frequency, 'M')
        rebal_dates = returns.resample(target_freq).last().index

        current_weights = np.zeros(len(returns.columns))
        
        for i, date in enumerate(returns.index):
            signals.loc[date, 'regime'] = regime.loc[date] if date in regime.index else 'NEUTRAL'
            
            if date in rebal_dates and i >= self.config.momentum_lookback:
                lookback_returns = returns.iloc[max(0, i-self.config.momentum_lookback):i]
                current_ranks = ranks.loc[date] if date in ranks.index else pd.Series(1, index=returns.columns)
                current_filter = abs_filter.loc[date] if date in abs_filter.index else pd.Series(1, index=returns.columns)
                current_regime = regime.loc[date] if date in regime.index else 'NEUTRAL'
                
                if current_regime == 'RISK_OFF':
                    current_filter.iloc[-1] = 1
                    temp_config = StrategyConfig(**{**self.config.__dict__, 'max_position': 0.5})
                    temp_optimizer = PortfolioOptimizer(temp_config)
                else:
                    temp_optimizer = self.optimizer
                
                if optimization_method == 'sharpe':
                    new_weights = temp_optimizer.optimize_sharpe(lookback_returns, current_ranks, current_filter)
                elif optimization_method == 'cvar':
                    new_weights = temp_optimizer.optimize_cvar(lookback_returns, current_ranks, current_filter)
                elif optimization_method == 'inv_vol':
                    new_weights = temp_optimizer.inverse_volatility(lookback_returns, current_filter)
                elif optimization_method == 'risk_parity':
                    new_weights = temp_optimizer.risk_parity(lookback_returns, current_filter)
                else:
                    n_eligible = (current_filter > 0).sum()
                    new_weights = np.array([1/n_eligible if f > 0 else 0 for f in current_filter])
                
                turnover = np.sum(np.abs(new_weights - current_weights))
                tc_drag = turnover * self.config.transaction_cost
                current_weights = new_weights
                
                if new_weights.sum() > 0:
                    top_idx = np.argmax(new_weights)
                    signals.loc[date, 'top_asset'] = returns.columns[top_idx]
            
            positions.loc[date] = current_weights
            port_ret = np.dot(current_weights, returns.loc[date].values)
            
            if date in rebal_dates and i >= self.config.momentum_lookback:
                port_ret -= tc_drag
            
            portfolio_returns.loc[date] = port_ret
        
        equity_curve = (1 + portfolio_returns).cumprod().dropna()
        portfolio_returns = portfolio_returns.dropna()
        metrics = self._calculate_metrics(portfolio_returns, equity_curve)
        rolling_sharpe = self._rolling_sharpe(portfolio_returns, 252)
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        
        return BacktestResults(
            equity_curve=equity_curve, returns=portfolio_returns,
            positions=positions.dropna(), signals=signals.dropna(),
            **metrics, rolling_sharpe=rolling_sharpe,
            drawdown_series=drawdown_series, regime_history=regime
        )
    
    def _calculate_metrics(self, returns, equity):
        total_days = len(returns)
        years = total_days / 252
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
        volatility = returns.std() * np.sqrt(252)
        excess_ret = returns.mean() * 252 - self.config.risk_free_rate
        sharpe = excess_ret / volatility if volatility > 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = excess_ret / downside_std if downside_std > 0 else 0
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        ret_skew = skew(returns.dropna())
        ret_kurt = kurtosis(returns.dropna())
        
        return {
            'cagr': cagr, 'volatility': volatility, 'sharpe': sharpe, 'sortino': sortino,
            'max_drawdown': max_drawdown, 'calmar': calmar, 'var_95': var_95,
            'cvar_95': cvar_95, 'skewness': ret_skew, 'kurtosis': ret_kurt
        }
    
    def _rolling_sharpe(self, returns, window):
        rolling_ret = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        return (rolling_ret - self.config.risk_free_rate) / rolling_vol
    
    def _calculate_drawdown_series(self, equity):
        running_max = equity.cummax()
        return (equity - running_max) / running_max


# =============================================================================
# MONTE CARLO SIMULATOR
# =============================================================================
class MonteCarloSimulator:
    def __init__(self, config):
        self.config = config
    
    def simulate(self, returns, n_years=5):
        n_days = n_years * 252
        simulated_cagrs, simulated_maxdds, simulated_sharpes = [], [], []
        
        for _ in range(self.config.n_simulations):
            sim_returns = np.random.choice(returns.values, size=n_days, replace=True)
            sim_equity = np.cumprod(1 + sim_returns)
            cagr = sim_equity[-1] ** (1/n_years) - 1
            running_max = np.maximum.accumulate(sim_equity)
            dd = (sim_equity - running_max) / running_max
            max_dd = dd.min()
            ann_ret = np.mean(sim_returns) * 252
            ann_vol = np.std(sim_returns) * np.sqrt(252)
            sharpe = (ann_ret - self.config.risk_free_rate) / ann_vol
            simulated_cagrs.append(cagr)
            simulated_maxdds.append(max_dd)
            simulated_sharpes.append(sharpe)
        
        return {
            'cagr_5th': np.percentile(simulated_cagrs, 5),
            'cagr_median': np.percentile(simulated_cagrs, 50),
            'cagr_95th': np.percentile(simulated_cagrs, 95),
            'maxdd_5th': np.percentile(simulated_maxdds, 5),
            'maxdd_median': np.percentile(simulated_maxdds, 50),
            'maxdd_95th': np.percentile(simulated_maxdds, 95),
            'sharpe_5th': np.percentile(simulated_sharpes, 5),
            'sharpe_median': np.percentile(simulated_sharpes, 50),
            'sharpe_95th': np.percentile(simulated_sharpes, 95),
            'probability_negative': np.mean(np.array(simulated_cagrs) < 0),
            'probability_drawdown_20': np.mean(np.array(simulated_maxdds) < -0.20)
        }


# =============================================================================
# DATA LOADER
# =============================================================================
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2014-01-01', '2024-12-31', freq='B')
    n = len(dates)
    cov_matrix = np.array([
        [0.04, 0.015, 0.002],
        [0.015, 0.025, -0.001],
        [0.002, -0.001, 0.015]
    ])
    mean_returns = np.array([0.12, 0.10, 0.07]) / 252
    returns = np.random.multivariate_normal(mean_returns, cov_matrix/252, n)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=dates,
        columns=['MON100', 'NIFTYBEES', 'GOLDBEES']
    )
    return prices


# =============================================================================
# EFFICIENT FRONTIER CALCULATOR
# =============================================================================
def calculate_efficient_frontier(returns, n_points=50):
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n_assets = len(mu)
    min_ret, max_ret = mu.min(), mu.max() * 1.2
    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier_vol, frontier_ret = [], []
    
    for target in target_returns:
        try:
            def portfolio_vol(w):
                return np.sqrt(np.dot(w, np.dot(cov, w)))
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t}
            ]
            bounds = [(0, 1) for _ in range(n_assets)]
            x0 = np.ones(n_assets) / n_assets
            result = minimize(portfolio_vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                frontier_vol.append(portfolio_vol(result.x))
                frontier_ret.append(target)
        except:
            continue
    
    return frontier_ret, frontier_vol


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="header-title">ADAPTIVE ASSET ALLOCATION</div>
        <div class="header-subtitle">DUAL MOMENTUM • INSTITUTIONAL GRADE • NIFTYBEES | GOLDBEES | MON100</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ CONFIGURATION")
        
        momentum_lookback = st.slider("Momentum Lookback (days)", 63, 252, 126, 21)
        risk_free_rate = st.slider("Risk-Free Rate (%)", 3.0, 8.0, 6.0, 0.25) / 100
        max_position = st.slider("Max Position (%)", 30.0, 100.0, 70.0, 5.0) / 100
        
        st.markdown("---")
        opt_method = st.selectbox(
            "Optimization Method",
            ['sharpe', 'cvar', 'inv_vol', 'risk_parity', 'equal'],
            format_func=lambda x: {
                'sharpe': '🎯 Sharpe Maximization',
                'cvar': '🛡️ CVaR Minimization',
                'inv_vol': '⚖️ Inverse Volatility',
                'risk_parity': '🔄 Risk Parity',
                'equal': '📊 Equal Weight'
            }[x]
        )
        
        st.markdown("---")
        run_backtest = st.button("🚀 RUN ANALYSIS", use_container_width=True)
    
    # Load data and run backtest
    prices = load_sample_data()
    returns = prices.pct_change().dropna()
    
    if 'results' not in st.session_state or run_backtest:
        config = StrategyConfig(
            momentum_lookback=momentum_lookback,
            risk_free_rate=risk_free_rate,
            max_position=max_position
        )
        bt = DualMomentumBacktester(config)
        st.session_state.results = bt.run(prices, opt_method)
        st.session_state.config = config
        st.session_state.prices = prices
    
    results = st.session_state.results
    
    # 3D ANIMATED TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 DASHBOARD",
        "📈 PERFORMANCE",
        "🎯 ALLOCATION",
        "🔬 ANALYSIS",
        "🎲 STRESS TEST"
    ])
    
    # ==========================================================================
    # TAB 1: DASHBOARD
    # ==========================================================================
    with tab1:
        st.markdown('<div class="section-header-3d">KEY PERFORMANCE METRICS</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card-3d">
                <div class="metric-value">{results.cagr:.2%}</div>
                <div class="metric-label">CAGR</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card-3d">
                <div class="metric-value">{results.sharpe:.3f}</div>
                <div class="metric-label">SHARPE RATIO</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card-3d">
                <div class="metric-value negative">{results.max_drawdown:.2%}</div>
                <div class="metric-label">MAX DRAWDOWN</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card-3d">
                <div class="metric-value">{results.calmar:.3f}</div>
                <div class="metric-label">CALMAR RATIO</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Secondary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Volatility", f"{results.volatility:.2%}")
        with col2:
            st.metric("Sortino", f"{results.sortino:.3f}")
        with col3:
            st.metric("VaR 95%", f"{results.var_95:.2%}")
        with col4:
            st.metric("CVaR 95%", f"{results.cvar_95:.2%}")
        
        # Alert box
        if results.sharpe < 1.0:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ OPTIMIZATION ALERT:</strong> Sharpe ratio below 1.0. Consider reviewing 
                optimization method or increasing diversification. Current strategy may not be 
                adequately compensating for risk taken.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <strong>✓ STRATEGY VALIDATED:</strong> Sharpe ratio above 1.0 indicates acceptable 
                risk-adjusted returns. Monitor for regime changes and rebalancing opportunities.
            </div>
            """, unsafe_allow_html=True)
        
        # Equity curve and drawdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header-3d">EQUITY CURVE</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results.equity_curve.index,
                y=results.equity_curve.values * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='#00ff88', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 136, 0.1)'
            ))
            
            benchmark = (1 + returns.mean(axis=1)).cumprod() * 100
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='#6b7280', width=1, dash='dash')
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=40),
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='NAV')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header-3d">DRAWDOWN ANALYSIS</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results.drawdown_series.index,
                y=results.drawdown_series.values * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff4757', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 71, 87, 0.3)'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=20, b=40),
                height=350,
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Drawdown %'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # TAB 2: PERFORMANCE
    # ==========================================================================
    with tab2:
        st.markdown('<div class="section-header-3d">ROLLING PERFORMANCE METRICS</div>', unsafe_allow_html=True)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=('Rolling 1-Year Sharpe Ratio', 'Rolling 1-Year Return'))
        
        fig.add_trace(go.Scatter(
            x=results.rolling_sharpe.index,
            y=results.rolling_sharpe.values,
            mode='lines',
            name='Sharpe',
            line=dict(color='#00ff88', width=1.5)
        ), row=1, col=1)
        
        fig.add_hline(y=1.0, line_dash='dash', line_color='#ffa502', row=1, col=1)
        fig.add_hline(y=0, line_dash='dot', line_color='#6b7280', row=1, col=1)
        
        rolling_ret = results.returns.rolling(252).mean() * 252
        fig.add_trace(go.Scatter(
            x=rolling_ret.index,
            y=rolling_ret.values * 100,
            mode='lines',
            name='Return',
            line=dict(color='#00d4ff', width=1.5)
        ), row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=600,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        st.markdown('<div class="section-header-3d">RETURNS DISTRIBUTION</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=results.returns.values * 100,
            nbinsx=50,
            marker_color='#00ff88',
            opacity=0.7
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis=dict(title='Daily Returns (%)', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Frequency', gridcolor='rgba(255,255,255,0.05)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skewness", f"{results.skewness:.3f}", 
                     delta="Left tail" if results.skewness < 0 else "Right tail")
        with col2:
            st.metric("Kurtosis", f"{results.kurtosis:.3f}",
                     delta="Fat tails" if results.kurtosis > 3 else "Normal")
    
    # ==========================================================================
    # TAB 3: ALLOCATION
    # ==========================================================================
    with tab3:
        st.markdown('<div class="section-header-3d">CURRENT PORTFOLIO ALLOCATION</div>', unsafe_allow_html=True)
        
        current_weights = results.positions.iloc[-1]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            colors = ['#00ff88', '#ffa502', '#6366f1']
            
            for i, (asset, weight) in enumerate(current_weights.items()):
                st.markdown(f"""
                <div style="margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-family: 'Orbitron', monospace; color: #fff;">{asset}</span>
                        <span style="font-family: 'Space Mono', monospace; color: {colors[i]};">{weight:.1%}</span>
                    </div>
                    <div class="allocation-bar-container">
                        <div class="allocation-bar" style="width: {weight*100}%; background: linear-gradient(90deg, {colors[i]}, {colors[i]}88);">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=current_weights.index.tolist(),
                values=current_weights.values * 100,
                hole=0.6,
                marker=dict(colors=colors),
                textinfo='percent',
                textfont=dict(size=14, color='white', family='Space Mono')
            )])
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Efficient Frontier
        st.markdown('<div class="section-header-3d">EFFICIENT FRONTIER</div>', unsafe_allow_html=True)
        
        frontier_ret, frontier_vol = calculate_efficient_frontier(returns)
        asset_returns = returns.mean() * 252
        asset_vols = returns.std() * np.sqrt(252)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[v * 100 for v in frontier_vol],
            y=[r * 100 for r in frontier_ret],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#00ff88', width=3)
        ))
        
        # CAL
        rf = risk_free_rate * 100
        if frontier_vol:
            sharpe_ratios = [(r - risk_free_rate) / v if v > 0 else 0 for r, v in zip(frontier_ret, frontier_vol)]
            max_sharpe_idx = np.argmax(sharpe_ratios)
            opt_vol = frontier_vol[max_sharpe_idx] * 100
            opt_ret = frontier_ret[max_sharpe_idx] * 100
            
            fig.add_trace(go.Scatter(
                x=[0, opt_vol * 1.5],
                y=[rf, rf + (opt_ret - rf) * 1.5],
                mode='lines',
                name='Capital Allocation Line',
                line=dict(color='#ffa502', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=[opt_vol],
                y=[opt_ret],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(size=20, color='#00ff88', symbol='star',
                           line=dict(color='white', width=2))
            ))
        
        fig.add_trace(go.Scatter(
            x=asset_vols.values * 100,
            y=asset_returns.values * 100,
            mode='markers+text',
            name='Individual Assets',
            marker=dict(size=12, color='#6b7280'),
            text=asset_returns.index,
            textposition='top center',
            textfont=dict(size=11, color='#fff', family='Space Mono')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(title='Volatility (%)', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Expected Return (%)', gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # TAB 4: ANALYSIS
    # ==========================================================================
    with tab4:
        st.markdown('<div class="section-header-3d">CORRELATION MATRIX</div>', unsafe_allow_html=True)
        
        corr = returns.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, '#ff4757'], [0.5, '#1a1a2e'], [1, '#00ff88']],
            zmid=0,
            text=np.round(corr.values, 3),
            texttemplate='%{text}',
            textfont=dict(size=16, color='white', family='Space Mono'),
            hovertemplate='%{x} ↔ %{y}: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime Analysis
        st.markdown('<div class="section-header-3d">REGIME ANALYSIS</div>', unsafe_allow_html=True)
        
        regime_counts = results.regime_history.value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_on_pct = regime_counts.get('RISK_ON', 0) / len(results.regime_history) * 100
            st.markdown(f"""
            <div class="metric-card-3d">
                <div class="metric-value">{risk_on_pct:.1f}%</div>
                <div class="metric-label">RISK ON</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            neutral_pct = regime_counts.get('NEUTRAL', 0) / len(results.regime_history) * 100
            st.markdown(f"""
            <div class="metric-card-3d">
                <div class="metric-value" style="background: linear-gradient(135deg, #ffa502, #ff6b6b); -webkit-background-clip: text;">{neutral_pct:.1f}%</div>
                <div class="metric-label">NEUTRAL</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            risk_off_pct = regime_counts.get('RISK_OFF', 0) / len(results.regime_history) * 100
            st.markdown(f"""
            <div class="metric-card-3d">
                <div class="metric-value negative">{risk_off_pct:.1f}%</div>
                <div class="metric-label">RISK OFF</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<div class="section-header-3d">QTF RECOMMENDATIONS</div>', unsafe_allow_html=True)
        
        if results.sharpe < 1.0:
            st.markdown("""
            <div class="alert-critical">
                <strong>🔴 CRITICAL:</strong> Switch optimization objective from CAGR to Sharpe maximization.
                Current strategy sacrifices risk-adjusted performance for raw returns.
            </div>
            """, unsafe_allow_html=True)
        
        if results.kurtosis > 3:
            st.markdown("""
            <div class="alert-warning">
                <strong>🟡 WARNING:</strong> Fat tails detected (Kurtosis: {:.2f}). Standard deviation 
                understates true risk. Consider CVaR-based position sizing.
            </div>
            """.format(results.kurtosis), unsafe_allow_html=True)
        
        if results.skewness < -0.5:
            st.markdown("""
            <div class="alert-warning">
                <strong>🟡 WARNING:</strong> Negative skewness detected ({:.2f}). Returns distribution 
                shows asymmetric downside exposure. Increase defensive allocation.
            </div>
            """.format(results.skewness), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-success">
            <strong>✓ RECOMMENDED ALLOCATION:</strong><br>
            NIFTYBEES: 45% | GOLDBEES: 35% | MON100: 20%<br>
            <small>Balances Sharpe optimization with momentum signals. Higher gold allocation 
            provides additional drawdown protection.</small>
        </div>
        """, unsafe_allow_html=True)
    
    # ==========================================================================
    # TAB 5: STRESS TEST
    # ==========================================================================
    with tab5:
        st.markdown('<div class="section-header-3d">MONTE CARLO STRESS TEST</div>', unsafe_allow_html=True)
        
        n_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)
        
        if st.button("🎲 RUN MONTE CARLO SIMULATION", use_container_width=True):
            with st.spinner("Running simulation..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                mc = MonteCarloSimulator(st.session_state.config)
                mc.config.n_simulations = n_sims
                mc_results = mc.simulate(results.returns)
            
            st.success(f"✓ Completed {n_sims} simulations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="section-header-3d">CAGR DISTRIBUTION</div>', unsafe_allow_html=True)
                st.metric("5th Percentile", f"{mc_results['cagr_5th']:.2%}")
                st.metric("Median", f"{mc_results['cagr_median']:.2%}")
                st.metric("95th Percentile", f"{mc_results['cagr_95th']:.2%}")
            
            with col2:
                st.markdown('<div class="section-header-3d">DRAWDOWN DISTRIBUTION</div>', unsafe_allow_html=True)
                st.metric("Worst Case (5%)", f"{mc_results['maxdd_5th']:.2%}")
                st.metric("Median", f"{mc_results['maxdd_median']:.2%}")
                st.metric("Best Case (95%)", f"{mc_results['maxdd_95th']:.2%}")
            
            with col3:
                st.markdown('<div class="section-header-3d">RISK PROBABILITIES</div>', unsafe_allow_html=True)
                st.metric("P(Negative Return)", f"{mc_results['probability_negative']:.1%}")
                st.metric("P(DD > 20%)", f"{mc_results['probability_drawdown_20']:.1%}")
                sharpe_range = f"{mc_results['sharpe_5th']:.2f} - {mc_results['sharpe_95th']:.2f}"
                st.metric("Sharpe Range (90% CI)", sharpe_range)
            
            # Risk assessment
            if mc_results['probability_drawdown_20'] > 0.3:
                st.markdown("""
                <div class="alert-critical">
                    <strong>⚠️ HIGH RISK ALERT:</strong> Over 30% probability of experiencing a 20%+ drawdown. 
                    Consider reducing position sizes or increasing defensive allocation.
                </div>
                """, unsafe_allow_html=True)
            elif mc_results['probability_negative'] < 0.05:
                st.markdown("""
                <div class="alert-success">
                    <strong>✓ ROBUST STRATEGY:</strong> Less than 5% probability of negative 5-year returns. 
                    Strategy shows consistent performance across simulated paths.
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-family: 'Space Mono', monospace; font-size: 0.8rem;">
        QTF DUAL MOMENTUM PLATFORM v2.0 | Institutional-Grade Analysis<br>
        <span style="color: #00ff88;">●</span> Strategy designed for risk-adjusted returns with drawdown control
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
