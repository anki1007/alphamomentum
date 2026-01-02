"""
================================================================================
MULTI ASSET MOMENTUM RISK PARITY STRATEGY - STREAMLIT DASHBOARD
================================================================================
Implements the "MULTI ASSET MOMO UC RP" strategy from factorlab's 
Adaptive Asset Allocation case study using Yahoo Finance data.

Framework from PDF:
-------------------
1. Universe: 4 uncorrelated assets
   - GOLDBEES (Gold ETF) 
   - NIFTYBEES (Nifty 50 ETF)
   - MON100 (NASDAQ 100 ETF)
   - LIQUIDBEES (Liquid Fund)

2. Strategy Logic (Monthly):
   - Calculate 6-month momentum for each asset
   - Select TOP 2 by momentum rank
   - Weight by INVERSE VOLATILITY (Risk Parity)
   - Hold for next month (lagged execution)

Expected Results: CAGR ~16.6%, Sharpe ~1.58, Max DD ~-8.9%

Reference: https://medium.com/@factorlab/adaptive-asset-allocation-case-study
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Multi Asset Momentum RP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0d1117 100%);
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #12121f);
        border: 1px solid #2d2d4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b8b8b;
        text-transform: uppercase;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .header-subtitle {
        color: #6b7280;
        text-align: center;
        font-size: 1rem;
    }
    .success-box {
        background: rgba(0,255,136,0.1);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background: rgba(255,165,2,0.1);
        border: 1px solid #ffa502;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ASSET CONFIGURATIONS
# =============================================================================

# Indian ETF Universe (from PDF)
INDIAN_ASSETS = {
    'GOLDBEES': {'ticker': 'GOLDBEES.NS', 'name': 'Nippon India Gold BeES', 'class': 'Commodity'},
    'NIFTYBEES': {'ticker': 'NIFTYBEES.NS', 'name': 'Nippon India Nifty 50 BeES', 'class': 'Equity'},
    'MON100': {'ticker': 'MON100.NS', 'name': 'Motilal Oswal NASDAQ 100 ETF', 'class': 'Equity'},
    'LIQUIDBEES': {'ticker': 'LIQUIDBEES.NS', 'name': 'Nippon India Liquid BeES', 'class': 'Debt'},
}

# US ETF Universe (alternative)
US_ASSETS = {
    'GOLD': {'ticker': 'GLD', 'name': 'SPDR Gold Trust', 'class': 'Commodity'},
    'SPY': {'ticker': 'SPY', 'name': 'S&P 500 ETF', 'class': 'Equity'},
    'QQQ': {'ticker': 'QQQ', 'name': 'Invesco QQQ (NASDAQ)', 'class': 'Equity'},
    'BIL': {'ticker': 'BIL', 'name': 'SPDR T-Bill ETF', 'class': 'Debt'},
}

# Global Diversified Universe
GLOBAL_ASSETS = {
    'GOLD': {'ticker': 'GLD', 'name': 'Gold', 'class': 'Commodity'},
    'VTI': {'ticker': 'VTI', 'name': 'Total US Stock Market', 'class': 'Equity'},
    'VEA': {'ticker': 'VEA', 'name': 'Developed Markets', 'class': 'Equity'},
    'BND': {'ticker': 'BND', 'name': 'Total Bond Market', 'class': 'Debt'},
}


# =============================================================================
# DATA FETCHER
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_yahoo_data(tickers: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from Yahoo Finance and resample to monthly."""
    all_data = {}
    errors = []
    
    for name, info in tickers.items():
        try:
            ticker = yf.Ticker(info['ticker'])
            df = ticker.history(start=start_date, end=end_date)
            if len(df) > 0:
                all_data[name] = df['Close']
            else:
                errors.append(f"{name}: No data available")
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
    
    if not all_data:
        raise ValueError(f"No data fetched. Errors: {'; '.join(errors)}")
    
    # Combine and process
    prices = pd.DataFrame(all_data)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    
    # Forward fill and resample to monthly
    prices = prices.ffill()
    monthly = prices.resample('ME').last().dropna()
    
    return monthly, errors


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class MomentumRiskParity:
    """Multi Asset Momentum Risk Parity Strategy."""
    
    def __init__(self, momentum_lookback: int = 6, vol_lookback: int = 6, 
                 top_n: int = 2, risk_free_rate: float = 0.06):
        self.momentum_lookback = momentum_lookback
        self.vol_lookback = vol_lookback
        self.top_n = top_n
        self.risk_free_rate = risk_free_rate
        self.results = None
        self.metrics = None
    
    def run(self, prices: pd.DataFrame) -> Dict:
        """Run backtest."""
        assets = prices.columns.tolist()
        
        # Calculate signals
        monthly_returns = prices.pct_change()
        momentum = prices.pct_change(self.momentum_lookback)
        volatility = monthly_returns.rolling(self.vol_lookback).std()
        
        # Backtest
        results = []
        nav = 100.0
        max_nav = 100.0
        prev_weights = None
        
        start_idx = max(self.momentum_lookback, self.vol_lookback)
        
        for i in range(start_idx, len(prices)):
            date = prices.index[i]
            mom_values = momentum.iloc[i]
            vol_values = volatility.iloc[i]
            
            if mom_values.isna().any() or vol_values.isna().any():
                continue
            
            # Rank and select
            ranks = mom_values.rank(ascending=False).astype(int)
            selected = (ranks <= self.top_n).astype(int)
            
            # Inverse vol weights
            inv_vol = (1.0 / vol_values) * selected
            current_weights = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else pd.Series(0, index=vol_values.index)
            
            month_returns = monthly_returns.iloc[i]
            
            # Calculate return using previous weights (lagged execution)
            if prev_weights is not None:
                port_return = (prev_weights * month_returns).sum()
                nav = nav * (1 + port_return)
                max_nav = max(max_nav, nav)
                drawdown = (nav - max_nav) / max_nav
            else:
                port_return = 0.0
                drawdown = 0.0
            
            result_row = {
                'Date': date, 'NAV': nav, 'Monthly_Return': port_return,
                'Max_NAV': max_nav, 'Drawdown': drawdown,
            }
            for asset in assets:
                result_row[f'{asset}_Weight'] = current_weights.get(asset, 0)
                result_row[f'{asset}_Rank'] = ranks.get(asset, 0)
                result_row[f'{asset}_Selected'] = selected.get(asset, 0)
            
            results.append(result_row)
            prev_weights = current_weights
        
        self.results = pd.DataFrame(results).set_index('Date')
        self._calculate_metrics()
        
        return {'metrics': self.metrics, 'history': self.results}
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        h = self.results
        total_months = len(h)
        total_years = total_months / 12
        final_nav = h['NAV'].iloc[-1]
        
        cagr = (final_nav / 100) ** (1 / total_years) - 1
        vol = h['Monthly_Return'].std() * np.sqrt(12)
        sharpe = (cagr - self.risk_free_rate) / vol if vol > 0 else 0
        
        downside = h['Monthly_Return'][h['Monthly_Return'] < 0]
        downside_std = downside.std() * np.sqrt(12) if len(downside) > 0 else 0
        sortino = (cagr - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        max_dd = h['Drawdown'].min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        win_rate = (h['Monthly_Return'] > 0).sum() / total_months
        
        self.metrics = {
            'CAGR': cagr, 'Volatility': vol, 'Sharpe': sharpe, 'Sortino': sortino,
            'Calmar': calmar, 'Max_DD': max_dd, 'Win_Rate': win_rate,
            'Best_Month': h['Monthly_Return'].max(), 'Worst_Month': h['Monthly_Return'].min(),
            'Final_NAV': final_nav, 'Total_Return': final_nav/100 - 1,
            'Total_Months': total_months, 'Total_Years': total_years,
            'Start_Date': h.index[0], 'End_Date': h.index[-1],
            'Skewness': h['Monthly_Return'].skew(), 'Kurtosis': h['Monthly_Return'].kurtosis(),
        }
    
    def get_allocation(self) -> Dict[str, float]:
        """Get current allocation."""
        if self.results is None:
            return {}
        last = self.results.iloc[-1]
        return {col.replace('_Weight', ''): last[col] 
                for col in self.results.columns if col.endswith('_Weight') and last[col] > 0.001}


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_nav_chart(history: pd.DataFrame) -> go.Figure:
    """Create equity curve chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history['NAV'],
        mode='lines', name='Portfolio NAV',
        line=dict(color='#00ff88', width=2),
        fill='tozeroy', fillcolor='rgba(0,255,136,0.1)'
    ))
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date', yaxis_title='NAV',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400, margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


def create_drawdown_chart(history: pd.DataFrame) -> go.Figure:
    """Create drawdown chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=history['Drawdown'] * 100,
        mode='lines', name='Drawdown',
        line=dict(color='#ff4757', width=2),
        fill='tozeroy', fillcolor='rgba(255,71,87,0.3)'
    ))
    fig.update_layout(
        title='Drawdown Profile',
        xaxis_title='Date', yaxis_title='Drawdown (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    return fig


def create_allocation_chart(allocation: Dict[str, float]) -> go.Figure:
    """Create allocation pie chart."""
    colors = ['#00ff88', '#00d4ff', '#a855f7', '#ffa502']
    fig = go.Figure(data=[go.Pie(
        labels=list(allocation.keys()),
        values=[v * 100 for v in allocation.values()],
        hole=0.6, marker=dict(colors=colors[:len(allocation)]),
        textinfo='label+percent', textfont=dict(size=14)
    )])
    fig.update_layout(
        title='Current Allocation',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_monthly_heatmap(history: pd.DataFrame) -> go.Figure:
    """Create monthly returns heatmap."""
    df = history[['Monthly_Return']].copy()
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    pivot = df.pivot_table(values='Monthly_Return', index='Year', columns='Month', aggfunc='sum') * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        y=pivot.index,
        colorscale=[[0, '#ff4757'], [0.5, '#1a1a2e'], [1, '#00ff88']],
        text=np.round(pivot.values, 1), texttemplate='%{text}%', textfont=dict(size=10),
    ))
    fig.update_layout(
        title='Monthly Returns Heatmap (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400, margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def create_correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    """Create correlation matrix heatmap."""
    corr = prices.pct_change().dropna().corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, '#ff4757'], [0.5, '#1a1a2e'], [1, '#00ff88']],
        text=np.round(corr.values, 2), texttemplate='%{text}', textfont=dict(size=12),
        zmin=-1, zmax=1
    ))
    fig.update_layout(
        title='Asset Correlation Matrix (Monthly Returns)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400, margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="header-title">📈 Multi Asset Momentum Risk Parity</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Adaptive Asset Allocation | Top N Momentum + Inverse Volatility Weighting</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Strategy Parameters")
        
        # Asset Universe Selection
        universe = st.selectbox(
            "Asset Universe",
            ["Indian ETFs (NSE)", "US ETFs", "Global Diversified"],
            help="Select the asset universe for backtesting"
        )
        
        if universe == "Indian ETFs (NSE)":
            assets = INDIAN_ASSETS
        elif universe == "US ETFs":
            assets = US_ASSETS
        else:
            assets = GLOBAL_ASSETS
        
        st.markdown("**Selected Assets:**")
        for name, info in assets.items():
            st.caption(f"• {name}: {info['name']}")
        
        st.markdown("---")
        
        # Strategy Parameters
        momentum_lookback = st.slider("Momentum Lookback (months)", 3, 12, 6,
                                      help="Period for momentum calculation")
        vol_lookback = st.slider("Volatility Lookback (months)", 3, 12, 6,
                                 help="Period for volatility calculation")
        top_n = st.slider("Top N Assets", 1, len(assets), 2,
                          help="Number of top momentum assets to select")
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.0, 0.5) / 100
        
        st.markdown("---")
        
        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime(2010, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        st.markdown("---")
        
        run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Main Content
    if run_backtest or 'results' in st.session_state:
        
        if run_backtest:
            with st.spinner("Fetching data from Yahoo Finance..."):
                try:
                    prices, errors = fetch_yahoo_data(
                        assets, 
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                    
                    if errors:
                        st.warning(f"⚠️ Some assets had issues: {', '.join(errors)}")
                    
                    if len(prices.columns) < 2:
                        st.error("❌ Need at least 2 assets for the strategy. Please try a different universe.")
                        return
                    
                    st.session_state['prices'] = prices
                    
                except Exception as e:
                    st.error(f"❌ Error fetching data: {e}")
                    st.info("💡 Try selecting a different asset universe (US ETFs tend to be more reliable)")
                    return
            
            with st.spinner("Running backtest..."):
                strategy = MomentumRiskParity(
                    momentum_lookback=momentum_lookback,
                    vol_lookback=vol_lookback,
                    top_n=min(top_n, len(st.session_state['prices'].columns)),
                    risk_free_rate=risk_free_rate
                )
                results = strategy.run(st.session_state['prices'])
                st.session_state['results'] = results
                st.session_state['strategy'] = strategy
        
        # Display Results
        if 'results' in st.session_state:
            results = st.session_state['results']
            strategy = st.session_state['strategy']
            metrics = results['metrics']
            history = results['history']
            
            # Metrics Row
            cols = st.columns(6)
            metric_items = [
                ("📈 CAGR", f"{metrics['CAGR']*100:.1f}%"),
                ("📊 Volatility", f"{metrics['Volatility']*100:.1f}%"),
                ("⚡ Sharpe", f"{metrics['Sharpe']:.2f}"),
                ("🎯 Sortino", f"{metrics['Sortino']:.2f}"),
                ("📉 Max DD", f"{metrics['Max_DD']*100:.1f}%"),
                ("💰 Final NAV", f"{metrics['Final_NAV']:.1f}"),
            ]
            for col, (label, value) in zip(cols, metric_items):
                col.metric(label, value)
            
            st.markdown("---")
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Performance", "📊 Allocation", "🗓️ Monthly Returns", 
                "🔗 Correlation", "📋 Data"
            ])
            
            with tab1:
                st.plotly_chart(create_nav_chart(history), use_container_width=True)
                st.plotly_chart(create_drawdown_chart(history), use_container_width=True)
                
                # Additional stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Win Rate", f"{metrics['Win_Rate']*100:.1f}%")
                col2.metric("Best Month", f"{metrics['Best_Month']*100:.1f}%")
                col3.metric("Worst Month", f"{metrics['Worst_Month']*100:.1f}%")
                col4.metric("Calmar Ratio", f"{metrics['Calmar']:.2f}")
            
            with tab2:
                col1, col2 = st.columns([1, 1])
                with col1:
                    allocation = strategy.get_allocation()
                    if allocation:
                        st.plotly_chart(create_allocation_chart(allocation), use_container_width=True)
                    else:
                        st.info("No current allocation (strategy not invested)")
                
                with col2:
                    st.markdown("### Current Holdings")
                    allocation = strategy.get_allocation()
                    for asset, weight in sorted(allocation.items(), key=lambda x: -x[1]):
                        st.progress(weight, text=f"{asset}: {weight*100:.1f}%")
                    
                    st.markdown("### Strategy Summary")
                    st.markdown(f"""
                    - **Period**: {metrics['Start_Date'].strftime('%Y-%m-%d')} to {metrics['End_Date'].strftime('%Y-%m-%d')}
                    - **Duration**: {metrics['Total_Years']:.1f} years ({metrics['Total_Months']} months)
                    - **Total Return**: {metrics['Total_Return']*100:.1f}%
                    - **Skewness**: {metrics['Skewness']:.2f}
                    - **Kurtosis**: {metrics['Kurtosis']:.2f}
                    """)
            
            with tab3:
                st.plotly_chart(create_monthly_heatmap(history), use_container_width=True)
                
                # Yearly returns
                yearly = history['NAV'].resample('YE').last().pct_change().dropna() * 100
                yearly_df = pd.DataFrame({
                    'Year': yearly.index.year,
                    'Return (%)': yearly.values
                }).set_index('Year')
                st.dataframe(yearly_df.T.round(1), use_container_width=True)
            
            with tab4:
                if 'prices' in st.session_state:
                    st.plotly_chart(create_correlation_heatmap(st.session_state['prices']), use_container_width=True)
                    
                    st.markdown("### Correlation Insights from PDF Framework")
                    st.markdown("""
                    The PDF emphasizes that **correlation is key** to the strategy's success:
                    
                    - Low/negative correlation between assets enables true diversification
                    - High correlation pairs (like NIFTYBEES/JUNIORBEES at 0.87) should be avoided
                    - Gold typically has low correlation with equities (around 0)
                    - Bonds/Liquid funds provide stability with near-zero correlation to equities
                    
                    The "zig-zag" effect: two assets can both have positive returns but negative 
                    correlation, creating a smoother portfolio return path.
                    """)
            
            with tab5:
                st.markdown("### Backtest History (Last 24 Months)")
                display_cols = ['NAV', 'Monthly_Return', 'Drawdown'] + [c for c in history.columns if '_Weight' in c]
                st.dataframe(history[display_cols].tail(24).round(4), use_container_width=True)
                
                # Downloads
                col1, col2 = st.columns(2)
                with col1:
                    csv = history.to_csv()
                    st.download_button(
                        "📥 Download Full History (CSV)",
                        csv, "momentum_rp_backtest.csv", "text/csv",
                        use_container_width=True
                    )
                with col2:
                    metrics_df = pd.DataFrame([metrics])
                    st.download_button(
                        "📥 Download Metrics (CSV)",
                        metrics_df.to_csv(index=False), "metrics.csv", "text/csv",
                        use_container_width=True
                    )
            
            # Comparison with PDF
            st.markdown("---")
            st.markdown("### 📚 Comparison with PDF Benchmark (2014-2024)")
            
            col1, col2, col3, col4 = st.columns(4)
            benchmarks = [
                ("CAGR", metrics['CAGR']*100, 16.6),
                ("Volatility", metrics['Volatility']*100, 10.47),
                ("Sharpe", metrics['Sharpe'], 1.58),
                ("Max DD", metrics['Max_DD']*100, -8.91),
            ]
            for col, (name, actual, expected) in zip([col1, col2, col3, col4], benchmarks):
                delta = actual - expected
                col.metric(
                    f"{name}", 
                    f"{actual:.2f}{'%' if 'DD' in name or name in ['CAGR', 'Volatility'] else ''}",
                    f"{delta:+.2f} vs PDF"
                )
    
    else:
        # Welcome screen
        st.markdown("""
        ### 🎯 About This Strategy
        
        This dashboard implements the **Multi Asset Momentum Risk Parity** strategy from 
        [factorlab's Adaptive Asset Allocation case study](https://medium.com/@factorlab).
        
        #### Strategy Logic (from PDF):
        1. **Universe**: 4 uncorrelated assets (Gold, Equity, International Equity, Debt)
        2. **Signal**: 6-month momentum (returns)
        3. **Selection**: Top 2 assets by momentum rank
        4. **Weighting**: Inverse volatility (Risk Parity)
        5. **Rebalancing**: Monthly with 1-month lag
        
        #### Key Insights from PDF:
        - Equal weight diversification improved Sharpe from 0.80 to 1.37 vs NIFTY buy-and-hold
        - Removing highly correlated assets (JUNIORBEES) improved returns significantly
        - Volatility-adjusted weighting further improved Sharpe to 1.58
        - Max drawdown reduced from -29% (NIFTY) to -9% (final strategy)
        
        #### Expected Results (2014-2024):
        | Metric | Value |
        |--------|-------|
        | CAGR | 16.6% |
        | Volatility | 10.5% |
        | Sharpe Ratio | 1.58 |
        | Max Drawdown | -8.9% |
        
        👈 **Configure parameters in the sidebar and click "Run Backtest" to begin!**
        """)


if __name__ == '__main__':
    main()
