"""
================================================================================
MULTI ASSET MOMENTUM RISK PARITY STRATEGY - STREAMLIT DASHBOARD
================================================================================
Replicates the "MULTI ASSET MOMO UC RP" strategy from factorlab's 
Adaptive Asset Allocation case study.

Deploy on Streamlit Cloud: Include data file in repo OR use file uploader.

Strategy Rules:
---------------
1. Universe: 4 uncorrelated assets (GOLDBEES, NIFTYBEES, MON100, ASBL)
2. At end of each month:
   - Calculate 6-month momentum for each asset
   - Select TOP 2 momentum assets
   - Weight by INVERSE VOLATILITY (Risk Parity)
3. Hold weights for the NEXT month

Reference: https://medium.com/@factorlab/adaptive-asset-allocation-case-study
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import io


# Page config
st.set_page_config(
    page_title="Multi Asset Momentum RP",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
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
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,255,136,0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b8b8b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .header-subtitle {
        color: #6b7280;
        text-align: center;
        font-size: 1rem;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class StrategyConfig:
    """Configuration for the momentum risk parity strategy."""
    momentum_lookback: int = 6
    vol_lookback: int = 6
    top_n: int = 2
    initial_nav: float = 100.0


class MultiAssetMomoRP:
    """Multi Asset Momentum Risk Parity Backtester."""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.results = None
        self.metrics = None
        
    def load_data_from_file(self, uploaded_file) -> pd.DataFrame:
        """Load asset prices from uploaded Excel file."""
        df = pd.read_excel(uploaded_file, sheet_name='MULTI ASSET MOMO UC RP', header=None)
        data_rows = df.iloc[2:].reset_index(drop=True)
        
        prices = pd.DataFrame({
            'Date': pd.to_datetime(data_rows.iloc[:, 0], errors='coerce'),
            'GOLDBEES': pd.to_numeric(data_rows.iloc[:, 1], errors='coerce'),
            'NIFTYBEES': pd.to_numeric(data_rows.iloc[:, 2], errors='coerce'),
            'MON100': pd.to_numeric(data_rows.iloc[:, 3], errors='coerce'),
            'ASBL': pd.to_numeric(data_rows.iloc[:, 4], errors='coerce')
        })
        prices = prices.dropna().reset_index(drop=True)
        prices = prices.set_index('Date')
        return prices
    
    def load_data_from_path(self, filepath: str) -> pd.DataFrame:
        """Load asset prices from file path."""
        df = pd.read_excel(filepath, sheet_name='MULTI ASSET MOMO UC RP', header=None)
        data_rows = df.iloc[2:].reset_index(drop=True)
        
        prices = pd.DataFrame({
            'Date': pd.to_datetime(data_rows.iloc[:, 0], errors='coerce'),
            'GOLDBEES': pd.to_numeric(data_rows.iloc[:, 1], errors='coerce'),
            'NIFTYBEES': pd.to_numeric(data_rows.iloc[:, 2], errors='coerce'),
            'MON100': pd.to_numeric(data_rows.iloc[:, 3], errors='coerce'),
            'ASBL': pd.to_numeric(data_rows.iloc[:, 4], errors='coerce')
        })
        prices = prices.dropna().reset_index(drop=True)
        prices = prices.set_index('Date')
        return prices
    
    def _calculate_monthly_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change()
    
    def _calculate_momentum(self, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return prices.pct_change(lookback)
    
    def _calculate_rolling_vol(self, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
        return returns.rolling(lookback).std()
    
    def _rank_momentum(self, mom_values: pd.Series) -> pd.Series:
        return mom_values.rank(ascending=False).astype(int)
    
    def _select_top_n(self, ranks: pd.Series, n: int) -> pd.Series:
        return (ranks <= n).astype(int)
    
    def _calculate_inv_vol_weights(self, vol: pd.Series, selected: pd.Series) -> pd.Series:
        inv_vol = (1 / vol) * selected
        total = inv_vol.sum()
        return inv_vol / total if total > 0 else pd.Series(0, index=vol.index)
    
    def run(self, prices: pd.DataFrame) -> dict:
        """Run the backtest."""
        config = self.config
        assets = prices.columns.tolist()
        
        monthly_returns = self._calculate_monthly_returns(prices)
        momentum = self._calculate_momentum(prices, config.momentum_lookback)
        volatility = self._calculate_rolling_vol(monthly_returns, config.vol_lookback)
        
        results = []
        nav = config.initial_nav
        max_nav = config.initial_nav
        prev_weights = None
        
        start_idx = config.momentum_lookback
        
        for i in range(start_idx, len(prices)):
            date = prices.index[i]
            mom_values = momentum.iloc[i]
            vol_values = volatility.iloc[i]
            
            if mom_values.isna().any() or vol_values.isna().any():
                continue
            
            ranks = self._rank_momentum(mom_values)
            selected = self._select_top_n(ranks, config.top_n)
            current_weights = self._calculate_inv_vol_weights(vol_values, selected)
            month_returns = monthly_returns.iloc[i]
            
            if prev_weights is not None:
                port_return = (prev_weights * month_returns).sum()
                nav = nav * (1 + port_return)
                max_nav = max(max_nav, nav)
                drawdown = (nav - max_nav) / max_nav
            else:
                port_return = 0.0
                drawdown = 0.0
            
            results.append({
                'Date': date,
                'NAV': nav,
                'Monthly_Return': port_return,
                'Max_NAV': max_nav,
                'Drawdown': drawdown,
                **{f'{a}_Price': prices.iloc[i][a] for a in assets},
                **{f'{a}_Return': month_returns[a] for a in assets},
                **{f'{a}_Momentum_6M': mom_values[a] for a in assets},
                **{f'{a}_Rank': ranks[a] for a in assets},
                **{f'{a}_Selected': selected[a] for a in assets},
                **{f'{a}_Vol_6M': vol_values[a] for a in assets},
                **{f'{a}_Weight': current_weights[a] for a in assets},
            })
            
            prev_weights = current_weights
        
        self.results = pd.DataFrame(results).set_index('Date')
        
        total_months = len(self.results)
        total_years = total_months / 12
        final_nav = self.results['NAV'].iloc[-1]
        
        cagr = (final_nav / config.initial_nav) ** (1 / total_years) - 1
        vol_annual = self.results['Monthly_Return'].std() * np.sqrt(12)
        sharpe = cagr / vol_annual if vol_annual > 0 else 0
        max_dd = self.results['Drawdown'].min()
        
        # Sortino ratio
        downside_returns = self.results['Monthly_Return'][self.results['Monthly_Return'] < 0]
        downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
        sortino = cagr / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        self.metrics = {
            'CAGR': cagr,
            'Volatility': vol_annual,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Calmar_Ratio': calmar,
            'Max_Drawdown': max_dd,
            'Final_NAV': final_nav,
            'Total_Months': total_months,
            'Total_Years': total_years,
            'Start_Date': self.results.index[0],
            'End_Date': self.results.index[-1]
        }
        
        return {'metrics': self.metrics, 'history': self.results}
    
    def get_current_allocation(self) -> dict:
        """Get the most recent portfolio allocation."""
        if self.results is None:
            return {}
        
        last_row = self.results.iloc[-1]
        weight_cols = [c for c in self.results.columns if '_Weight' in c]
        
        allocation = {}
        for col in weight_cols:
            asset = col.replace('_Weight', '')
            weight = last_row[col]
            if weight > 0.001:
                allocation[asset] = weight
        
        return allocation


def create_nav_chart(history: pd.DataFrame) -> go.Figure:
    """Create NAV equity curve chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history['NAV'],
        mode='lines',
        name='Portfolio NAV',
        line=dict(color='#00ff88', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,255,136,0.1)'
    ))
    
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='NAV',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def create_drawdown_chart(history: pd.DataFrame) -> go.Figure:
    """Create drawdown chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history.index,
        y=history['Drawdown'] * 100,
        mode='lines',
        name='Drawdown',
        line=dict(color='#ff4757', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,71,87,0.3)'
    ))
    
    fig.update_layout(
        title='Drawdown Profile',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


def create_allocation_chart(allocation: dict) -> go.Figure:
    """Create allocation pie chart."""
    labels = list(allocation.keys())
    values = [v * 100 for v in allocation.values()]
    
    colors = ['#00ff88', '#00d4ff', '#a855f7', '#ffa502']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title='Current Portfolio Allocation',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True
    )
    
    return fig


def create_monthly_returns_heatmap(history: pd.DataFrame) -> go.Figure:
    """Create monthly returns heatmap."""
    # Pivot to year x month
    monthly_df = history[['Monthly_Return']].copy()
    monthly_df['Year'] = monthly_df.index.year
    monthly_df['Month'] = monthly_df.index.month
    
    pivot = monthly_df.pivot_table(
        values='Monthly_Return', 
        index='Year', 
        columns='Month', 
        aggfunc='sum'
    ) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot.index,
        colorscale=[
            [0, '#ff4757'],
            [0.5, '#1a1a2e'],
            [1, '#00ff88']
        ],
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        textfont=dict(size=10),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monthly Returns Heatmap (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="header-title">📈 Multi Asset Momentum Risk Parity</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Adaptive Asset Allocation Strategy | Top 2 Momentum + Inverse Volatility Weighting</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Strategy Configuration")
        
        momentum_lookback = st.slider("Momentum Lookback (months)", 3, 12, 6)
        vol_lookback = st.slider("Volatility Lookback (months)", 3, 12, 6)
        top_n = st.slider("Top N Assets to Select", 1, 4, 2)
        
        st.markdown("---")
        st.header("📁 Data Source")
        
        # Check for data file in repo first
        repo_data_path = Path("Multi_Asset_Momentum_using_4_ASSETS.xlsx")
        data_dir_path = Path("data/Multi_Asset_Momentum_using_4_ASSETS.xlsx")
        
        data_source = None
        
        if repo_data_path.exists():
            st.success("✅ Data file found in repository")
            data_source = 'repo'
            data_path = repo_data_path
        elif data_dir_path.exists():
            st.success("✅ Data file found in data/ directory")
            data_source = 'data_dir'
            data_path = data_dir_path
        else:
            st.warning("⚠️ No data file in repo. Please upload.")
            uploaded_file = st.file_uploader(
                "Upload Excel Data File",
                type=['xlsx', 'xls'],
                help="Upload 'Multi_Asset_Momentum_using_4_ASSETS.xlsx'"
            )
            if uploaded_file is not None:
                data_source = 'upload'
                data_path = uploaded_file
        
        run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Main content
    if data_source is None:
        st.info("👆 Please upload the data file using the sidebar to begin.")
        
        st.markdown("""
        ### About This Strategy
        
        This dashboard implements the **Multi Asset Momentum Risk Parity** strategy from 
        [factorlab's Adaptive Asset Allocation case study](https://medium.com/@factorlab).
        
        **Strategy Logic:**
        1. **Universe**: 4 uncorrelated Indian assets (GOLDBEES, NIFTYBEES, MON100, ASBL)
        2. **Signal**: 6-month momentum (returns)
        3. **Selection**: Top 2 assets by momentum
        4. **Weighting**: Inverse volatility (Risk Parity)
        5. **Rebalancing**: Monthly
        
        **Expected Results (2014-2024):**
        - CAGR: ~16.6%
        - Sharpe Ratio: ~1.58
        - Max Drawdown: ~-8.9%
        """)
        return
    
    if run_backtest or 'results' not in st.session_state:
        with st.spinner("Running backtest..."):
            config = StrategyConfig(
                momentum_lookback=momentum_lookback,
                vol_lookback=vol_lookback,
                top_n=top_n,
                initial_nav=100.0
            )
            strategy = MultiAssetMomoRP(config)
            
            try:
                if data_source == 'upload':
                    prices = strategy.load_data_from_file(data_path)
                else:
                    prices = strategy.load_data_from_path(data_path)
                
                results = strategy.run(prices)
                st.session_state['results'] = results
                st.session_state['strategy'] = strategy
                st.session_state['config'] = config
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        strategy = st.session_state['strategy']
        metrics = results['metrics']
        history = results['history']
        
        # Metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("📈 CAGR", f"{metrics['CAGR']*100:.2f}%")
        with col2:
            st.metric("📊 Volatility", f"{metrics['Volatility']*100:.2f}%")
        with col3:
            st.metric("⚡ Sharpe", f"{metrics['Sharpe_Ratio']:.2f}")
        with col4:
            st.metric("🎯 Sortino", f"{metrics['Sortino_Ratio']:.2f}")
        with col5:
            st.metric("📉 Max DD", f"{metrics['Max_Drawdown']*100:.2f}%")
        with col6:
            st.metric("💰 Final NAV", f"{metrics['Final_NAV']:.2f}")
        
        st.markdown("---")
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Allocation", "🗓️ Monthly Returns", "📋 Data"])
        
        with tab1:
            st.plotly_chart(create_nav_chart(history), use_container_width=True)
            st.plotly_chart(create_drawdown_chart(history), use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                allocation = strategy.get_current_allocation()
                st.plotly_chart(create_allocation_chart(allocation), use_container_width=True)
            
            with col2:
                st.markdown("### Current Holdings")
                for asset, weight in sorted(allocation.items(), key=lambda x: -x[1]):
                    st.progress(weight, text=f"{asset}: {weight*100:.1f}%")
                
                st.markdown("### Strategy Summary")
                st.markdown(f"""
                - **Period**: {metrics['Start_Date'].strftime('%Y-%m-%d')} to {metrics['End_Date'].strftime('%Y-%m-%d')}
                - **Duration**: {metrics['Total_Years']:.1f} years ({metrics['Total_Months']} months)
                - **Calmar Ratio**: {metrics['Calmar_Ratio']:.2f}
                """)
        
        with tab3:
            st.plotly_chart(create_monthly_returns_heatmap(history), use_container_width=True)
            
            # Yearly returns
            yearly_returns = history['NAV'].resample('Y').last().pct_change().dropna() * 100
            yearly_df = pd.DataFrame({'Year': yearly_returns.index.year, 'Return (%)': yearly_returns.values})
            st.dataframe(yearly_df.set_index('Year').T, use_container_width=True)
        
        with tab4:
            st.markdown("### Backtest History")
            display_cols = ['NAV', 'Monthly_Return', 'Drawdown'] + [c for c in history.columns if '_Weight' in c]
            st.dataframe(history[display_cols].tail(24).round(4), use_container_width=True)
            
            # Download button
            csv = history.to_csv()
            st.download_button(
                label="📥 Download Full History (CSV)",
                data=csv,
                file_name="momentum_rp_backtest.csv",
                mime="text/csv"
            )


if __name__ == '__main__':
    main()
