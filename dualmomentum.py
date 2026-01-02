"""
================================================================================
MULTI ASSET MOMENTUM RISK PARITY STRATEGY
================================================================================
Replicates the "MULTI ASSET MOMO UC RP" strategy from factorlab's 
Adaptive Asset Allocation case study exactly.

Strategy Rules:
---------------
1. Universe: 4 uncorrelated assets (GOLDBEES, NIFTYBEES, MON100, ASBL)
   - Removed JUNIORBEES due to 0.87 correlation with NIFTYBEES
   
2. At end of each month:
   a) Calculate 6-month momentum (returns) for each asset
   b) Rank assets by momentum (1 = highest return)
   c) Select TOP 2 momentum assets
   d) Calculate 6-month rolling volatility
   e) Weight selected assets by INVERSE VOLATILITY (Risk Parity)
   
3. Hold these weights for the NEXT month
   - Return from T to T+1 = weights_T * returns_T+1

Results (Jul 2014 - May 2024):
------------------------------
- CAGR: 16.6%
- Max Drawdown: -8.9%
- Volatility: 10.5%
- Sharpe Ratio: 1.58

Reference: https://medium.com/@factorlab/adaptive-asset-allocation-case-study
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    """Configuration for the momentum risk parity strategy."""
    momentum_lookback: int = 6      # Months for momentum calculation
    vol_lookback: int = 6           # Months for volatility calculation
    top_n: int = 2                  # Number of top momentum assets to select
    initial_nav: float = 100.0      # Starting NAV


class MultiAssetMomoRP:
    """
    Multi Asset Momentum Risk Parity Backtester.
    
    Exactly replicates the Excel "MULTI ASSET MOMO UC RP" calculations.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.results = None
        self.metrics = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load asset prices from Excel file."""
        df = pd.read_excel(filepath, sheet_name='MULTI ASSET MOMO UC RP', header=None)
        
        # Skip header rows (row 0 = section headers, row 1 = column names)
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
        """Calculate monthly returns."""
        return prices.pct_change()
    
    def _calculate_momentum(self, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate N-month momentum (returns)."""
        return prices.pct_change(lookback)
    
    def _calculate_rolling_vol(self, returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate rolling N-month volatility."""
        return returns.rolling(lookback).std()
    
    def _rank_momentum(self, mom_values: pd.Series) -> pd.Series:
        """Rank assets by momentum (1 = highest)."""
        return mom_values.rank(ascending=False).astype(int)
    
    def _select_top_n(self, ranks: pd.Series, n: int) -> pd.Series:
        """Select top N assets (1 = selected, 0 = not)."""
        return (ranks <= n).astype(int)
    
    def _calculate_inv_vol_weights(self, vol: pd.Series, selected: pd.Series) -> pd.Series:
        """Calculate inverse volatility weights for selected assets."""
        inv_vol = (1 / vol) * selected
        total = inv_vol.sum()
        return inv_vol / total if total > 0 else pd.Series(0, index=vol.index)
    
    def run(self, prices: pd.DataFrame) -> dict:
        """
        Run the backtest.
        
        Parameters:
        -----------
        prices : DataFrame
            Monthly prices with DatetimeIndex and asset columns
            
        Returns:
        --------
        dict with 'metrics' and 'history' DataFrames
        """
        config = self.config
        assets = prices.columns.tolist()
        
        # Calculate signals
        monthly_returns = self._calculate_monthly_returns(prices)
        momentum = self._calculate_momentum(prices, config.momentum_lookback)
        volatility = self._calculate_rolling_vol(monthly_returns, config.vol_lookback)
        
        # Initialize
        results = []
        nav = config.initial_nav
        max_nav = config.initial_nav
        prev_weights = None
        
        # Start after lookback period
        start_idx = config.momentum_lookback
        
        for i in range(start_idx, len(prices)):
            date = prices.index[i]
            
            # Get current signals
            mom_values = momentum.iloc[i]
            vol_values = volatility.iloc[i]
            
            if mom_values.isna().any() or vol_values.isna().any():
                continue
            
            # Rank and select
            ranks = self._rank_momentum(mom_values)
            selected = self._select_top_n(ranks, config.top_n)
            current_weights = self._calculate_inv_vol_weights(vol_values, selected)
            
            # Get returns
            month_returns = monthly_returns.iloc[i]
            
            # Calculate portfolio return using PREVIOUS weights (lagged execution)
            if prev_weights is not None:
                port_return = (prev_weights * month_returns).sum()
                nav = nav * (1 + port_return)
                max_nav = max(max_nav, nav)
                drawdown = (nav - max_nav) / max_nav
            else:
                port_return = 0.0
                drawdown = 0.0
            
            # Store results
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
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results).set_index('Date')
        
        # Calculate metrics
        total_months = len(self.results)
        total_years = total_months / 12
        final_nav = self.results['NAV'].iloc[-1]
        
        cagr = (final_nav / config.initial_nav) ** (1 / total_years) - 1
        vol_annual = self.results['Monthly_Return'].std() * np.sqrt(12)
        sharpe = cagr / vol_annual if vol_annual > 0 else 0
        max_dd = self.results['Drawdown'].min()
        
        self.metrics = {
            'CAGR': cagr,
            'Volatility': vol_annual,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_dd,
            'Final_NAV': final_nav,
            'Total_Months': total_months,
            'Total_Years': total_years,
            'Start_Date': self.results.index[0],
            'End_Date': self.results.index[-1]
        }
        
        return {'metrics': self.metrics, 'history': self.results}
    
    def print_results(self):
        """Print formatted backtest results."""
        if self.metrics is None:
            print("No results available. Run backtest first.")
            return
            
        m = self.metrics
        print("=" * 65)
        print("MULTI ASSET MOMENTUM RISK PARITY - BACKTEST RESULTS")
        print("=" * 65)
        print(f"\nPeriod: {m['Start_Date'].strftime('%Y-%m-%d')} to {m['End_Date'].strftime('%Y-%m-%d')}")
        print(f"Duration: {m['Total_Years']:.2f} years ({m['Total_Months']} months)")
        print()
        print("-" * 45)
        print("PERFORMANCE METRICS")
        print("-" * 45)
        print(f"  CAGR:             {m['CAGR']*100:>10.2f}%")
        print(f"  Volatility:       {m['Volatility']*100:>10.2f}%")
        print(f"  Sharpe Ratio:     {m['Sharpe_Ratio']:>10.2f}")
        print(f"  Max Drawdown:     {m['Max_Drawdown']*100:>10.2f}%")
        print(f"  Final NAV:        {m['Final_NAV']:>10.2f}")
        print("-" * 45)
        
        # Print expected vs actual
        expected = {'CAGR': 0.166, 'Volatility': 0.1047, 'Sharpe': 1.58, 'MaxDD': -0.0891}
        print("\n" + "-" * 45)
        print("COMPARISON WITH EXCEL TARGET")
        print("-" * 45)
        print(f"  {'Metric':<15} {'Python':<12} {'Excel':<12} {'Match'}")
        print(f"  {'CAGR':<15} {m['CAGR']*100:>10.2f}%  {expected['CAGR']*100:>10.2f}%  {'✓' if abs(m['CAGR']-expected['CAGR'])<0.005 else '✗'}")
        print(f"  {'Volatility':<15} {m['Volatility']*100:>10.2f}%  {expected['Volatility']*100:>10.2f}%  {'✓' if abs(m['Volatility']-expected['Volatility'])<0.005 else '✗'}")
        print(f"  {'Sharpe':<15} {m['Sharpe_Ratio']:>10.2f}   {expected['Sharpe']:>10.2f}   {'✓' if abs(m['Sharpe_Ratio']-expected['Sharpe'])<0.05 else '✗'}")
        print(f"  {'Max Drawdown':<15} {m['Max_Drawdown']*100:>10.2f}%  {expected['MaxDD']*100:>10.2f}%  {'✓' if abs(m['Max_Drawdown']-expected['MaxDD'])<0.005 else '✗'}")
        print("-" * 45)
    
    def export_results(self, output_path: str):
        """Export results to Excel."""
        if self.results is None:
            print("No results to export. Run backtest first.")
            return
            
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Full history
            self.results.reset_index().to_excel(writer, sheet_name='History', index=False)
            
            # Summary metrics
            metrics_df = pd.DataFrame([{
                'Metric': k,
                'Value': f"{v*100:.2f}%" if any(x in k for x in ['CAGR', 'Volatility', 'Drawdown']) 
                         else f"{v:.2f}" if isinstance(v, float) else str(v)
            } for k, v in self.metrics.items()])
            metrics_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Monthly allocation history
            weight_cols = [c for c in self.results.columns if '_Weight' in c]
            weights_df = self.results[weight_cols].copy()
            weights_df.columns = [c.replace('_Weight', '') for c in weights_df.columns]
            weights_df.reset_index().to_excel(writer, sheet_name='Allocations', index=False)
            
        print(f"Results exported to: {output_path}")
    
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
            if weight > 0.001:  # Only include non-zero weights
                allocation[asset] = weight
        
        return allocation


def main():
    """Run the backtest."""
    # Path to data file
    data_path = Path('/mnt/user-data/uploads/Multi_Asset_Momentum_using_4_ASSETS.xlsx')
    
    # Initialize strategy
    config = StrategyConfig(
        momentum_lookback=6,
        vol_lookback=6,
        top_n=2,
        initial_nav=100.0
    )
    strategy = MultiAssetMomoRP(config)
    
    # Load data
    print("Loading data from Excel...")
    prices = strategy.load_data(data_path)
    print(f"Loaded {len(prices)} months of data for {len(prices.columns)} assets")
    print(f"Assets: {prices.columns.tolist()}")
    print(f"Date Range: {prices.index[0]} to {prices.index[-1]}")
    
    # Run backtest
    print("\nRunning backtest...")
    results = strategy.run(prices)
    
    # Print results
    strategy.print_results()
    
    # Export results
    output_path = '/mnt/user-data/outputs/multi_asset_momo_rp_backtest.xlsx'
    strategy.export_results(output_path)
    
    # Show current allocation
    print("\n" + "=" * 65)
    print("CURRENT ALLOCATION (as of last period)")
    print("=" * 65)
    allocation = strategy.get_current_allocation()
    for asset, weight in sorted(allocation.items(), key=lambda x: -x[1]):
        print(f"  {asset:<15} {weight*100:>8.2f}%")
    
    # Save CSV for quick inspection
    results['history'].to_csv('/mnt/user-data/outputs/multi_asset_momo_rp_history.csv')
    print(f"\nHistory also saved to: /mnt/user-data/outputs/multi_asset_momo_rp_history.csv")
    
    return strategy


if __name__ == '__main__':
    strategy = main()
