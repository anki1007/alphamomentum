import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
import io
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. FETCH STOCK LIST AND SECTORS
# ---------------------------------------------------------
print("Fetching Nifty Total Market list...")
url = "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
response = requests.get(url, headers=headers)
nifty_df = pd.read_csv(io.StringIO(response.text))

# Append '.NS' for Yahoo Finance India stocks
nifty_df['Yahoo_Symbol'] = nifty_df['Symbol'].astype(str).str.strip() + ".NS"
sector_map = dict(zip(nifty_df['Yahoo_Symbol'], nifty_df['Industry']))

# IMPORTANT: To test the script quickly, let's limit to the first 50 stocks. 
# Change this to tickers = nifty_df['Yahoo_Symbol'].tolist() for the full 750 list.
tickers = nifty_df['Yahoo_Symbol'].tolist()[:50] 
print(f"Downloading data for {len(tickers)} stocks. This may take a moment...")

# ---------------------------------------------------------
# 2. DOWNLOAD HISTORICAL DATA (2020 - 2026)
# ---------------------------------------------------------
# We need data from Jan 2020 to calculate historical ranges, and up to 2026 for current LTP
data = yf.download(tickers, start="2020-01-01", end="2026-12-31", group_by='ticker', threads=True)

results = []

print("Processing indicators and historical support...")
for ticker in tickers:
    try:
        # Handle single vs multi-ticker download structure
        if len(tickers) == 1:
            df = data.copy()
        else:
            df = data[ticker].copy()
            
        if df.empty or len(df) < 260: # Need enough data for 252 RSI
            continue

        df.dropna(inplace=True)
        
        # Current Data
        ltp = df['Close'].iloc[-1]
        
        # ---------------------------------------------------------
        # 3. CALCULATE HISTORICAL SUPPORT (2020, 2021, 2022)
        # ---------------------------------------------------------
        # Extract data for 2020-2022 to find the absolute lowest monthly range
        historical_period = df.loc['2020-01-01':'2022-12-31']
        if historical_period.empty:
            continue
            
        # We will use the lowest 'Low' of that 3-year period as the ultimate base support
        hist_support = historical_period['Low'].min()
        
        # Check if 2026 LTP is "taking support" (within 10% of that historical bottom)
        is_near_support = "Yes" if (hist_support * 0.95) <= ltp <= (hist_support * 1.10) else "No"

        # ---------------------------------------------------------
        # 4. CALCULATE INDICATORS (EMAs & RSIs)
        # ---------------------------------------------------------
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['EMA_200'] = ta.ema(df['Close'], length=200)
        
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_21'] = ta.rsi(df['Close'], length=21)
        df['RSI_63'] = ta.rsi(df['Close'], length=63)
        df['RSI_126'] = ta.rsi(df['Close'], length=126)
        df['RSI_252'] = ta.rsi(df['Close'], length=252)

        # Get latest values
        latest = df.iloc[-1]
        
        # Trend Logic
        trend_aligned = "Yes" if (ltp > latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']) else "No"
        
        # LTP Above/Below 200 EMA (General breadth metric per stock)
        ltp_status = "Above 200 EMA" if ltp > latest['EMA_200'] else "Below 200 EMA"

        results.append({
            'Stock': ticker.replace('.NS', ''),
            'Sector': sector_map.get(ticker, 'Unknown'),
            'LTP': round(ltp, 2),
            'Status': ltp_status,
            'Near 20-22 Support': is_near_support,
            'EMA 20': round(latest['EMA_20'], 2),
            'EMA 50': round(latest['EMA_50'], 2),
            'EMA 200': round(latest['EMA_200'], 2),
            'LTP>20>50>200': trend_aligned,
            'RSI 14': round(latest['RSI_14'], 2),
            'RSI 21': round(latest['RSI_21'], 2),
            'RSI 63': round(latest['RSI_63'], 2),
            'RSI 126': round(latest['RSI_126'], 2),
            'RSI 252': round(latest['RSI_252'], 2)
        })

    except Exception as e:
        continue

# Create DataFrame
result_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 5. MARKET BREADTH CALCULATION
# ---------------------------------------------------------
total_stocks = len(result_df)
above_200 = len(result_df[result_df['Status'] == 'Above 200 EMA'])
breadth_percent = round((above_200 / total_stocks) * 100, 2) if total_stocks > 0 else 0

# ---------------------------------------------------------
# 6. NEON THEME HTML GENERATION
# ---------------------------------------------------------
print("Generating Neon Teal HTML Dashboard...")

def color_rsi(val):
    if pd.isna(val): return ''
    if isinstance(val, (int, float)):
        if val > 50:
            return 'color: #39FF14; text-shadow: 0 0 5px #39FF14; font-weight: bold;' # Neon Green
        else:
            return 'color: #FF0033; text-shadow: 0 0 5px #FF0033; font-weight: bold;' # Neon Red
    return ''

def color_trend(val):
    if val == "Yes": return 'color: #39FF14; text-shadow: 0 0 5px #39FF14;'
    if val == "No": return 'color: #FF0033; text-shadow: 0 0 5px #FF0033;'
    return ''

# Apply Pandas Styler
styler = result_df.style.applymap(color_rsi, subset=['RSI 14', 'RSI 21', 'RSI 63', 'RSI 126', 'RSI 252'])
styler = styler.applymap(color_trend, subset=['LTP>20>50>200', 'Near 20-22 Support'])

# Set table HTML with custom CSS
html_content = f"""
<html>
<head>
    <title>Neon Python Scanner 2026</title>
    <style>
        body {{
            background-color: #004d4d; /* Teal Blue Background */
            color: #00FFFF; /* Neon Cyan Text */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
        }}
        h1, h2, h3 {{
            text-align: center;
            color: #FF00FF; /* Neon Magenta */
            text-shadow: 0 0 10px #FF00FF;
        }}
        .breadth-box {{
            background-color: #002222;
            border: 2px solid #39FF14;
            padding: 15px;
            text-align: center;
            width: 50%;
            margin: 0 auto 30px auto;
            border-radius: 10px;
            box-shadow: 0 0 15px #39FF14;
        }}
        .breadth-text {{
            font-size: 24px;
            color: #39FF14;
            text-shadow: 0 0 8px #39FF14;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #002222; /* Darker teal for table background */
            box-shadow: 0 0 20px #00FFFF;
        }}
        th {{
            background-color: #001111;
            color: #FF00FF;
            border: 1px solid #00FFFF;
            padding: 12px;
            text-transform: uppercase;
        }}
        td {{
            border: 1px solid #00FFFF;
            padding: 10px;
            text-align: center;
        }}
        tr:hover {{
            background-color: #004444;
        }}
    </style>
</head>
<body>
    <h1>🚀 2026 NEON ALGO SCANNER</h1>
    
    <div class="breadth-box">
        <h3>Market Breadth</h3>
        <span class="breadth-text">{above_200} out of {total_stocks} stocks ({breadth_percent}%) are trading ABOVE their 200 EMA.</span>
    </div>

    {styler.to_html()}

</body>
</html>
"""

with open("neon_scanner_dashboard.html", "w", encoding="utf-8") as file:
    file.write(html_content)

print("✅ Done! Open 'neon_scanner_dashboard.html' in your web browser to view the results.")
