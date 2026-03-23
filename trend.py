import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
import io
import warnings

warnings.filterwarnings('ignore')

# Set the page configuration for a wider layout
st.set_page_config(page_title="Neon Algo Scanner", layout="wide")

st.title("🚀 2026 Neon Algo Scanner")
st.write("Scan the Nifty Total Market for momentum, trend alignment, and historical support.")

# Wrap the heavy processing in a function and a button
if st.button("Run Market Scan"):
    with st.spinner("Fetching market data and calculating indicators... This may take a couple of minutes."):
        
        # 1. FETCH STOCK LIST
        url = "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        nifty_df = pd.read_csv(io.StringIO(response.text))

        nifty_df['Yahoo_Symbol'] = nifty_df['Symbol'].astype(str).str.strip() + ".NS"
        sector_map = dict(zip(nifty_df['Yahoo_Symbol'], nifty_df['Industry']))

        # LIMITER: Currently set to 50 for testing. 
        # Change to `tickers = nifty_df['Yahoo_Symbol'].tolist()` for the full 750 list.
        tickers = nifty_df['Yahoo_Symbol'].tolist()[:50] 

        # 2. DOWNLOAD HISTORICAL DATA
        data = yf.download(tickers, start="2020-01-01", end="2026-12-31", group_by='ticker', threads=True)
        results = []

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()
                    
                if df.empty or len(df) < 260: 
                    continue

                df.dropna(inplace=True)
                ltp = df['Close'].iloc[-1]
                
                # 3. HISTORICAL SUPPORT (2020-2022)
                historical_period = df.loc['2020-01-01':'2022-12-31']
                if historical_period.empty:
                    continue
                    
                hist_support = historical_period['Low'].min()
                is_near_support = "Yes" if (hist_support * 0.95) <= ltp <= (hist_support * 1.10) else "No"

                # 4. INDICATORS
                df['EMA_20'] = ta.ema(df['Close'], length=20)
                df['EMA_50'] = ta.ema(df['Close'], length=50)
                df['EMA_200'] = ta.ema(df['Close'], length=200)
                
                df['RSI_14'] = ta.rsi(df['Close'], length=14)
                df['RSI_21'] = ta.rsi(df['Close'], length=21)
                df['RSI_63'] = ta.rsi(df['Close'], length=63)
                df['RSI_126'] = ta.rsi(df['Close'], length=126)
                df['RSI_252'] = ta.rsi(df['Close'], length=252)

                latest = df.iloc[-1]
                trend_aligned = "Yes" if (ltp > latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']) else "No"
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

        result_df = pd.DataFrame(results)

        # 5. BREADTH CALCULATION
        total_stocks = len(result_df)
        above_200 = len(result_df[result_df['Status'] == 'Above 200 EMA'])
        breadth_percent = round((above_200 / total_stocks) * 100, 2) if total_stocks > 0 else 0

        # 6. NEON HTML GENERATION
        def color_rsi(val):
            if pd.isna(val): return ''
            if isinstance(val, (int, float)):
                if val > 50: return 'color: #39FF14; text-shadow: 0 0 5px #39FF14; font-weight: bold;'
                else: return 'color: #FF0033; text-shadow: 0 0 5px #FF0033; font-weight: bold;'
            return ''

        def color_trend(val):
            if val == "Yes": return 'color: #39FF14; text-shadow: 0 0 5px #39FF14;'
            if val == "No": return 'color: #FF0033; text-shadow: 0 0 5px #FF0033;'
            return ''

        styler = result_df.style.applymap(color_rsi, subset=['RSI 14', 'RSI 21', 'RSI 63', 'RSI 126', 'RSI 252'])
        styler = styler.applymap(color_trend, subset=['LTP>20>50>200', 'Near 20-22 Support'])

        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    background-color: #004d4d;
                    color: #00FFFF;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
                h3 {{ text-align: center; color: #FF00FF; text-shadow: 0 0 10px #FF00FF; }}
                .breadth-box {{
                    background-color: #002222; border: 2px solid #39FF14; padding: 15px;
                    text-align: center; width: 60%; margin: 0 auto 20px auto;
                    border-radius: 10px; box-shadow: 0 0 15px #39FF14;
                }}
                .breadth-text {{ font-size: 20px; color: #39FF14; text-shadow: 0 0 8px #39FF14; }}
                table {{
                    width: 100%; border-collapse: collapse; background-color: #002222;
                    box-shadow: 0 0 20px #00FFFF;
                }}
                th {{ background-color: #001111; color: #FF00FF; border: 1px solid #00FFFF; padding: 10px; }}
                td {{ border: 1px solid #00FFFF; padding: 8px; text-align: center; }}
                tr:hover {{ background-color: #004444; }}
            </style>
        </head>
        <body>
            <div class="breadth-box">
                <h3>Market Breadth</h3>
                <span class="breadth-text">{above_200} out of {total_stocks} stocks ({breadth_percent}%) are trading ABOVE their 200 EMA.</span>
            </div>
            {styler.to_html()}
        </body>
        </html>
        """

        # Display the HTML in Streamlit
        components.html(html_content, height=800, scrolling=True)
        st.success("Scan Complete!")
