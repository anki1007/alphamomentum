# Create a single Python file with the updated scanner including EMA10/20/40 logic

code = r'''
# NSE STOCK SCANNER PRO (Revamped with EMA10/20/40 Structure)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="NSE Stock Scanner Pro", layout="wide")

NIFTY_URL = "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"

@st.cache_data(ttl=86400)
def fetch_stock_list():
    r = requests.get(NIFTY_URL, timeout=20)
    return pd.read_csv(StringIO(r.text))

@st.cache_data(ttl=3600)
def fetch_daily_data(symbol):
    df = yf.Ticker(f"{symbol}.NS").history(start="2020-01-01", interval="1d")
    return df.dropna()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def scan_stock(symbol):
    df = fetch_daily_data(symbol)
    if df.empty or len(df) < 200:
        return None

    close = df["Close"]
    ltp = close.iloc[-1]

    e10 = ema(close, 10).iloc[-1]
    e20 = ema(close, 20).iloc[-1]
    e40 = ema(close, 40).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]

    return {
        "Symbol": symbol,
        "LTP": round(ltp,2),
        "EMA10": round(e10,2),
        "EMA20": round(e20,2),
        "EMA40": round(e40,2),
        "EMA50": round(e50,2),
        "EMA200": round(e200,2),
        "EMA10 > EMA20": "YES" if e10 > e20 else "NO",
        "EMA20 > EMA40": "YES" if e20 > e40 else "NO",
        "Trend": "STRONG" if (ltp > e20 > e50 > e200) else "WEAK",
    }

st.title("⚡ NSE Scanner (EMA Structure Upgrade)")

stocks = fetch_stock_list()
symbols = stocks.iloc[:,0].head(50)

filter_ema10 = st.checkbox("EMA10 > EMA20")
filter_ema20 = st.checkbox("EMA20 > EMA40")

results = []

if st.button("Run Scanner"):
    for s in symbols:
        r = scan_stock(s)
        if r:
            results.append(r)

    df = pd.DataFrame(results)

    if filter_ema10:
        df = df[df["EMA10 > EMA20"] == "YES"]

    if filter_ema20:
        df = df[df["EMA20 > EMA40"] == "YES"]

    st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False), "scanner.csv")
'''

file_path = "/mnt/data/trend.py"
with open(file_path, "w") as f:
    f.write(code)

file_path
