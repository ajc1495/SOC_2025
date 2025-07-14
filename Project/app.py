import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# ========== DUMMY Sentiment function ==========
def get_sentiment(text):
    # 🔥 Replace with real FinBERT later if needed
    return 0.0  # Always return a float

# ========== Function to fetch stock data ==========
@st.cache_data
def fetch_data(ticker, start="2015-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = df['Close'].rolling(window=14).apply(
        lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / -x.diff().clip(upper=0).sum())))
    )
    df['Price_Change'] = df['Close'].pct_change()
    df['Headline'] = "Market is stable"  # Dummy headlines
    df.dropna(inplace=True)
    return df

# ========== Function to run agent ==========

def run_agent(model, data, initial_balance=10000):
    balance = initial_balance
    holdings = 0
    net_worth = initial_balance

    net_worths = []
    actions = []

    for step in range(len(data)):
        try:
            # Ensure all values are scalars
            close = float(data.iloc[step]["Close"])
            sma = float(data.iloc[step]["SMA"])
            rsi = float(data.iloc[step]["RSI"])
            sentiment = float(get_sentiment(data.iloc[step]["Headline"]))
        except Exception as e:
            print(f"Error at step {step}: {e}")
            continue  # Skip this step if error

        observation = np.array([
            close,
            sma,
            rsi,
            balance,
            holdings,
            sentiment
        ], dtype=np.float32)

        action, _ = model.predict(observation, deterministic=True)
        current_price = close

        # Execute action
        if action == 1 and balance >= current_price:  # Buy
            holdings += 1
            balance -= current_price
        elif action == 2 and holdings > 0:  # Sell
            holdings -= 1
            balance += current_price

        net_worth = balance + holdings * current_price
        net_worths.append(net_worth)
        actions.append(action)

    return pd.DataFrame({
        "Date": data.index[:len(net_worths)],  # Align index
        "Net_Worth": net_worths,
        "Action": actions
    })

# ========== Streamlit UI ==========
st.title("📈 DRL Stock Trading Agent with PPO")
st.write("This app loads your trained PPO agent and simulates trading on selected stock data.")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
if st.button("Run Simulation"):
    st.write("⏳ Fetching data and loading model...")
    data = fetch_data(ticker)
    model = PPO.load("ppo_trading_agent.zip")  # 👈 Ensure ZIP file in same folder
    result = run_agent(model, data)

    st.success("✅ Simulation complete!")
    st.line_chart(result.set_index("Date")["Net_Worth"])
    st.dataframe(result)
