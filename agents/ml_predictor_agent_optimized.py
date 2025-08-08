'''
This agent does the following.

1. Fetch the past 3 years performance of all mega cap stocks from yahoo finance. It uses Close price, RSI, MACD and Bollinger bands.
2. Splits up the data set into 80-20 for training and validation. Creates LSTM (Long Short Term Memory) model that uses 3 hidden layers, 1 dense output layer with adam optimizer. The learning rate at 0.001 with 100 epochs.
3. Trains and validates the model.
4. Plots the loss curve for all tickers.
5. Saves the model so that it does not have to train again and again for the same tickers. This improves the Agentic AI performance drastically. It just have to do forecasting in further calls.
6. Uses the model to forecast the stock prices of the top option tickers we obtained from options_agent on that expiry date.
7. It identifies the best strike price option that is closest to predicted stock price from the model.
8. It updates this best_strike_prices in the graph state and returns it.

The optimizations done are,

1. Checking for 100 epochs,
2. Dropout of 0.3 was used for generalization and making model not to overfit.
3. Used 3 years past performance data instead of 2 years.
4. Learning rate modified to 0.001.
5. Number of epochs increased to 100

The state output is of format,

'best_strike_prices': {'AAPL': {'strike': 70.0, 'openInterest': 2526, 'contractSymbol': 'AAPL261218C00070000'}, 'ABBV': {'strike': 250.0, 'openInterest': 806, 'contractSymbol': 'ABBV251121C00250000'}, 'ABT': {'strike': 180.0, 'openInterest': 118, 'contractSymbol': 'ABT251121C00180000'}, 'AVGO': {'strike': 80.0, 'openInterest': 5369, 'contractSymbol': 'AVGO251219C00080000'}}
'''

import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from langchain_core.runnables import RunnableLambda
import numpy as np
import datetime
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
LOSS_PLOTS_DIR = "loss_plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOSS_PLOTS_DIR, exist_ok=True)

# ----- Technical Indicators -----
def add_technical_indicators(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['Close'])
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + num_std * std, sma - num_std * std

# ----- LSTM Model Definition -----
class OptimizedLSTMModel(nn.Module):
    # Add hidden size, hidden layers, dropout optimizations
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=1, dropout=0.3):
        super(OptimizedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ----- Predict Future Price -----
def predict_price_n_days(model, scaler, seq, input_size, days):
    for _ in range(days):
        input_seq = torch.tensor(seq[-len(seq):], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            next_pred = model(input_seq).item()
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0][0] = next_pred
        next_close = scaler.inverse_transform(dummy)[0][0]
        seq = np.vstack([seq, np.hstack([next_pred] * input_size)])
    return round(next_close, 2)

# ----- Plot Losses -----
def plot_loss_per_ticker(loss_dict):
    plt.figure(figsize=(10, 6))
    for ticker, losses in loss_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=ticker)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Per Ticker")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOSS_PLOTS_DIR, "loss_plot.png"))
    plt.close()

# ----- Model Summary -----
def print_model_summary():
    print("\nModel Summary (Shared Across All Tickers)")
    print("=" * 50)
    print(f"{'Model':<20}: LSTM")
    print(f"{'Hidden Layers':<20}: 3 × LSTM(128 units) + Dropout(0.3)")
    print(f"{'Output Layer':<20}: Dense(1)")
    print(f"{'Optimizer':<20}: Adam")
    print(f"{'Loss Function':<20}: Mean Squared Error")
    print(f"{'Epochs':<20}: 100")
    print("=" * 50 + "\n")

# ----- Core Agent Logic -----
def _ml_predictor_agent(state):
    tickers = state["tickers"]
    expiries = state.get("expiries", {})
    top_strikes_all = state.get("top_strikes", {})

    print_model_summary()
    loss_history = {}
    results = {}

    for ticker in tickers:
        try:
            expiry_str = expiries.get(ticker)
            if not expiry_str:
                results[ticker] = {"error": "No expiry"}
                continue

            expiry = datetime.datetime.strptime(expiry_str, "%Y-%m-%d").date()
            today = datetime.date.today()
            days_ahead = (expiry - today).days
            if days_ahead <= 0:
                results[ticker] = {"error": "Invalid expiry"}
                continue

            model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pt")
            scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

            df = yf.download(ticker, period="3y", interval="1d", progress=False)
            df = add_technical_indicators(df)
            features = df[['Close', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']].values

            seq_length = 30
            input_size = features.shape[1]

            # Check for model presence per ticker. If present, avoid re-training for performance optimization
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                # Scale the data using MinMaxScaler
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(features)
                X, y = [], []
                for i in range(len(data_scaled) - seq_length):
                    X.append(data_scaled[i:i+seq_length])
                    y.append(data_scaled[i+seq_length][0])

                # Train-validation split (80-20)
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

                X_train = torch.tensor(X_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                X_val = torch.tensor(X_val, dtype=torch.float32)
                y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

                model = OptimizedLSTMModel(input_size=input_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss_fn = nn.MSELoss()

                ticker_losses = []
                
                for epoch in range(100):
                    model.train()
                    optimizer.zero_grad()
                    output = model(X_train)
                    loss = loss_fn(output, y_train)
                    loss.backward()
                    optimizer.step()

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_output = model(X_val)
                        val_loss = loss_fn(val_output, y_val).item()
                    ticker_losses.append(val_loss)

                # Save the model for future use and avoid re-training for each call from Web app
                torch.save(model.state_dict(), model_path)
                joblib.dump(scaler, scaler_path)
                loss_history[ticker] = ticker_losses
            else:
                # Store the scaled data appropriate for each ticker
                scaler = joblib.load(scaler_path)
                data_scaled = scaler.transform(features)
                model = OptimizedLSTMModel(input_size=input_size)
                model.load_state_dict(torch.load(model_path))
                model.eval()

            recent_seq = data_scaled[-seq_length:]

            # Predict the price of expiry date in future
            predicted_price = predict_price_n_days(model, scaler, recent_seq.copy(), input_size, days_ahead)

            strikes = top_strikes_all.get(ticker, [])
            if strikes:
                # Get the option with closest strike price to LSTM predicted price
                best_strike = min(strikes, key=lambda s: abs(s["strike"] - predicted_price))
            else:
                best_strike = None

            results[ticker] = {
                "predicted_price_on_expiry": predicted_price,
                "best_matching_option": best_strike
            }

        except Exception as e:
            results[ticker] = {"error": str(e)}

    if loss_history:
        plot_loss_per_ticker(loss_history)

    predicted_prices = {
        ticker: data["predicted_price_on_expiry"]
        for ticker, data in results.items()
        if "predicted_price_on_expiry" in data
    }

    best_strike_prices = {
        ticker: data["best_matching_option"]
        for ticker, data in results.items()
        if "best_matching_option" in data
    }

    return {
        "predicted_prices": predicted_prices,
        "best_strike_prices": best_strike_prices
    }

ml_predictor_agent = RunnableLambda(_ml_predictor_agent)
