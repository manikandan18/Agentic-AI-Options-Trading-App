# File: agents/ml_predictor_agent.py

import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import os

# Technical indicators
def add_technical_indicators(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['Close'])
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band

# LSTM model for price prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][0]  # predict Close price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_lstm_on_stock(ticker, seq_length=30, epochs=20):
    df = yf.download(ticker, period="2y", interval="1d")
    df = add_technical_indicators(df)
    features = df[['Close', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(features)

    X, y = create_sequences(data_scaled, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = LSTMModel(input_size=features.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model, scaler, seq_length, features.shape[1]


def predict_future_price(model, scaler, seq_length, ticker, input_size):
    df = yf.download(ticker, period="2y", interval="1d")
    df = add_technical_indicators(df)
    features = df[['Close', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']].values
    data_scaled = scaler.transform(features)
    last_seq = torch.tensor(data_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq)
    # Only return the predicted Close value
    close_index = 0
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, close_index] = pred.item()
    pred_price = scaler.inverse_transform(dummy)[0][close_index]
    return pred_price


def ml_predictor_agent(state):
    ticker = state["ticker"]
    top_strikes = state["top_strikes"]

    model, scaler, seq_length, input_size = train_lstm_on_stock(ticker)
    predicted_price = predict_future_price(model, scaler, seq_length, ticker, input_size)

    # Find closest match from top open interest strikes
    strikes = [x['strike'] for x in top_strikes]
    best_strike = min(strikes, key=lambda x: abs(x - predicted_price))

    return {
        "ticker": ticker,
        "predicted_price": predicted_price,
        "best_strike_price": best_strike,
        "top_strikes": top_strikes
    }
