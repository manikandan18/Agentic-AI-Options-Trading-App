# options_agent.py

import datetime
import yfinance as yf
from dateutil.parser import parse
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda


def options_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    tickers: List[str] = state.get("tickers", [])
    sentiment_map: Dict[str, str] = state.get("sentiment", {})

    today = datetime.date.today()
    expiries = {}
    top_strikes = {}

    for ticker in tickers:
        if sentiment_map.get(ticker) != "positive":
            continue  # Only consider tickers with positive sentiment

        stock = yf.Ticker(ticker)
        expiry_list = stock.options

        if not expiry_list:
            expiries[ticker] = None
            top_strikes[ticker] = []
            continue

        best_call = None
        best_expiry = None
        max_oi = -1

        for expiry in expiry_list:
            try:
                expiry_date = parse(expiry).date()
                if (expiry_date - today).days < 90:
                    continue  # Only consider expiries >= 90 days

                options_chain = stock.option_chain(expiry)
                calls_df = options_chain.calls

                if calls_df.empty:
                    continue

                top_row = calls_df.sort_values(by="openInterest", ascending=False).iloc[0]

                if top_row["openInterest"] > max_oi:
                    max_oi = top_row["openInterest"]
                    best_call = {
                        "strike": float(top_row["strike"]),
                        "openInterest": int(top_row["openInterest"]),
                        "contractSymbol": top_row["contractSymbol"]
                    }
                    best_expiry = expiry

            except Exception as e:
                print(f"[{ticker}] Skipping expiry {expiry} due to error: {e}")
                continue

        expiries[ticker] = best_expiry
        top_strikes[ticker] = [best_call] if best_call else []

    return {
        "expiries": expiries,
        "top_strikes": top_strikes
    }


# LangGraph-compatible RunnableLambda
options_agent = RunnableLambda(options_agent)
