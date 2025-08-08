'''
This agent does the following.

1. Checks the state for positive sentiment mega cap tickers and 
2. It tries to find the option with highest open interest and expiry date >= 90 days for each ticker.
3. It updates the graph state with 'expiries' and 'top_strikes' for each ticker.

The output graph state is of format,

'expiries': {'ABBV': '2025-11-21', 'ABT': '2025-11-21'},

'top_strikes': {'ABBV': [{'strike': 250.0, 'openInterest': 806, 'contractSymbol': 'ABBV251121C00250000'}], 'ABT': [{'strike': 180.0, 'openInterest': 118, 'contractSymbol': 'ABT251121C00180000'}]
'''

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
                    
                # Get the top open interest option on that particular expiry date
                top_row = calls_df.sort_values(by="openInterest", ascending=False).iloc[0]

                # Get the top open interest option out of all expiry dates for that particular ticker
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
    
    # update the graph state with options data
    return {
        "expiries": expiries,
        "top_strikes": top_strikes
    }


# LangGraph-compatible RunnableLambda
options_agent = RunnableLambda(options_agent)
