def ticker_agent(state):
    ticker = state.get("ticker")
    if not ticker:
        raise ValueError("Ticker is required.")
    return {"ticker": ticker}