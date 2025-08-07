from langchain_core.runnables import RunnableLambda

def ticker_agent_fn(state):
    tickers = state.get("tickers") or state.get("ticker")
    if not tickers:
        return {
            "tickers": [
                'AAPL', 'ABBV', 'ABT', 'AMD', 'AMZN', 'ASML', 'AVGO', 'AXP',
                'AZN', 'BABA', 'BAC', 'BRK-A', 'BRK-B', 'BX', 'CAT', 'COST',
                'CRM', 'CSCO', 'CVX', 'DIS', 'GE', 'GOOG', 'GOOGL', 'GS', 'HD',
                'HSBC', 'IBM', 'INTU', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'MA',
                'MCD', 'META', 'MS', 'MSFT', 'NFLX', 'NVDA', 'NVS', 'ORCL',
                'PG', 'PLTR', 'PM', 'RTX', 'SAP', 'SHEL', 'TM', 'TMUS', 'TSLA',
                'TSM', 'UNH', 'V', 'WFC', 'WMT', 'XOM'
            ]
        }
    if isinstance(tickers, str):
        return {"tickers": [tickers]}  # Normalize to list
    return {"tickers": tickers}  # Already a list

ticker_agent = RunnableLambda(ticker_agent_fn)