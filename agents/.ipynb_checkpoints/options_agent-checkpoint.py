import yfinance as yf
import datetime

def options_agent(state):
    ticker = state["ticker"]
    stock = yf.Ticker(ticker)

    # Target expiry: current date + 3 months
    today = datetime.date.today()
    expiry_list = stock.options
    target_expiry = (today + datetime.timedelta(days=90)).strftime("%Y-%m-%d")

    expiry = next((e for e in expiry_list if e >= target_expiry), expiry_list[-1])

    options_chain = stock.option_chain(expiry)
    calls_df = options_chain.calls
    top_calls = calls_df.sort_values(by="openInterest", ascending=False).head(10)

    return {
        "ticker": ticker,
        "expiry": expiry,
        "top_strikes": top_calls[["strike", "openInterest"]].to_dict(orient="records")
    }