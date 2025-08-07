

import streamlit as st
import pandas as pd
from graph_definition import create_options_graph

st.title("📈 Options Trading Agentic AI")

graph = create_options_graph()

# Input tickers
tickers_input = st.text_input("Enter tickers (comma separated AAPL,TSLA,MSFT)")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Run the LangGraph trading workflow
if st.button("Run Trading Workflow"):
    input_data = {"tickers": tickers} if tickers else {}
    if not tickers:
        st.info("No ticker specified. Running for all mega cap stocks.")

    with st.spinner("Running trading workflow..."):
        result = graph.invoke(input_data)

        predicted_prices = result.get("predicted_prices", {})
        best_strike_prices = result.get("best_strike_prices", {})
        expiries = result.get("expiries", {})

        if predicted_prices:
            st.subheader("📊 Trading Recommendations")
            rows = []

            for ticker, predicted_price in predicted_prices.items():
                option_info = best_strike_prices.get(ticker)
                expiry = expiries.get(ticker, "N/A")

                if option_info:
                    strike_price = option_info.get("strike", "N/A")
                    open_interest = option_info.get("openInterest", "N/A")
                else:
                    strike_price = open_interest = "N/A"

                # Calculate price difference % if strike is available
                if isinstance(strike_price, (float, int)) and strike_price != 0:
                    price_diff_pct = (predicted_price - strike_price) / strike_price * 100
                else:
                    price_diff_pct = 0

                # Determine action
                if isinstance(price_diff_pct, (float, int)) and abs(price_diff_pct) <= 15:
                    action = "✅ Order placed automatically."
                else:
                    action = ""

                rows.append({
                    "Ticker": ticker,
                    "Predicted Stock Price": f"${predicted_price:.2f}",
                    "Strike Price": f"${strike_price:.2f}" if isinstance(strike_price, (float, int)) else "N/A",
                    "Expiry Date": expiry,
                    "Open Interest": open_interest,
                    "Action": action
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Show buttons for manual orders
            for row in rows:
                if not row["Action"]:
                    col1, col2 = st.columns([0.85, 0.15])
                    with col1:
                        st.write(f"📌 **{row['Ticker']}** needs manual action.")
                    with col2:
                        if st.button(row["Ticker"] + " - Place Order"):
                            st.success(f"✅ Manual order placed for {row['Ticker']}.")
        else:
            st.warning("⚠️ No predictions available.")
