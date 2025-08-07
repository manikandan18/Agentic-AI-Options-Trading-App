# Agentic-AI-Options-Trading-App

This is LangGrpah based Agentic AI Options Trading App. This Agentic AI application,

 1. Fetches the mega cap stocks with market cap more than $200B.
 2. Fetches the top headlines from yahoo finance for those mega cap stocks.
 3. Does the sentiment analysis with FinBert model to see if it is positive or negative.
 4. Fetches the call options for those sentiment positive mega cap stocks with expiry more than 90 days and highest open interest.
 5. Creates an LSTM model that uses these mega cap stocks past 3 years performance data from yahoo finance and combine that with technical indicators like RSI, MACD and Bollinger bands.
 6. The LSTM model is trained with 3 hidden layers and 1 dense output layer, mean squared error loss function and Adam optimizer for 100 epochs.
 7. The best model gets saved so that it doesn't get trained again and again for same set of data.
 8. The model takes the option expiry dates and predicts the stock price on that date in future.
 9. Finds the option strike prices that are closest +-15% from the predicted strike price and places option call order automatically.
 10. It also notifies the user of option prices that are not placed orders with for human judgement.

The Project source code folder Structure:-

Agentic-AI-Options-Trading-App/
├── graph_definition.py - LangGraph state and agent calls
├── agents/
│   ├── __init__.py
│   ├── ticker_agent.py - To fetch mega cap stocks.
│   ├── sentiment_agent_optimized.py - Do sentiment analysis using FinBert model
│   ├── options_agent.py - Fetch options from Yahoo finance
│   └── ml_predictor_agent_optimized.py - Core LSTM model that does stock forecasting
|___ai_trading_webapp.py - Streamlit App to fetch user input or place automated/manual trades

The Generated files folder Structure after executing ai_trading_webapp.py:-

The LSTM model would generate the below models and save it so that it doesn't get trained every time the call is made from webapp. This will improve the performance drastically.

Agentic-AI-Options-Trading-App/
├── models/
│   ├── AMZN_model.pt
|   ├── AMZN_model.pkl
|   ├── ...
├── loss_plots/
│   ├── loss_plots.png

To run the application from the terminal:-
git clone https://github.com/manikandan18/Agentic-AI-Options-Trading-App
cd Agentic-AI-Options-Trading-App
streamlit run ai_trading_webapp.py
