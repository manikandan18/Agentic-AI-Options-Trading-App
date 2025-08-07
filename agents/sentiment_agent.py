from transformers import pipeline
import yfinance as yf
from langchain_core.runnables import RunnableLambda

sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_news_headlines(ticker):
    company = yf.Ticker(ticker)
    news = getattr(company, "news", [])

    headlines = []
    for item in news:
        try:
            title = item["content"]["title"]
            if title:
                headlines.append(title)
        except (KeyError, TypeError):
            continue

    return headlines[:10]  # Limit to top 10

def sentiment_agent(state):
    tickers = state["tickers"]
    sentiment_result = {}

    for ticker in tickers:
        headlines = get_news_headlines(ticker)

        if not headlines:
            sentiment_result[ticker] = "positive"
        else:
            results = sentiment_model(headlines)
            negative_count = sum(1 for r in results if r['label'].lower() == 'negative')
            sentiment_result[ticker] = "negative" if negative_count > 2 else "positive"

    return {
        "sentiment": sentiment_result,
        "__output__": sentiment_result  # ✅ for LangGraph routing
    }

sentiment_agent = RunnableLambda(sentiment_agent)
