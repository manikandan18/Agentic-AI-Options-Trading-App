from transformers import pipeline
import yfinance as yf
import datetime

sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def get_news_headlines(ticker):
    company = yf.Ticker(ticker)
    news = company.news if hasattr(company, 'news') else []
    headlines = [item['title'] for item in news][:10]
    return headlines

def sentiment_agent(state):
    ticker = state["ticker"]
    headlines = get_news_headlines(ticker)

    if not headlines:
        return {"sentiment": "positive"}

    results = sentiment_model(headlines)
    negative_count = sum(1 for r in results if r['label'].lower() == 'negative')

    return {"sentiment": "negative" if negative_count > 2 else "positive"}
