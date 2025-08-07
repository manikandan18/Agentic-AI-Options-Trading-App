# graph_definition.py
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import List, Dict, Optional, Literal, TypedDict, Any

from agents.ticker_agent import ticker_agent
from agents.sentiment_agent import sentiment_agent
from agents.options_agent import options_agent
from agents.ml_predictor_agent import ml_predictor_agent

# Define shared state schema
class GraphState(TypedDict, total=False):
    tickers: List[str]
    sentiment: Dict[str, Literal["positive", "negative"]]
    expiries: Dict[str, Optional[str]]
    top_strikes: Dict[str, List[Dict]]  # ticker -> list of strikes
    predicted_prices: Dict[str, float]  # ticker -> predicted price
    best_strike_prices: Dict[str, Dict[str, Any]]  # ticker -> best matching option dict


def create_options_graph():
    builder = StateGraph(GraphState)

    # Register the nodes (all must be Runnables or callables)
    builder.add_node("TickerAgent", ticker_agent)
    builder.add_node("SentimentAgent", sentiment_agent)
    builder.add_node("OptionsAgent", options_agent)
    builder.add_node("MLPredictorAgent", ml_predictor_agent)
 
    # Set flow
    builder.set_entry_point("TickerAgent")
    builder.add_edge("TickerAgent", "SentimentAgent")
    builder.add_edge("SentimentAgent", "OptionsAgent")
    builder.add_edge("OptionsAgent", "MLPredictorAgent")
    builder.add_edge("MLPredictorAgent", END)
    return builder.compile()
