from typing import TypedDict, Optional

class TradingState(TypedDict, total=False):
    ticker: str
    sentiment: Optional[str]
    options: Optional[dict]
    prediction: Optional[str]
