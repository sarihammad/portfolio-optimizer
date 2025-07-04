"""
Module for fetching historical price and fundamental data
for a list of stock tickers using yfinance.
"""

import yfinance as yf
import pandas as pd
from typing import List


def fetch_price_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetches adjusted close price data for the given tickers.

    Args:
        tickers (List[str]): List of stock tickers.
        start (str): Start date in 'YYYY-MM-DD'.
        end (str): End date in 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame with date as index and tickers as columns.
    """
    df = yf.download(tickers, start=start, end=end)["Close"]
    return df.dropna(how="all")


def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches basic fundamental metrics for each ticker.

    Currently retrieves:
        - Market cap
        - Price-to-earnings ratio
        - Price-to-book ratio
        - Dividend yield

    Args:
        tickers (List[str]): List of stock tickers.

    Returns:
        pd.DataFrame: DataFrame with fundamentals indexed by ticker.
    """
    records = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        records.append({
            "ticker": ticker,
            "marketCap": info.get("marketCap", None),
            "trailingPE": info.get("trailingPE", None),
            "priceToBook": info.get("priceToBook", None),
            "dividendYield": info.get("dividendYield", None)
        })
    return pd.DataFrame(records).set_index("ticker")
