"""
This module computes factor scores for a list of stocks
and ranks them based on combined factor signals.

Factors implemented:
- Momentum (6-month return)
- Volatility (inverse of std deviation)
- Value (Earnings Yield: 1 / PE ratio)

All factors are normalized and equally weighted.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_factors(
    price_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    momentum_window: int = 126  # 6 months (21 days * 6)
) -> pd.DataFrame:
    """
    Computes momentum, volatility, and value factors for stocks.

    Args:
        price_df (pd.DataFrame): Daily price data with tickers as columns.
        fundamentals_df (pd.DataFrame): Fundamental data indexed by ticker.
        momentum_window (int): Lookback window for momentum.

    Returns:
        pd.DataFrame: DataFrame of normalized factor scores indexed by ticker.
    """
    latest_prices = price_df.iloc[-1]
    past_prices = price_df.shift(momentum_window).iloc[-1]

    momentum = (latest_prices - past_prices) / past_prices
    volatility = price_df.pct_change().rolling(momentum_window).std().iloc[-1]
    inverse_volatility = 1 / volatility

    pe_ratio = fundamentals_df["trailingPE"]
    earnings_yield = 1 / pe_ratio.replace(0, np.nan)

    factor_df = pd.DataFrame({
        "momentum": momentum,
        "inverse_volatility": inverse_volatility,
        "earnings_yield": earnings_yield
    })

    factor_df = factor_df.dropna()

    # normalize each column
    normalized_factors = (factor_df - factor_df.mean()) / factor_df.std()

    return normalized_factors


def rank_stocks(factor_scores: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Ranks stocks by combined factor scores and selects the top N.

    Args:
        factor_scores (pd.DataFrame): Normalized factor score DataFrame.
        top_n (int): Number of top-ranked stocks to return.

    Returns:
        pd.DataFrame: DataFrame of top N tickers and their combined score.
    """
    factor_scores["combined_score"] = factor_scores.mean(axis=1)
    ranked = factor_scores.sort_values("combined_score", ascending=False)
    return ranked.head(top_n)[["combined_score"]]
