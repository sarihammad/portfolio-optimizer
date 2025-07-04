"""
This module provides a simple backtesting engine for evaluating
the performance of a static or periodically rebalanced portfolio.

Assumptions:
- Long-only portfolio
- No leverage or shorting
- Monthly rebalancing
- No transaction costs (can be added later)
"""

import pandas as pd
import numpy as np
from typing import Dict


def run_backtest(
    price_df: pd.DataFrame,
    weights: Dict[str, float],
    rebalance_freq: str = "ME"
) -> pd.DataFrame:
    """
    Runs a backtest of a weighted stock portfolio with periodic rebalancing.

    Args:
        price_df (pd.DataFrame): Daily adjusted close prices with tickers as columns.
        weights (Dict[str, float]): Dictionary of {ticker: weight}.
        rebalance_freq (str): Rebalance frequency ('ME' = monthly, 'QE' = quarterly).

    Returns:
        pd.DataFrame: Backtest results with equity curve and daily returns.
    """
    tickers = list(weights.keys())
    price_df = price_df[tickers].dropna()

    # calculate normalized prices (starting at 1)
    norm_prices = price_df / price_df.iloc[0]

    # calculate periodic returns
    periodic_prices = norm_prices.resample(rebalance_freq).first()
    weight_df = pd.DataFrame(index=periodic_prices.index, columns=tickers)
    for date in periodic_prices.index:
        weight_df.loc[date] = weights

    # forward-fill weights and align with price index
    weight_df = weight_df.ffill().reindex(price_df.index).ffill()

    # calculate daily returns
    daily_returns = price_df.pct_change().fillna(0)

    # portfolio returns
    portfolio_returns = (daily_returns * weight_df).sum(axis=1)

    # calculate cumulative equity curve
    equity_curve = (1 + portfolio_returns).cumprod()

    # combine into result DataFrame
    result = pd.DataFrame({
        "PortfolioValue": equity_curve,
        "DailyReturn": portfolio_returns
    })

    return result
