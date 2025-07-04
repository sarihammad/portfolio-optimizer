"""
This module provides functions to evaluate and visualize
portfolio performance, including:
- Equity curve
- Drawdowns
- Sharpe ratio
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: pd.DataFrame) -> None:
    """
    Plots the portfolio's equity curve and drawdown chart.

    Args:
        results (pd.DataFrame): DataFrame with 'PortfolioValue' and 'DailyReturn' columns.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # equity curve
    results["PortfolioValue"].plot(ax=ax1, color="navy", linewidth=2)
    ax1.set_title("Portfolio Equity Curve")
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True)

    # drawdowns
    peak = results["PortfolioValue"].cummax()
    drawdown = (results["PortfolioValue"] - peak) / peak
    drawdown.plot(ax=ax2, color="crimson", linewidth=1.5)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # metrics
    sharpe = calculate_sharpe(results["DailyReturn"])
    max_dd = drawdown.min()

    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates annualized Sharpe ratio.

    Args:
        returns (pd.Series): Daily return series.
        risk_free_rate (float): Risk-free rate, default is 0.

    Returns:
        float: Sharpe ratio (annualized).
    """
    excess_returns = returns - risk_free_rate / 252
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    sharpe_ratio = np.sqrt(252) * mean_return / std_return
    return sharpe_ratio if std_return != 0 else 0.0
