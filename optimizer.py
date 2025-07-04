"""
This module performs portfolio optimization using mean-variance optimization
with constraints on weights. It uses cvxpy to solve the quadratic program.

Objective:
    Maximize expected return for a given risk tolerance (risk-aversion parameter).
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Dict


def optimize_portfolio(
    ranked_stocks: pd.DataFrame,
    expected_returns: pd.Series = None,
    cov_matrix: pd.DataFrame = None,
    min_weight: float = 0.0,
    max_weight: float = 0.2,
    risk_aversion: float = 0.1
) -> Dict[str, float]:
    """
    Optimizes a portfolio based on selected stocks using mean-variance optimization.

    Args:
        ranked_stocks (pd.DataFrame): DataFrame with index as tickers and combined score column.
        expected_returns (pd.Series): Optional expected return per stock. If None, use combined score.
        cov_matrix (pd.DataFrame): Optional covariance matrix. If None, assume identity matrix.
        min_weight (float): Lower bound for each asset weight.
        max_weight (float): Upper bound for each asset weight.
        risk_aversion (float): Trade-off factor between risk and return.

    Returns:
        Dict[str, float]: Dictionary of {ticker: weight} for selected stocks.
    """
    tickers = ranked_stocks.index.tolist()
    n = len(tickers)

    # use combined score as proxy for expected return if none given
    mu = expected_returns if expected_returns is not None else ranked_stocks["combined_score"]

    # use identity matrix as fallback if no covariance matrix provided
    Sigma = cov_matrix if cov_matrix is not None else np.identity(n)

    # convert to numpy arrays
    mu = mu.loc[tickers].values
    Sigma = np.array(Sigma)

    # define optimization variables
    w = cp.Variable(n)

    # objective: maximize return - risk penalty
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))

    # constraints: fully invested and bounded weights
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if w.value is None:
        raise ValueError("Optimization failed. Check inputs.")

    weights = {ticker: float(weight) for ticker, weight in zip(tickers, w.value)}
    return weights
