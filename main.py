"""
Main entry point for the Factor-Based Portfolio Optimizer.

Workflow:
1. Load price and fundamental data
2. Compute factor scores
3. Rank and select top stocks
4. Optimize portfolio using mean-variance optimization
5. Run backtest with monthly rebalancing
6. Visualize results (equity curve, drawdown, Sharpe ratio)
"""

from data_fetcher import fetch_price_data, fetch_fundamentals
from factor_model import compute_factors, rank_stocks
from optimizer import optimize_portfolio
from backtest import run_backtest
from evaluation import plot_results
import config


def main():
    print("Fetching price and fundamental data...")
    price_df = fetch_price_data(config.TICKERS, config.START_DATE, config.END_DATE)
    fundamentals_df = fetch_fundamentals(config.TICKERS)

    print("Calculating factor scores...")
    factor_scores = compute_factors(
        price_df,
        fundamentals_df,
        momentum_window=config.MOMENTUM_LOOKBACK_DAYS
    )

    print("Ranking stocks and selecting top performers...")
    ranked_stocks = rank_stocks(factor_scores, top_n=config.TOP_N_STOCKS)

    print("Optimizing portfolio allocation...")
    weights = optimize_portfolio(
        ranked_stocks,
        min_weight=config.MIN_WEIGHT,
        max_weight=config.MAX_WEIGHT,
        risk_aversion=config.RISK_AVERSION
    )

    print("Running backtest...")
    results = run_backtest(
        price_df,
        weights,
        rebalance_freq=config.REBALANCE_FREQUENCY
    )

    print("Visualizing performance...")
    plot_results(results)


if __name__ == "__main__":
    main()
