"""
Configuration module for the Factor-Based Portfolio Optimizer.
Defines tickers, date ranges, model parameters, and constraints.
"""

# portfolio universe
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "V", "JNJ", "NVDA"
]

# backtest dates
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# factor parameters
MOMENTUM_LOOKBACK_DAYS = 126  # 6 months
TOP_N_STOCKS = 5

# optimization parameters
RISK_AVERSION = 0.1           # trade-off between risk and return
MAX_WEIGHT = 0.3              # per asset cap
MIN_WEIGHT = 0.0              # no short selling

# backtesting parameters
REBALANCE_FREQUENCY = "M"     # monthly
