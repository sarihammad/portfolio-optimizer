# Factor-Based Portfolio Optimizer

An end-to-end project to build, optimize, and evaluate equity portfolios using fundamental and price-based factors.

## Overview

This project ranks stocks based on three factors:

- **Momentum** (6-month return)
- **Value** (earnings yield)
- **Volatility** (inverse std. dev)

It selects the top N stocks, optimizes their weights using mean-variance optimization (cvxpy), and backtests their performance over time.

## How to Run

```bash
pip install -r requirements.txt
python main.py
