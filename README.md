# AI-Powered Portfolio Optimizer

Interactive portfolio research app built with:
- Streamlit (UI)
- CVXPY (portfolio optimization)
- Random Forest + PyTorch LSTM (return forecasting)
- Walk-forward backtesting (equity curve + drawdowns)

## Features
- Forecast expected returns: Sample Mean / Random Forest / LSTM + ensembles
- Validate forecasts: time-series CV (RF) and holdout metrics (LSTM)
- Estimate risk: Sample covariance, Ledoit-Wolf shrinkage, or EWMA
- Optimize: mean-variance objective with risk aversion + max weight constraint
- Backtest: walk-forward simulation with rebalance frequency, lookback window, forecast horizon
- Optional transaction costs and turnover limit

## Project Structure
```text
ai-portfolio-optimizer/
|- main.py            # Streamlit application
|- requirements.txt   # Dependencies
|- Script.sh          # Quick start script
|- README.md
`- src/               # (Currently empty; reserved for modularization)
```

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run main.py
```

## Roadmap
- Transformer-based forecaster
- Benchmark comparison (e.g., SPY, AGG, 60/40)
- Rolling window hyperparameter optimization
- Black-Litterman optimizer
