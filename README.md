# AI-Powered Portfolio Optimizer

Interactive portfolio research app built with:
- Streamlit (UI)
- CVXPY (portfolio optimization)
- Random Forest + PyTorch LSTM (return forecasting)
- Walk-forward backtesting (equity curve + drawdowns)

## Prerequisites
- Python 3.10+ recommended
- Internet access (the app pulls market data via `yfinance`)

## Features
- Forecast expected returns: Sample Mean / Random Forest / LSTM + ensembles
- Validate forecasts: time-series CV (RF) and holdout metrics (LSTM) (see following link for more information on LSTM https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
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
```

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run main.py
```

Then open the Streamlit URL printed in your terminal (usually `http://localhost:8501`).

## Alternative (Script)
```bash
chmod +x Script.sh
./Script.sh
```

## Installation (without venv)
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run main.py
```

## Troubleshooting
- If `cvxpy` fails to install/build, upgrade tooling first: `python -m pip install -U pip setuptools wheel`
- If `torch` install fails, use the official install selector for your OS/Python: https://pytorch.org/get-started/locally/

## Roadmap
- Transformer-based forecaster
- Benchmark comparison (e.g., SPY, AGG, 60/40)
- Rolling window hyperparameter optimization
- Black-Litterman optimizer
