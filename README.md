# 📈 AI-Powered Portfolio Optimizer  
### Machine Learning • LSTM Forecasting (PyTorch) • Random Forest • Mean-Variance Optimization • Walk-Forward Backtesting

This project is a **full-stack quantitative investing research platform** built using:

- **Streamlit** (interactive UI)
- **PyTorch LSTM models**
- **Random Forest regressors**
- **Mean-Variance optimization (CVXPY)**
- **Walk-forward backtesting engine**
- **Cumulative returns, Sharpe ratio, drawdowns, rolling risk analysis**

The app lets you:

- Forecast returns using **Sample Means**, **Random Forest**, or **LSTM**  
- Optimize portfolios using **Max Sharpe ratio**  
- Run **historical walk-forward backtests**  
- Visualize **equity curves**, **drawdowns**, **risk metrics**  
- Compare different forecasting methods over time  

---

## 🚀 Features

### 🔮 Forecasting Models
| Model | Description |
|-------|-------------|
| **Sample Mean** | Classic expected return (simple baseline) |
| **Random Forest (sklearn)** | Predicts next-day returns from lagged features |
| **LSTM (PyTorch)** | Deep learning sequence model capturing time patterns |

---

### 📊 Portfolio Optimization
- Max Sharpe Ratio optimization  
- Fully invested, long-only constraint  
- Uses **CVXPy** quadratic programming under the hood  

---

### 📅 Walk-Forward Backtesting
- True *out-of-sample* simulation  
- Supports:
  - **Daily**
  - **Weekly**
  - **Monthly** rebalancing  
- Uses rolling **lookback window** (60–756 days)
- Computes:
  - CAGR  
  - Annualized Volatility  
  - Sharpe Ratio  
  - Max Drawdown  
  - Equity curve growth  

---

## 🖥️ Screenshots


This modular structure allows you to expand the project:
- Add new ML models  
- Add more backtest logic  
- Add optimizers (risk-parity, min-variance, Black-Litterman)  

---

## 🧠 Project Architecture

ai-portfolio-optimizer/
│
├── app.py # Main Streamlit application
├── requirements.txt # Packages
├── README.md
│
├── src/
│ ├── models/
│ │ ├── lstm_model.py # PyTorch LSTM model
│ │ └── rf_model.py # RandomForest forecaster
│ │
│ ├── backtest/
│ │ └── backtest_engine.py # Walk-forward backtesting logic
│ │
│ ├── optimizer/
│ │ └── optimizer.py # CVXPY portfolio optimizer
│ │
│ ├── utils/
│ ├── metrics.py # Sharpe, volatility, drawdown
│ ├── plot_utils.py # Equity curve, drawdown plotting
│ └── data_loader.py # Price data + preprocessing

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/ai-portfolio-optimizer.git
cd ai-portfolio-optimizer
```


## 🧭 Future Enhancements (Roadmap)
🚀 Transformer-based forecaster (PyTorch)
⚖️ Transaction cost model
🆚 Benchmark comparison (SPY, AGG, 60/40)
🔁 Rolling window hyperparameter optimization
📉 Factor model risk attribution
🧮 Black-Litterman optimizer
📊 Performance heatmaps + scatter plots
🛠️ Dockerized deployment
