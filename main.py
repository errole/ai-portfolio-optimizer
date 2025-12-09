# app.py
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import streamlit as st
import matplotlib.pyplot as plt

# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# PyTorch LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Portfolio Optimizer + ML + LSTM + Backtest",
    layout="wide",
)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if "Close" in data.columns:
        return data["Close"].dropna()
    return data.dropna()


def compute_returns(prices):
    rets = np.log(prices / prices.shift(1)).dropna()
    mu = rets.mean() * 252
    sigma = rets.cov() * 252
    return rets, mu, sigma


# ============================================================
# PORTFOLIO OPTIMIZATION
# ============================================================
def optimize_max_sharpe(mu, sigma, rf=0.03, long_only=True):
    n = len(mu)
    w = cp.Variable(n)

    mu_vec = mu.values
    Sigma = sigma.values

    port_ret = mu_vec @ w
    port_vol = cp.sqrt(cp.quad_form(w, Sigma))

    objective = cp.Maximize((port_ret - rf) / port_vol)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True)

    if w.value is None:
        raise ValueError("Optimization failed")

    return np.array(w.value)


# ============================================================
# RANDOM FOREST FORECASTER
# ============================================================
def ml_forecast_rf(returns, rolling_window=20):
    """
    Train RF separately per asset to predict next-day return from lagged returns.
    """
    ml_mu = {}

    for asset in returns.columns:
        r = returns[asset].dropna()
        if len(r) <= rolling_window + 10:
            continue

        df = pd.DataFrame({"target": r})

        for lag in range(1, rolling_window + 1):
            df[f"lag_{lag}"] = r.shift(lag)

        df = df.dropna()

        X = df.drop("target", axis=1).values
        y = df["target"].values

        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Predict the next value using last available lags
        next_pred = model.predict(X[-1:].reshape(1, -1))[0]

        ml_mu[asset] = {
            "annual_mu": next_pred * 252,
            "r2": r2_score(y_test, preds),
            "mse": mean_squared_error(y_test, preds),
            "pred": next_pred,
        }

    return ml_mu


# ============================================================
# PYTORCH LSTM IMPLEMENTATION
# ============================================================
class ReturnDataset(Dataset):
    def __init__(self, series, window):
        self.series = series.values.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.series) - self.window

    def __getitem__(self, idx):
        X = self.series[idx: idx + self.window]
        y = self.series[idx + self.window]
        return X.reshape(-1, 1), y


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out


def lstm_forecast(returns, window=30, epochs=10, lr=0.001):
    """
    Train a separate LSTM per asset to predict next-day return.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_results = {}

    for asset in returns.columns:
        series = returns[asset].dropna()
        if len(series) <= window + 10:
            continue

        dataset = ReturnDataset(series, window)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = LSTMForecaster().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Training loop
        model.train()
        for _ in range(epochs):
            for X, y in loader:
                X = X.to(device)
                y = y.to(device)

                optim.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y.unsqueeze(1))
                loss.backward()
                optim.step()

        # Predict next day's return
        model.eval()
        x_last = torch.tensor(series.values[-window:], dtype=torch.float32)
        x_last = x_last.reshape(1, window, 1).to(device)
        with torch.no_grad():
            next_pred = model(x_last).item()

        lstm_results[asset] = {
            "pred": next_pred,
            "annual_mu": next_pred * 252,
            "mse": float(loss.item()),
        }

    return lstm_results


# ============================================================
# BACKTESTING ENGINE
# ============================================================
def performance_stats(portfolio_values, rf_rate=0.0):
    """
    Compute CAGR, volatility, Sharpe, max drawdown from portfolio value series.
    """
    v = portfolio_values.dropna()
    if len(v) < 2:
        return None

    returns = v.pct_change().dropna()
    if returns.empty:
        return None

    total_days = (v.index[-1] - v.index[0]).days
    cagr = (v.iloc[-1] / v.iloc[0]) ** (252 / max(total_days, 1)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - rf_rate) / vol if vol > 0 else np.nan

    running_max = v.cummax()
    drawdown = v / running_max - 1
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }


def get_rebalance_dates(returns, freq="M"):
    """
    Returns DatetimeIndex of rebalance dates at given frequency.
    freq: 'D', 'W', 'M'
    """
    if freq == "D":
        return returns.index
    elif freq == "W":
        return returns.resample("W-FRI").last().index
    elif freq == "M":
        return returns.resample("M").last().index
    else:
        return returns.index


def backtest_strategy(returns, forecast_method, rf_rate=0.03,
                      rebalance_freq="M", lookback_days=252):
    """
    Walk-forward backtest:
    - At each rebalance date, use past 'lookback_days' of returns
      to estimate mu and sigma.
    - Depending on forecast_method, use Sample Mean / RF / LSTM
      for expected returns.
    - Optimize weights (max Sharpe), hold until next rebalance.
    - Returns portfolio value series.
    """
    rets = returns.copy()
    rebal_dates = get_rebalance_dates(rets, rebalance_freq)
    if len(rebal_dates) < 2:
        return None, None

    # Start after we have enough history
    start_date = rets.index[0] + pd.Timedelta(days=lookback_days)
    rebal_dates = [d for d in rebal_dates if d >= start_date]
    if len(rebal_dates) < 2:
        return None, None

    portfolio_values = []
    portfolio_index = []

    current_value = 1.0
    prev_rebal_date = rebal_dates[0]

    for i, d in enumerate(rebal_dates[:-1]):
        # History up to rebalance date
        hist_window = rets.loc[:d].tail(lookback_days)
        if len(hist_window) < 30:
            continue

        # Covariance always from history
        _, mu_sample, sigma_hist = compute_returns(
            (hist_window + 1).cumprod()
        )  # recompute from hist_window

        # Expected returns choice
        if forecast_method == "Sample Mean":
            mu = mu_sample

        elif forecast_method == "Random Forest":
            rf_results = ml_forecast_rf(hist_window)
            if not rf_results:
                mu = mu_sample
            else:
                mu = pd.Series(
                    {a: rf_results[a]["annual_mu"] for a in rf_results},
                    index=mu_sample.index,
                ).fillna(mu_sample)

        elif forecast_method == "LSTM (PyTorch)":
            lstm_results = lstm_forecast(hist_window, epochs=5)
            if not lstm_results:
                mu = mu_sample
            else:
                mu = pd.Series(
                    {a: lstm_results[a]["annual_mu"] for a in lstm_results},
                    index=mu_sample.index,
                ).fillna(mu_sample)
        else:
            mu = mu_sample

        # Optimize weights at rebalance date
        try:
            w = optimize_max_sharpe(mu, sigma_hist, rf=rf_rate, long_only=True)
        except Exception:
            # Fallback: equal weights if optimization fails
            w = np.ones(len(mu_sample)) / len(mu_sample)

        # Apply weights from this rebalance date until next
        next_d = rebal_dates[i + 1]
        period_rets = rets.loc[(rets.index > d) & (rets.index <= next_d)]
        if period_rets.empty:
            continue

        # Portfolio daily returns in this period
        for t, row in period_rets.iterrows():
            r_p = np.dot(w, row.values)
            current_value *= (1 + r_p)
            portfolio_values.append(current_value)
            portfolio_index.append(t)

        prev_rebal_date = d

    if not portfolio_values:
        return None, None

    equity_curve = pd.Series(portfolio_values, index=portfolio_index, name="Portfolio")
    stats = performance_stats(equity_curve, rf_rate=rf_rate)
    return equity_curve, stats


# ============================================================
# SIDEBAR UI
# ============================================================
st.sidebar.title("Settings")

tickers = st.sidebar.text_input("Tickers", "TLT,LQD,HYG,SPY")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", dt.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

rf_rate = st.sidebar.number_input("Risk-Free Rate (annual)", value=0.03, format="%.3f")

forecast_method = st.sidebar.selectbox(
    "Forecast Method",
    ["Sample Mean", "Random Forest", "LSTM (PyTorch)"],
)

rebalance_freq_label = st.sidebar.selectbox(
    "Backtest Rebalance Frequency",
    ["Monthly", "Weekly", "Daily"],
    index=0,
)

rebalance_freq_map = {
    "Monthly": "M",
    "Weekly": "W",
    "Daily": "D",
}
rebalance_freq = rebalance_freq_map[rebalance_freq_label]

lookback_days = st.sidebar.slider(
    "Lookback Window (days)",
    min_value=60,
    max_value=756,
    value=252,
    step=30,
)

run_backtest = st.sidebar.button("🚀 Run Backtest")


# ============================================================
# LOAD DATA
# ============================================================
if not tickers:
    st.error("Please enter at least one valid ticker.")
    st.stop()

prices = load_price_data(tickers, start_date, end_date)
if prices.empty:
    st.error("No price data. Check tickers/date range.")
    st.stop()

returns, mu_sample, sigma = compute_returns(prices)

st.title("📈 Portfolio Optimizer with ML, LSTM & Backtesting")
st.subheader("Price History")
st.line_chart(prices)


# ============================================================
# FORECASTING CHOICE (CURRENT OPTIMAL PORTFOLIO)
# ============================================================
st.subheader("Expected Returns (Current Estimation)")

if forecast_method == "Sample Mean":
    mu = mu_sample
    st.write("Using **Sample Mean** annualized returns:")
    st.dataframe(mu.to_frame("μ_sample").style.format("{:.2%}"))

elif forecast_method == "Random Forest":
    st.write("🔮 Using **Random Forest** ML Forecasting")
    rf_results = ml_forecast_rf(returns)
    if not rf_results:
        st.warning("Not enough data for RF; falling back to sample means.")
        mu = mu_sample
    else:
        mu = pd.Series({a: rf_results[a]["annual_mu"] for a in rf_results})
        perf_df = pd.DataFrame(rf_results).T
        st.dataframe(perf_df.style.format({"annual_mu": "{:.2%}", "pred": "{:.4f}"}))

elif forecast_method == "LSTM (PyTorch)":
    st.write("🧠 Using **LSTM (PyTorch)** Forecasting")
    lstm_results = lstm_forecast(returns, epochs=10)
    if not lstm_results:
        st.warning("Not enough data for LSTM; falling back to sample means.")
        mu = mu_sample
    else:
        mu = pd.Series({a: lstm_results[a]["annual_mu"] for a in lstm_results})
        perf_df = pd.DataFrame(lstm_results).T
        st.dataframe(perf_df.style.format({"annual_mu": "{:.2%}", "pred": "{:.4f}"}))

else:
    mu = mu_sample


# Ensure mu aligned with sigma
mu = mu.reindex(mu_sample.index).fillna(mu_sample)


# ============================================================
# CURRENT OPTIMAL PORTFOLIO (MAX SHARPE, STATIC)
# ============================================================
st.subheader("Optimal Portfolio (Max Sharpe) - Static Based on Full Sample")

try:
    weights = optimize_max_sharpe(mu, sigma, rf=rf_rate, long_only=True)
    opt_ret = float(weights @ mu.values)
    opt_vol = float(np.sqrt(weights @ sigma.values @ weights))
    opt_sharpe = (opt_ret - rf_rate) / opt_vol if opt_vol > 0 else np.nan

    df_w = pd.DataFrame({"Weight": weights}, index=mu.index)
    st.dataframe(df_w.style.format("{:.2%}"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Return (annual)", f"{opt_ret:.2%}")
    c2.metric("Volatility (annual)", f"{opt_vol:.2%}")
    c3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
except Exception as e:
    st.error(f"Optimization failed: {e}")


# ============================================================
# BACKTESTING SECTION
# ============================================================
st.markdown("---")
st.subheader("📊 Backtest: Walk-Forward Cumulative Returns")

if run_backtest:
    with st.spinner("Running backtest..."):
        equity_curve, stats = backtest_strategy(
            returns,
            forecast_method=forecast_method,
            rf_rate=rf_rate,
            rebalance_freq=rebalance_freq,
            lookback_days=lookback_days,
        )

    if equity_curve is None:
        st.error("Backtest could not be run (not enough data or dates).")
    else:
        # Equity curve
        st.write("**Equity Curve (Growth of $1)**")
        st.line_chart(equity_curve)

        # Performance stats
        if stats:
            st.write("**Backtest Performance Summary**")
            stats_df = pd.DataFrame(stats, index=["Backtest"])
            st.dataframe(stats_df.style.format("{:.2%}"))

            # Drawdown plot
            running_max = equity_curve.cummax()
            drawdown = equity_curve / running_max - 1
            fig, ax = plt.subplots()
            ax.plot(drawdown.index, drawdown.values)
            ax.set_title("Drawdown")
            ax.set_ylabel("Drawdown")
            st.pyplot(fig)
        else:
            st.warning("Not enough data to compute performance statistics.")
else:
    st.info("Set your options in the sidebar and click **🚀 Run Backtest** to simulate historical performance.")
# ============================================================
# END OF FILE