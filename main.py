from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset

TRADING_DAYS_PER_YEAR = 252
FREQ_TO_DEFAULT_HORIZON_DAYS = {"D": 1, "W": 5, "M": 21}

st.set_page_config(page_title="AI Portfolio Optimizer", layout="wide")


def infer_horizon_days(rebalance_freq: str) -> int:
    return int(FREQ_TO_DEFAULT_HORIZON_DAYS.get(rebalance_freq, 1))


def annualize_horizon_return(horizon_return: float, horizon_days: int) -> float:
    horizon_days = max(int(horizon_days), 1)
    horizon_return = float(np.clip(horizon_return, -0.99, 10.0))
    return (1.0 + horizon_return) ** (TRADING_DAYS_PER_YEAR / horizon_days) - 1.0


def forward_cumulative_return(series: pd.Series, horizon_days: int) -> pd.Series:
    """
    Align forward horizon return to time t (uses returns t+1..t+horizon_days).
    """
    horizon_days = max(int(horizon_days), 1)
    if horizon_days == 1:
        return series.shift(-1)

    future = (1.0 + series).shift(-1)
    fwd = future.rolling(horizon_days).apply(np.prod, raw=True) - 1.0
    return fwd.shift(-(horizon_days - 1))


@st.cache_data(show_spinner=False)
def load_price_data(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data["Close"] if "Close" in data.columns else data

    if isinstance(close, pd.Series):
        name = tickers[0] if tickers else (close.name or "PRICE")
        close = close.rename(str(name).upper()).to_frame()
    else:
        close = close.copy()
        close.columns = [str(c).upper() for c in close.columns]

    return close.dropna(how="all")


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def estimate_covariance_daily(
    returns: pd.DataFrame,
    method: str,
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    returns = returns.dropna(how="all")

    if method == "Ledoit-Wolf":
        clean = returns.dropna(how="any")
        if len(clean) < 5:
            return returns.cov()
        lw = LedoitWolf().fit(clean.to_numpy())
        return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

    if method == "EWMA":
        x = returns.fillna(0.0).to_numpy(dtype=float)
        n = x.shape[0]
        if n < 2:
            return returns.cov()

        lam = float(np.clip(ewma_lambda, 0.5, 0.999))
        weights = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1)
        weights = weights / weights.sum()

        mean = weights @ x
        xc = x - mean
        cov = (xc.T * weights) @ xc
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

    return returns.cov()


def estimate_annualized_moments(
    returns: pd.DataFrame,
    cov_method: str,
    ewma_lambda: float,
) -> tuple[pd.Series, pd.DataFrame]:
    mu_annual = returns.mean() * TRADING_DAYS_PER_YEAR
    cov_daily = estimate_covariance_daily(returns, cov_method, ewma_lambda=ewma_lambda)
    sigma_annual = cov_daily * TRADING_DAYS_PER_YEAR
    return mu_annual, sigma_annual


def optimize_portfolio_mean_variance(
    mu_annual: pd.Series,
    sigma_annual: pd.DataFrame,
    rf_annual: float,
    risk_aversion: float,
    long_only: bool,
    max_weight: float | None,
    prev_weights: np.ndarray | None,
    turnover_limit: float | None,
) -> np.ndarray:
    mu_annual = pd.Series(mu_annual).dropna()
    if mu_annual.empty:
        raise ValueError("No assets to optimize.")

    sigma_annual = pd.DataFrame(sigma_annual).loc[mu_annual.index, mu_annual.index]
    n_assets = len(mu_annual)

    sigma_np = sigma_annual.to_numpy(dtype=float)
    sigma_np = 0.5 * (sigma_np + sigma_np.T)
    sigma_np = sigma_np + np.eye(n_assets) * 1e-10

    w = cp.Variable(n_assets)
    excess = mu_annual.to_numpy(dtype=float) - float(rf_annual)
    risk_aversion = float(max(risk_aversion, 1e-8))

    objective = cp.Maximize(excess @ w - risk_aversion * cp.quad_form(w, sigma_np))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)

    if max_weight is not None:
        constraints.append(w <= float(max_weight))

    if turnover_limit is not None and prev_weights is not None:
        constraints.append(cp.norm1(w - prev_weights) <= float(turnover_limit))

    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True)

    if w.value is None:
        raise ValueError("Optimization failed.")

    weights = np.array(w.value, dtype=float).reshape(-1)
    if long_only:
        weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Optimization produced invalid weights.")
    return weights / total


def _rf_build_dataset(
    series: pd.Series,
    rolling_window: int,
    horizon_days: int,
) -> pd.DataFrame:
    target = forward_cumulative_return(series, horizon_days=horizon_days)
    df = pd.DataFrame({"target": target})
    for lag in range(1, rolling_window + 1):
        df[f"lag_{lag}"] = series.shift(lag - 1)
    return df.dropna()


def _beta_from_mse(mse: float, target_var: float) -> float:
    mse = float(mse)
    target_var = float(target_var)
    if not np.isfinite(mse) or mse <= 0 or not np.isfinite(target_var) or target_var <= 0:
        return 0.0
    return float(np.clip(1.0 / (1.0 + mse / (target_var + 1e-12)), 0.0, 1.0))


@st.cache_data(show_spinner=False)
def rf_forecast(
    returns: pd.DataFrame,
    rolling_window: int,
    horizon_days: int,
    cv_splits: int,
    random_state: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    for asset in returns.columns:
        series = returns[asset].dropna()
        if len(series) < max(rolling_window + horizon_days + 20, 80):
            continue

        df = _rf_build_dataset(series, rolling_window=rolling_window, horizon_days=horizon_days)
        if len(df) < 50:
            continue

        X = df.drop(columns=["target"]).to_numpy()
        y = df["target"].to_numpy()

        cv_splits_eff = int(np.clip(cv_splits, 2, 10))
        cv_splits_eff = min(cv_splits_eff, max(2, len(df) // 25))
        tscv = TimeSeriesSplit(n_splits=cv_splits_eff)

        fold_mse: list[float] = []
        fold_r2: list[float] = []
        for train_idx, test_idx in tscv.split(X):
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            fold_mse.append(float(mean_squared_error(y[test_idx], preds)))
            fold_r2.append(float(r2_score(y[test_idx], preds)))

        cv_mse = float(np.mean(fold_mse)) if fold_mse else float("nan")
        cv_r2 = float(np.mean(fold_r2)) if fold_r2 else float("nan")
        target_var = float(np.var(y)) if len(y) else float("nan")

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X, y)

        last = series.tail(rolling_window)
        x_next = np.array([last.iloc[-lag] for lag in range(1, rolling_window + 1)], dtype=float)
        pred_horizon = float(model.predict(x_next.reshape(1, -1))[0])

        results[asset] = {
            "pred_horizon": pred_horizon,
            "annual_mu": annualize_horizon_return(pred_horizon, horizon_days=horizon_days),
            "cv_mse": cv_mse,
            "cv_r2": cv_r2,
            "beta": _beta_from_mse(cv_mse, target_var),
        }

    return results


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


@dataclass(frozen=True)
class LSTMParams:
    window: int
    hidden: int
    layers: int
    dropout: float
    lr: float
    max_epochs: int
    patience: int
    grad_clip: float
    clip_returns: float
    seed: int


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _lstm_build_xy(
    series: pd.Series,
    horizon_days: int,
    window: int,
    clip_returns: float,
) -> tuple[np.ndarray, np.ndarray]:
    series = series.astype(float).copy()
    if clip_returns is not None and np.isfinite(clip_returns) and clip_returns > 0:
        series = series.clip(-clip_returns, clip_returns)

    target = forward_cumulative_return(series, horizon_days=horizon_days)
    values = series.to_numpy(dtype=float)
    target_values = target.to_numpy(dtype=float)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for t in range(window - 1, len(values)):
        if t >= len(target_values):
            break
        if not np.isfinite(target_values[t]):
            continue
        if t - window + 1 < 0:
            continue
        X_list.append(values[t - window + 1 : t + 1].reshape(window, 1))
        y_list.append(float(target_values[t]))

    if not X_list:
        return np.empty((0, window, 1)), np.empty((0,))
    return np.stack(X_list, axis=0), np.array(y_list, dtype=float)


@st.cache_data(show_spinner=False)
def lstm_forecast(
    returns: pd.DataFrame,
    horizon_days: int,
    params: LSTMParams,
) -> dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict] = {}

    _set_torch_seed(params.seed)

    for asset in returns.columns:
        series = returns[asset].dropna()
        if len(series) < max(params.window + horizon_days + 50, 120):
            continue

        X, y = _lstm_build_xy(
            series,
            horizon_days=horizon_days,
            window=params.window,
            clip_returns=params.clip_returns,
        )
        if len(y) < 80:
            continue

        split = int(len(y) * 0.8)
        if split < 50 or len(y) - split < 10:
            continue

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        x_mean = float(X_train.mean())
        x_std = float(X_train.std() + 1e-8)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() + 1e-8)

        X_train_n = (X_train - x_mean) / x_std
        X_val_n = (X_val - x_mean) / x_std
        y_train_n = (y_train - y_mean) / y_std
        y_val_n = (y_val - y_mean) / y_std

        train_loader = DataLoader(
            SequenceDataset(X_train_n, y_train_n),
            batch_size=64,
            shuffle=True,
        )
        val_loader = DataLoader(
            SequenceDataset(X_val_n, y_val_n),
            batch_size=256,
            shuffle=False,
        )

        model = LSTMForecaster(
            hidden=params.hidden,
            layers=params.layers,
            dropout=params.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params.lr))
        loss_fn = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        patience_left = int(params.patience)

        for _epoch in range(int(params.max_epochs)):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)

                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                if params.grad_clip is not None and params.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(params.grad_clip))
                optimizer.step()

            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).unsqueeze(1)
                    pred = model(xb)
                    val_losses.append(float(loss_fn(pred, yb).item()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = int(params.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_pred_n = []
            for xb, _yb in val_loader:
                xb = xb.to(device)
                val_pred_n.append(model(xb).detach().cpu().numpy().reshape(-1))
        val_pred_n = np.concatenate(val_pred_n) if val_pred_n else np.array([])
        val_pred = val_pred_n * y_std + y_mean

        val_mse = float(mean_squared_error(y_val, val_pred)) if len(val_pred) else float("nan")
        val_r2 = float(r2_score(y_val, val_pred)) if len(val_pred) > 1 else float("nan")
        target_var = float(np.var(y)) if len(y) else float("nan")

        last = series.tail(params.window).to_numpy(dtype=float).reshape(1, params.window, 1)
        last = np.clip(last, -params.clip_returns, params.clip_returns)
        last_n = (last - x_mean) / x_std
        x_last = torch.from_numpy(last_n.astype(np.float32)).to(device)
        with torch.no_grad():
            pred_n = float(model(x_last).item())
        pred_horizon = pred_n * y_std + y_mean

        results[asset] = {
            "pred_horizon": float(pred_horizon),
            "annual_mu": annualize_horizon_return(pred_horizon, horizon_days=horizon_days),
            "val_mse": val_mse,
            "val_r2": val_r2,
            "beta": _beta_from_mse(val_mse, target_var),
        }

    return results


def blend_with_sample_mean(
    mu_sample: pd.Series,
    model_results: dict[str, dict],
) -> tuple[pd.Series, pd.DataFrame]:
    mu_model = pd.Series({k: v["annual_mu"] for k, v in model_results.items()}, dtype=float)
    betas = pd.Series({k: v.get("beta", 0.0) for k, v in model_results.items()}, dtype=float)

    mu_model = mu_model.reindex(mu_sample.index)
    betas = betas.reindex(mu_sample.index).fillna(0.0).clip(0.0, 1.0)

    mu_blend = (1.0 - betas) * mu_sample + betas * mu_model.fillna(mu_sample)

    info = pd.DataFrame(
        {
            "mu_sample": mu_sample,
            "mu_model": mu_model,
            "beta": betas,
            "mu_blend": mu_blend,
        }
    )
    return mu_blend, info


def performance_stats(
    equity_curve: pd.Series,
    rf_rate: float = 0.0,
    turnover: float | None = None,
    cost_paid: float | None = None,
) -> dict[str, float] | None:
    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 2:
        return None

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.empty:
        return None

    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (
        TRADING_DAYS_PER_YEAR / len(daily_returns)
    ) - 1
    vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (cagr - rf_rate) / vol if vol > 0 else float("nan")

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = float(drawdown.min())

    stats: dict[str, float] = {
        "CAGR": float(cagr),
        "Volatility": vol,
        "Sharpe": float(sharpe),
        "Max Drawdown": max_dd,
    }
    if turnover is not None:
        stats["Total Turnover"] = float(turnover)
    if cost_paid is not None:
        stats["Costs Paid"] = float(cost_paid)
    return stats


def get_rebalance_dates(returns: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    if freq == "D":
        return returns.index
    if freq == "W":
        return returns.resample("W-FRI").last().index
    if freq == "M":
        return returns.resample("M").last().index
    return returns.index


def backtest_strategy(
    returns: pd.DataFrame,
    forecast_method: str,
    rf_rate: float,
    rebalance_freq: str,
    lookback_days: int,
    horizon_days: int,
    cov_method: str,
    ewma_lambda: float,
    risk_aversion: float,
    max_weight: float | None,
    turnover_limit: float | None,
    tcost_bps: float,
    rf_rolling_window: int,
    rf_cv_splits: int,
    lstm_params: LSTMParams,
) -> tuple[pd.Series | None, dict[str, float] | None]:
    rebal_dates = list(get_rebalance_dates(returns, rebalance_freq))
    if len(rebal_dates) < 2:
        return None, None

    lookback_days = int(lookback_days)
    start_pos = min(max(lookback_days, 1), len(returns) - 1)
    start_date = returns.index[start_pos]
    rebal_dates = [d for d in rebal_dates if d >= start_date]
    if len(rebal_dates) < 2:
        return None, None

    portfolio_values: list[float] = []
    portfolio_index: list[pd.Timestamp] = []
    current_value = 1.0

    prev_w: np.ndarray | None = None
    total_turnover = 0.0
    total_cost_paid = 0.0
    cost_rate = float(max(tcost_bps, 0.0)) / 10_000.0

    for i, d in enumerate(rebal_dates[:-1]):
        hist = returns.loc[:d].tail(lookback_days)
        if len(hist) < max(60, int(lstm_params.window) + horizon_days + 10):
            continue

        mu_sample, sigma = estimate_annualized_moments(hist, cov_method=cov_method, ewma_lambda=ewma_lambda)

        if forecast_method == "Sample Mean":
            mu = mu_sample
        elif forecast_method == "Random Forest":
            rf_results = rf_forecast(
                hist,
                rolling_window=rf_rolling_window,
                horizon_days=horizon_days,
                cv_splits=rf_cv_splits,
                random_state=42,
            )
            mu = pd.Series({a: rf_results[a]["annual_mu"] for a in rf_results}).reindex(mu_sample.index)
            mu = mu.fillna(mu_sample)
        elif forecast_method == "LSTM (PyTorch)":
            lstm_results = lstm_forecast(hist, horizon_days=horizon_days, params=lstm_params)
            mu = pd.Series({a: lstm_results[a]["annual_mu"] for a in lstm_results}).reindex(mu_sample.index)
            mu = mu.fillna(mu_sample)
        elif forecast_method == "Ensemble (Sample + RF)":
            rf_results = rf_forecast(
                hist,
                rolling_window=rf_rolling_window,
                horizon_days=horizon_days,
                cv_splits=rf_cv_splits,
                random_state=42,
            )
            mu, _details = blend_with_sample_mean(mu_sample, rf_results)
        elif forecast_method == "Ensemble (Sample + LSTM)":
            lstm_results = lstm_forecast(hist, horizon_days=horizon_days, params=lstm_params)
            mu, _details = blend_with_sample_mean(mu_sample, lstm_results)
        else:
            mu = mu_sample

        mu = mu.reindex(mu_sample.index).fillna(mu_sample)

        try:
            w = optimize_portfolio_mean_variance(
                mu_annual=mu,
                sigma_annual=sigma,
                rf_annual=rf_rate,
                risk_aversion=risk_aversion,
                long_only=True,
                max_weight=max_weight,
                prev_weights=prev_w,
                turnover_limit=turnover_limit,
            )
        except Exception:
            w = np.ones(len(mu_sample), dtype=float) / len(mu_sample)

        turnover = float(np.sum(np.abs(w - prev_w))) if prev_w is not None else float(np.sum(np.abs(w)))
        total_turnover += turnover
        cost_paid = cost_rate * turnover
        total_cost_paid += cost_paid
        current_value *= max(0.0, 1.0 - cost_paid)

        prev_w = w

        next_d = rebal_dates[i + 1]
        period_rets = returns.loc[(returns.index > d) & (returns.index <= next_d)]
        if period_rets.empty:
            continue

        for t, row in period_rets.iterrows():
            r_p = float(np.dot(w, row.fillna(0.0).to_numpy()))
            current_value *= 1.0 + r_p
            portfolio_values.append(current_value)
            portfolio_index.append(t)

    if not portfolio_values:
        return None, None

    equity_curve = pd.Series(portfolio_values, index=portfolio_index, name="Portfolio")
    stats = performance_stats(
        equity_curve,
        rf_rate=rf_rate,
        turnover=total_turnover,
        cost_paid=total_cost_paid,
    )
    return equity_curve, stats


st.sidebar.title("Settings")

tickers_raw = st.sidebar.text_input("Tickers", "TLT,LQD,HYG,SPY")
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", dt.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

rf_rate = st.sidebar.number_input("Risk-Free Rate (annual)", value=0.03, format="%.3f")

forecast_method = st.sidebar.selectbox(
    "Forecast Method",
    [
        "Sample Mean",
        "Random Forest",
        "LSTM (PyTorch)",
        "Ensemble (Sample + RF)",
        "Ensemble (Sample + LSTM)",
    ],
)

rebalance_freq_label = st.sidebar.selectbox(
    "Backtest Rebalance Frequency",
    ["Monthly", "Weekly", "Daily"],
    index=0,
)
rebalance_freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}
rebalance_freq = rebalance_freq_map[rebalance_freq_label]

horizon_mode = st.sidebar.selectbox("Forecast Horizon", ["Auto (rebalance)", "Manual"], index=0)
if horizon_mode == "Auto (rebalance)":
    horizon_days = infer_horizon_days(rebalance_freq)
else:
    horizon_days = st.sidebar.slider("Horizon (trading days)", 1, 60, infer_horizon_days(rebalance_freq))

lookback_days = st.sidebar.slider(
    "Lookback Window (trading days)",
    min_value=60,
    max_value=756,
    value=252,
    step=30,
)

cov_method = st.sidebar.selectbox("Covariance Estimator", ["Sample", "Ledoit-Wolf", "EWMA"], index=1)
ewma_lambda = st.sidebar.slider("EWMA lambda", 0.70, 0.99, 0.94, step=0.01)

risk_aversion = st.sidebar.slider("Risk Aversion (higher = safer)", 0.1, 50.0, 10.0, step=0.1)
max_weight = st.sidebar.slider("Max Weight per Asset", 0.05, 1.00, 0.60, step=0.05)

turnover_limit_enabled = st.sidebar.checkbox("Enable Turnover Limit", value=False)
turnover_limit = None
if turnover_limit_enabled:
    turnover_limit = st.sidebar.slider("Max Turnover per Rebalance (L1)", 0.05, 2.0, 0.50, step=0.05)

tcost_bps = st.sidebar.slider("Transaction Costs (bps per $ traded)", 0.0, 50.0, 5.0, step=0.5)

with st.sidebar.expander("Model Settings", expanded=False):
    rf_rolling_window = st.slider("RF rolling window (lags)", 5, 60, 20, step=5)
    rf_cv_splits = st.slider("RF time-series CV splits", 2, 8, 5, step=1)

    lstm_window = st.slider("LSTM window (days)", 10, 90, 30, step=5)
    lstm_hidden = st.slider("LSTM hidden size", 8, 128, 32, step=8)
    lstm_layers = st.slider("LSTM layers", 1, 4, 2, step=1)
    lstm_dropout = st.slider("LSTM dropout", 0.0, 0.5, 0.1, step=0.05)
    lstm_lr = st.slider("LSTM learning rate", 1e-4, 5e-3, 1e-3, step=1e-4, format="%.4f")
    lstm_max_epochs = st.slider("LSTM max epochs", 5, 50, 15, step=1)
    lstm_patience = st.slider("LSTM early-stop patience", 2, 10, 4, step=1)
    lstm_grad_clip = st.slider("LSTM grad clip", 0.0, 5.0, 1.0, step=0.5)
    lstm_clip_returns = st.slider("Clip returns (abs)", 0.02, 0.50, 0.20, step=0.02)
    lstm_seed = st.number_input("LSTM seed", value=42, step=1)

lstm_params = LSTMParams(
    window=int(lstm_window),
    hidden=int(lstm_hidden),
    layers=int(lstm_layers),
    dropout=float(lstm_dropout),
    lr=float(lstm_lr),
    max_epochs=int(lstm_max_epochs),
    patience=int(lstm_patience),
    grad_clip=float(lstm_grad_clip),
    clip_returns=float(lstm_clip_returns),
    seed=int(lstm_seed),
)

run_backtest = st.sidebar.button("Run Backtest")

if not tickers:
    st.error("Please enter at least one valid ticker.")
    st.stop()

prices = load_price_data(tickers, start_date, end_date)
if prices.empty:
    st.error("No price data. Check tickers/date range.")
    st.stop()

returns = compute_simple_returns(prices)
mu_sample, sigma = estimate_annualized_moments(returns, cov_method=cov_method, ewma_lambda=ewma_lambda)

n_assets = len(mu_sample)
min_feasible_max_weight = 1.0 / max(n_assets, 1)
if max_weight < min_feasible_max_weight - 1e-12:
    st.warning(
        f"`Max Weight per Asset` is too low for {n_assets} assets; "
        f"raising it to {min_feasible_max_weight:.2%} to keep the problem feasible."
    )
    max_weight = min_feasible_max_weight

st.title("AI Portfolio Optimizer")
st.caption(
    f"Forecast horizon: {horizon_days} trading day(s) • Covariance: {cov_method} • "
    f"Costs: {tcost_bps:.1f} bps per $ traded"
)

st.subheader("Price History")
st.line_chart(prices)

st.subheader("Expected Returns (Current Estimation)")

model_details = None
if forecast_method == "Sample Mean":
    mu = mu_sample
    st.write("Using **Sample Mean** annualized returns.")
    model_details = None

elif forecast_method in {"Random Forest", "Ensemble (Sample + RF)"}:
    st.write("Using **Random Forest** (time-series CV + next-horizon prediction).")
    rf_results = rf_forecast(
        returns,
        rolling_window=rf_rolling_window,
        horizon_days=horizon_days,
        cv_splits=rf_cv_splits,
        random_state=42,
    )
    if not rf_results:
        st.warning("Not enough data for RF; falling back to sample means.")
        mu = mu_sample
    elif forecast_method == "Random Forest":
        mu = pd.Series({a: rf_results[a]["annual_mu"] for a in rf_results}).reindex(mu_sample.index)
        mu = mu.fillna(mu_sample)
    else:
        mu, model_details = blend_with_sample_mean(mu_sample, rf_results)

    perf_df = pd.DataFrame(rf_results).T
    if not perf_df.empty:
        st.dataframe(
            perf_df[["pred_horizon", "annual_mu", "cv_r2", "cv_mse", "beta"]].style.format(
                {"pred_horizon": "{:.4f}", "annual_mu": "{:.2%}", "cv_r2": "{:.3f}", "cv_mse": "{:.6f}", "beta": "{:.2f}"}
            )
        )

elif forecast_method in {"LSTM (PyTorch)", "Ensemble (Sample + LSTM)"}:
    st.write("Using **LSTM (PyTorch)** (normalized inputs + early stopping).")
    lstm_results = lstm_forecast(returns, horizon_days=horizon_days, params=lstm_params)
    if not lstm_results:
        st.warning("Not enough data for LSTM; falling back to sample means.")
        mu = mu_sample
    elif forecast_method == "LSTM (PyTorch)":
        mu = pd.Series({a: lstm_results[a]["annual_mu"] for a in lstm_results}).reindex(mu_sample.index)
        mu = mu.fillna(mu_sample)
    else:
        mu, model_details = blend_with_sample_mean(mu_sample, lstm_results)

    perf_df = pd.DataFrame(lstm_results).T
    if not perf_df.empty:
        st.dataframe(
            perf_df[["pred_horizon", "annual_mu", "val_r2", "val_mse", "beta"]].style.format(
                {"pred_horizon": "{:.4f}", "annual_mu": "{:.2%}", "val_r2": "{:.3f}", "val_mse": "{:.6f}", "beta": "{:.2f}"}
            )
        )
else:
    mu = mu_sample

mu = mu.reindex(mu_sample.index).fillna(mu_sample)

if model_details is not None:
    details = model_details.reindex(mu_sample.index)
    fmt = {}
    for col in details.columns:
        if col in {"mu_sample", "mu_model", "mu_blend"}:
            fmt[col] = "{:.2%}"
        elif col == "beta":
            fmt[col] = "{:.2f}"
    st.dataframe(details.style.format(fmt))
else:
    st.dataframe(mu.to_frame("mu_annual").style.format("{:.2%}"))

st.subheader("Optimal Portfolio (Mean-Variance Objective)")
st.caption("Objective: maximize (mu - rf)^T w - risk_aversion * w^T Sigma w, with weights summing to 1.")

try:
    weights = optimize_portfolio_mean_variance(
        mu_annual=mu,
        sigma_annual=sigma,
        rf_annual=rf_rate,
        risk_aversion=risk_aversion,
        long_only=True,
        max_weight=max_weight,
        prev_weights=None,
        turnover_limit=None,
    )
    opt_ret = float(weights @ mu.to_numpy(dtype=float))
    opt_vol = float(np.sqrt(weights @ sigma.to_numpy(dtype=float) @ weights))
    opt_sharpe = (opt_ret - rf_rate) / opt_vol if opt_vol > 0 else float("nan")

    df_w = pd.DataFrame({"Weight": weights}, index=mu.index)
    st.dataframe(df_w.style.format("{:.2%}"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Return (annual)", f"{opt_ret:.2%}")
    c2.metric("Volatility (annual)", f"{opt_vol:.2%}")
    c3.metric("Sharpe (ex-ante)", f"{opt_sharpe:.2f}")
except Exception as e:
    st.error(f"Optimization failed: {e}")

st.markdown("---")
st.subheader("Backtest: Walk-Forward Cumulative Returns")

if run_backtest:
    with st.spinner("Running backtest..."):
        equity_curve, stats = backtest_strategy(
            returns,
            forecast_method=forecast_method,
            rf_rate=rf_rate,
            rebalance_freq=rebalance_freq,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            cov_method=cov_method,
            ewma_lambda=ewma_lambda,
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            turnover_limit=turnover_limit,
            tcost_bps=tcost_bps,
            rf_rolling_window=rf_rolling_window,
            rf_cv_splits=rf_cv_splits,
            lstm_params=lstm_params,
        )

    if equity_curve is None:
        st.error("Backtest could not be run (not enough data or dates).")
    else:
        st.write("**Equity Curve (Growth of $1)**")
        st.line_chart(equity_curve)

        if stats:
            st.write("**Backtest Performance Summary**")
            stats_df = pd.DataFrame(stats, index=["Backtest"])
            st.dataframe(
                stats_df.style.format(
                    {
                        "CAGR": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Sharpe": "{:.2f}",
                        "Max Drawdown": "{:.2%}",
                        "Total Turnover": "{:.2f}",
                        "Costs Paid": "{:.2%}",
                    }
                )
            )

            drawdown = equity_curve / equity_curve.cummax() - 1
            fig, ax = plt.subplots()
            ax.plot(drawdown.index, drawdown.values)
            ax.set_title("Drawdown")
            ax.set_ylabel("Drawdown")
            st.pyplot(fig)
        else:
            st.warning("Not enough data to compute performance statistics.")
else:
    st.info("Set your options in the sidebar and click **Run Backtest** to simulate historical performance.")
