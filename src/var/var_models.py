import numpy as np
import pandas as pd
from scipy.stats import norm
from src.config import CONF_LEVEL, MC_SIMULATIONS, RANDOM_SEED


def _check_returns(returns) -> pd.Series:
    """Bỏ NaN và kiểm tra đầu vào là log return thập phân."""
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        raise ValueError("Chuỗi lợi suất rỗng.")
    if r.abs().max() > 0.5:
        raise ValueError(
            "Đầu vào không giống log return thập phân (0.01 = 1%). "
            "Hãy truyền cột 'portfolio_return', không phải giá hay đơn vị %."
        )
    return r


def calculate_historical_var(returns, conf_level: float = CONF_LEVEL) -> float:
    r = _check_returns(returns).to_numpy()
    return float(-np.quantile(r, 1 - conf_level))


def calculate_historical_es(returns, conf_level: float = CONF_LEVEL) -> float:
    r = _check_returns(returns).to_numpy()
    q = np.quantile(r, 1 - conf_level)
    return float(-r[r <= q].mean())


def calculate_parametric_var(returns, conf_level: float = CONF_LEVEL) -> float:
    r = _check_returns(returns).to_numpy()
    mu, sigma = r.mean(), r.std(ddof=1)
    z = norm.ppf(conf_level)
    return float(z * sigma - mu)


def calculate_parametric_es(returns, conf_level: float = CONF_LEVEL) -> float:
    r = _check_returns(returns).to_numpy()
    mu, sigma = r.mean(), r.std(ddof=1)
    z = norm.ppf(conf_level)
    return float(sigma * norm.pdf(z) / (1 - conf_level) - mu)


def calculate_monte_carlo_var_es(returns, conf_level: float = CONF_LEVEL,
                                 n_sims: int = MC_SIMULATIONS,
                                 seed: int = RANDOM_SEED):
    """Mô phỏng r ~ N(mu, sigma) từ cửa sổ dữ liệu; trả về (VaR, ES)."""
    r = _check_returns(returns).to_numpy()
    mu, sigma = r.mean(), r.std(ddof=1)
    rng = np.random.default_rng(seed)          # generator cục bộ, không đổi seed toàn cục
    sim = rng.normal(mu, sigma, n_sims)
    q = np.quantile(sim, 1 - conf_level)
    return float(-q), float(-sim[sim <= q].mean())


def calculate_monte_carlo_var(returns, conf_level: float = CONF_LEVEL,
                              n_sims: int = MC_SIMULATIONS,
                              seed: int = RANDOM_SEED) -> float:
    return calculate_monte_carlo_var_es(returns, conf_level, n_sims, seed)[0]


# ---------- Rolling window (dùng cho backtesting) ----------
def run_rolling_var(portfolio_returns, window: int = 250,
                    conf_level: float = CONF_LEVEL,
                    n_sims: int = MC_SIMULATIONS,
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """VaR/ES dự báo cho ngày t từ `window` ngày liền trước.

    Output (index = Date, bắt đầu từ quan sát thứ window+1):
        VaR_Historical, VaR_Parametric, VaR_MonteCarlo,
        ES_Historical, ES_Parametric, ES_MonteCarlo
    Mọi cửa sổ Monte Carlo dùng cùng seed (mặc định 42) theo đề án.
    """
    ret = _check_returns(portfolio_returns)
    if len(ret) <= window:
        raise ValueError(f"Cần nhiều hơn {window} quan sát, hiện có {len(ret)}.")

    vals = ret.to_numpy()
    z = norm.ppf(conf_level)
    tail = 1 - conf_level
    rng_draws = np.random.default_rng(seed).standard_normal(n_sims)  # dùng chung mọi cửa sổ

    rows = []
    for i in range(window, len(vals)):
        w = vals[i - window:i]
        mu, sigma = w.mean(), w.std(ddof=1)

        # Historical
        q_h = np.quantile(w, tail)
        h_var, h_es = -q_h, -w[w <= q_h].mean()

        # Parametric (chuẩn)
        p_var = z * sigma - mu
        p_es = sigma * norm.pdf(z) / tail - mu

        # Monte Carlo
        sim = mu + sigma * rng_draws
        q_m = np.quantile(sim, tail)
        m_var, m_es = -q_m, -sim[sim <= q_m].mean()

        rows.append((h_var, p_var, m_var, h_es, p_es, m_es))

    cols = ["VaR_Historical", "VaR_Parametric", "VaR_MonteCarlo",
            "ES_Historical", "ES_Parametric", "ES_MonteCarlo"]
    out = pd.DataFrame(rows, index=ret.index[window:], columns=cols)
    out.index.name = "Date"
    return out