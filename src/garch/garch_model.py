import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

from src.config import CONF_LEVEL

SCALE = 100.0 


def _check_returns(returns) -> pd.Series:
    """Bỏ NaN và kiểm tra đầu vào là log return thập phân (không tự chuyển đổi)."""
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        raise ValueError("Chuỗi lợi suất rỗng.")
    if r.abs().max() > 0.5:
        raise ValueError(
            "Đầu vào không giống log return thập phân (0.01 = 1%). "
            "Hãy truyền cột 'portfolio_return', không phải giá hay đơn vị %."
        )
    return r


def calculate_ewma_volatility(returns, lambda_param: float = 0.94) -> float:
    """EWMA (RiskMetrics) - phương án dự phòng khi GARCH không hội tụ.
    `returns` là return thập phân, thứ tự cũ -> mới."""
    r = np.asarray(returns, dtype=float)
    weights = (1 - lambda_param) * lambda_param ** np.arange(len(r))[::-1]
    weights /= weights.sum()
    return float(np.sqrt(np.sum(weights * r ** 2)))


def _garch_forecast(window_pct: pd.Series):
    """Fit GARCH(1,1) trên cửa sổ (đơn vị %), trả về (mu, sigma) dạng thập phân.
    Ném lỗi nếu không hội tụ hoặc kết quả không hợp lệ."""
    am = arch_model(window_pct, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    if res.convergence_flag != 0:
        raise RuntimeError("GARCH không hội tụ")
    f = res.forecast(horizon=1, reindex=False)   # chỉ gọi 1 lần
    mu = float(f.mean.iloc[-1, 0]) / SCALE
    sigma = float(np.sqrt(f.variance.iloc[-1, 0])) / SCALE
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
        raise RuntimeError("Dự báo GARCH không hợp lệ")
    return mu, sigma


def calculate_garch_var_es_rolling(portfolio_returns: pd.Series,
                                   window: int = 250,
                                   conf_level: float = CONF_LEVEL) -> pd.DataFrame:
    """Output (index = Date):
        GARCH_Volatility, VaR_GARCH, ES_97_5,
        GARCH_Fallback (True nếu ngày đó phải dùng EWMA thay cho GARCH)
    """
    ret = _check_returns(portfolio_returns)
    if len(ret) <= window:
        raise ValueError(f"Cần nhiều hơn {window} quan sát, hiện có {len(ret)}.")

    z = norm.ppf(conf_level)
    es_factor = norm.pdf(z) / (1 - conf_level)

    rows = []
    for i in range(window, len(ret)):
        sub = ret.iloc[i - window:i]
        try:
            mu, sigma = _garch_forecast(sub * SCALE)
            fallback = False
        except Exception:
            mu = float(sub.mean())
            sigma = calculate_ewma_volatility(sub.to_numpy())
            fallback = True

        rows.append({
            "Date": ret.index[i],
            "GARCH_Volatility": sigma,
            "VaR_GARCH": z * sigma - mu,
            "ES_97_5": es_factor * sigma - mu,
            "GARCH_Fallback": fallback,
        })

    out = pd.DataFrame(rows).set_index("Date")
    n_fb = int(out["GARCH_Fallback"].sum())
    print(f"[GARCH] {n_fb}/{len(out)} ngày phải dùng EWMA dự phòng.")
    return out