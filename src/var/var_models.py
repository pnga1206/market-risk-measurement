import numpy as np
import pandas as pd
from scipy.stats import norm
from src.config import CONF_LEVEL, MC_SIMULATIONS, RANDOM_SEED

def _prepare_returns(returns: pd.Series) -> pd.Series:
    """Tự động chuyển đổi sang chuỗi Lợi suất (Returns) nếu đầu vào là Giá trị (Prices/Values)."""
    clean_series = returns.dropna()
    # Nếu giá trị trung bình tuyệt đối > 1, dữ liệu đầu vào khả năng cao là Giá trị danh mục -> Tính pct_change
    if clean_series.abs().mean() > 1.0:
        clean_series = clean_series.pct_change().dropna()
    return clean_series

def calculate_historical_var(returns: pd.Series, conf_level: float = CONF_LEVEL) -> float:
    ret = _prepare_returns(returns)
    # VaR là số dương thể hiện mức lỗ (lấy percentile ở đuôi trái)
    var_val = -np.percentile(ret, (1 - conf_level) * 100)
    return float(var_val)

def calculate_parametric_var(returns: pd.Series, conf_level: float = CONF_LEVEL) -> float:
    ret = _prepare_returns(returns)
    mu = np.mean(ret)
    sigma = np.std(ret, ddof=1)
    z = norm.ppf(conf_level)
    var_val = -(mu - z * sigma)
    return float(var_val)

def calculate_monte_carlo_var(returns: pd.Series, 
                               conf_level: float = CONF_LEVEL, 
                               n_sims: int = MC_SIMULATIONS, 
                               seed: int = RANDOM_SEED) -> float:
    ret = _prepare_returns(returns)
    mu = np.mean(ret)
    sigma = np.std(ret, ddof=1)
    
    np.random.seed(seed)
    simulated_returns = np.random.normal(mu, sigma, n_sims)
    var_val = -np.percentile(simulated_returns, (1 - conf_level) * 100)
    return float(var_val)

def run_rolling_var(portfolio_returns: pd.Series, 
                    window: int = 250, 
                    conf_level: float = CONF_LEVEL) -> pd.DataFrame:
    ret = _prepare_returns(portfolio_returns)
    var_results = []
    dates = ret.index
    
    for i in range(window, len(ret)):
        sub_returns = ret.iloc[i-window:i]
        current_date = dates[i]
        
        h_var = calculate_historical_var(sub_returns, conf_level)
        p_var = calculate_parametric_var(sub_returns, conf_level)
        mc_var = calculate_monte_carlo_var(sub_returns, conf_level, seed=RANDOM_SEED + i)
        
        var_results.append({
            'Date': current_date,
            'VaR_Historical': h_var,
            'VaR_Parametric': p_var,
            'VaR_MonteCarlo': mc_var
        })
        
    return pd.DataFrame(var_results).set_index('Date')