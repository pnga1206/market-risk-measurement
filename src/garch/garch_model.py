import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from src.config import CONF_LEVEL

def _prepare_returns(returns: pd.Series) -> pd.Series:
    clean_series = returns.dropna()
    if clean_series.abs().mean() > 1.0:
        clean_series = clean_series.pct_change().dropna()
    return clean_series

def calculate_ewma_volatility(returns: pd.Series, lambda_param: float = 0.94) -> float:
    """Phương án dự phòng EWMA (RiskMetrics) khi GARCH không hội tụ."""
    weights = (1 - lambda_param) * (lambda_param ** np.arange(len(returns))[::-1])
    weights /= weights.sum()
    var_ewma = np.sum(weights * (returns ** 2))
    return float(np.sqrt(var_ewma))

def calculate_garch_var_es_rolling(portfolio_returns: pd.Series, 
                                    window: int = 250, 
                                    conf_level: float = CONF_LEVEL) -> pd.DataFrame:
    ret = _prepare_returns(portfolio_returns)
    results = []
    dates = ret.index
    z = norm.ppf(conf_level)
    pdf_z = norm.pdf(z)
    es_factor = pdf_z / (1 - conf_level)
    
    for i in range(window, len(ret)):
        sub_returns = ret.iloc[i-window:i] * 100.0  # Scale 100 cho arch_model
        current_date = dates[i]
        
        try:
            # 1. Đổi sang mean='Constant' để mô hình tự ước lượng mu
            am = arch_model(sub_returns, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
            res = am.fit(disp='off')
            
            # Dự báo mu và variance cho ngày tiếp theo (chia 100 để về lại scale chuẩn)
            mu_forecast = res.forecast(horizon=1).mean.iloc[-1, 0] / 100.0
            next_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]) / 100.0
        except Exception:
            # Fallback sang EWMA chuyên nghiệp thay vì np.std
            mu_forecast = sub_returns.mean() / 100.0
            next_vol = calculate_ewma_volatility(sub_returns / 100.0)

        # 2. Áp dụng công thức VaR và ES có chứa thành phần mu chuẩn lý thuyết
        garch_var = (z * next_vol) - mu_forecast
        garch_es = (es_factor * next_vol) - mu_forecast

        results.append({
            'Date': current_date,
            'GARCH_Volatility': next_vol,
            'VaR_GARCH': garch_var,
            'ES_97_5': garch_es
        })

    return pd.DataFrame(results).set_index('Date')