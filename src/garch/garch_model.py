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
        # Scale 100 để thư viện arch tối ưu hóa tham số tốt hơn
        sub_returns = ret.iloc[i-window:i] * 100.0
        current_date = dates[i]
        
        try:
            am = arch_model(sub_returns, vol='Garch', p=1, q=1, dist='normal', mean='Zero')
            res = am.fit(disp='off')
            
            forecast = res.forecast(horizon=1)
            next_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100.0
        except Exception:
            next_vol = np.std(sub_returns / 100.0, ddof=1)

        garch_var = z * next_vol
        garch_es = es_factor * next_vol

        results.append({
            'Date': current_date,
            'GARCH_Volatility': next_vol,
            'VaR_GARCH': garch_var,
            'ES_97_5': garch_es
        })

    return pd.DataFrame(results).set_index('Date')