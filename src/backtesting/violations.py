"""
violations.py – Xác định các điểm vi phạm VaR
=================================================
Input:
    portfolio_returns : pd.Series – Chuỗi log-return thực tế của danh mục (index là Date)
    var_forecasts     : pd.DataFrame – DataFrame chứa các cột VaR dự báo với index là Date
                        (các cột điển hình: VaR_Historical, VaR_Parametric,
                         VaR_MonteCarlo, VaR_GARCH)

Output:
    pd.DataFrame – Cột actual_return + các cột VaR + các cột _violation tương ứng
                   và cột _hit nhị phân (1 = vi phạm, 0 = không vi phạm)
"""

import pandas as pd


def compute_violations(
    portfolio_returns: pd.Series,
    var_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gắn nhãn vi phạm cho từng phương pháp VaR.

    Vi phạm xảy ra khi: actual_return < -VaR_forecast
    (VaR được biểu diễn dưới dạng số DƯƠNG, thể hiện mức lỗ tối đa dự báo)

    Parameters
    ----------
    portfolio_returns : pd.Series
        Chuỗi log-return của danh mục, index là ngày giao dịch.
    var_forecasts : pd.DataFrame
        DataFrame VaR dự báo từ Coder 2, index là ngày giao dịch.
        Các cột: VaR_Historical, VaR_Parametric, VaR_MonteCarlo, VaR_GARCH, ...

    Returns
    -------
    pd.DataFrame
        Bảng kết hợp actual_return + VaR + cột _violation (_hit) nhị phân.
    """
    # Chỉ giữ khoảng thời gian có cả dự báo VaR lẫn lợi suất thực
    common_dates = var_forecasts.index.intersection(portfolio_returns.index)
    actual = portfolio_returns.loc[common_dates].rename("actual_return")
    var_df  = var_forecasts.loc[common_dates].copy()

    result = pd.concat([actual, var_df], axis=1)

    # Gắn nhãn vi phạm cho từng phương pháp VaR
    var_cols = [c for c in var_df.columns if c.startswith("VaR_")]
    for col in var_cols:
        hit_col = col.replace("VaR_", "") + "_violation"
        result[hit_col] = (result["actual_return"] < -result[col]).astype(int)

    return result


def violation_summary(violations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp số lượng và tỷ lệ vi phạm theo từng phương pháp.

    Parameters
    ----------
    violations_df : pd.DataFrame
        Đầu ra từ compute_violations().

    Returns
    -------
    pd.DataFrame
        Bảng tóm tắt: số ngày quan sát, số vi phạm, tỷ lệ vi phạm (%).
    """
    hit_cols = [c for c in violations_df.columns if c.endswith("_violation")]
    records = []
    n_obs = len(violations_df)

    for col in hit_cols:
        n_viol = violations_df[col].sum()
        rate   = n_viol / n_obs if n_obs > 0 else 0.0
        method = col.replace("_violation", "")
        records.append({
            "Phương pháp": method,
            "Số ngày QS"  : n_obs,
            "Số vi phạm"  : int(n_viol),
            "Tỷ lệ vi phạm (%)": round(rate * 100, 4),
        })

    return pd.DataFrame(records).set_index("Phương pháp")
