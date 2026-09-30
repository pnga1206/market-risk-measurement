"""
kupiec_test.py – Kiểm định Kupiec (Proportion Of Failures – POF Test)
=======================================================================
Mục đích:
    Kiểm tra liệu tỷ lệ vi phạm VaR quan sát có thống kê bằng với mức kỳ vọng
    từ mức độ tin cậy đã chọn hay không.

    H0: p_thực = p_kỳ_vọng  (mô hình VaR được hiệu chỉnh đúng)
    H1: p_thực ≠ p_kỳ_vọng

    Thống kê kiểm định LR_pof ~ chi-squared(1)

Input:
    hit_series   : pd.Series nhị phân (1 = vi phạm, 0 = không vi phạm)
    conf_level   : float, mức độ tin cậy (vd: 0.975)
    alpha        : float, mức ý nghĩa thống kê (mặc định 0.05)

Output:
    dict với các khóa: n_obs, n_violations, violation_rate, expected_rate,
                       LR_statistic, p_value, reject_H0
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_pof_test(
    hit_series: pd.Series,
    conf_level: float = 0.975,
    alpha: float = 0.05,
) -> dict:
    """
    Kupiec Proportion Of Failures (POF) Test.

    Parameters
    ----------
    hit_series : pd.Series
        Chuỗi nhị phân: 1 nếu thực tế < -VaR (vi phạm), 0 nếu không.
    conf_level : float
        Mức độ tin cậy của VaR (vd: 0.975 tương ứng tail 2.5%).
    alpha : float
        Mức ý nghĩa thống kê để bác bỏ H0 (mặc định 5%).

    Returns
    -------
    dict
        Kết quả kiểm định gồm: số quan sát, số vi phạm, tỷ lệ vi phạm,
        tỷ lệ kỳ vọng, thống kê LR, p-value, kết luận bác bỏ H0 hay không.
    """
    hits   = hit_series.dropna().astype(int)
    n      = len(hits)           # Tổng số ngày quan sát
    x      = int(hits.sum())     # Số ngày vi phạm

    p_hat = x / n if n > 0 else 0.0    # Tỷ lệ vi phạm quan sát
    p_exp = 1.0 - conf_level            # Tỷ lệ kỳ vọng (vd: 2.5%)

    # Tính log-likelihood ratio (LR_pof)
    # Tránh log(0) bằng cách xử lý trường hợp biên
    eps = 1e-10
    p_hat_clipped = np.clip(p_hat, eps, 1 - eps)
    p_exp_clipped = np.clip(p_exp, eps, 1 - eps)

    if x == 0:
        # Không có vi phạm nào: trường hợp biên
        ll_null = n * np.log(1 - p_exp_clipped)
        ll_alt  = 0.0  # log(1^0 * 0^0) ~ 0
    elif x == n:
        # Tất cả đều vi phạm: trường hợp biên
        ll_null = n * np.log(p_exp_clipped)
        ll_alt  = 0.0
    else:
        ll_null = (
            x * np.log(p_exp_clipped)
            + (n - x) * np.log(1 - p_exp_clipped)
        )
        ll_alt = (
            x * np.log(p_hat_clipped)
            + (n - x) * np.log(1 - p_hat_clipped)
        )

    lr_stat  = -2.0 * (ll_null - ll_alt)   # Thống kê LR
    p_value  = chi2.sf(lr_stat, df=1)       # p-value (chi2, bậc tự do = 1)
    reject   = bool(p_value < alpha)        # Có bác bỏ H0 không?

    return {
        "n_obs"            : n,
        "n_violations"     : x,
        "violation_rate"   : round(p_hat * 100, 4),      # %
        "expected_rate"    : round(p_exp * 100, 4),      # %
        "LR_statistic"     : round(float(lr_stat), 6),
        "p_value"          : round(float(p_value), 6),
        "reject_H0"        : reject,
        "alpha"            : alpha,
        "interpretation"   : (
            "Bác bỏ H0: Mô hình VaR KHÔNG được hiệu chỉnh đúng (p-value < alpha)."
            if reject else
            "Không bác bỏ H0: Mô hình VaR được hiệu chỉnh phù hợp (p-value ≥ alpha)."
        ),
    }


def run_kupiec_all(violations_df: pd.DataFrame, conf_level: float = 0.975, alpha: float = 0.05) -> pd.DataFrame:
    """
    Chạy Kupiec POF Test cho tất cả các phương pháp VaR trong violations_df.

    Parameters
    ----------
    violations_df : pd.DataFrame
        Đầu ra từ violations.compute_violations().
    conf_level : float
        Mức độ tin cậy.
    alpha : float
        Mức ý nghĩa thống kê.

    Returns
    -------
    pd.DataFrame
        Bảng kết quả Kupiec cho từng phương pháp.
    """
    hit_cols = [c for c in violations_df.columns if c.endswith("_violation")]
    rows = []
    for col in hit_cols:
        method = col.replace("_violation", "")
        result = kupiec_pof_test(violations_df[col], conf_level=conf_level, alpha=alpha)
        result["Phương pháp"] = method
        rows.append(result)

    df_out = pd.DataFrame(rows).set_index("Phương pháp")
    # Sắp xếp cột để dễ đọc
    ordered_cols = [
        "n_obs", "n_violations", "violation_rate", "expected_rate",
        "LR_statistic", "p_value", "alpha", "reject_H0", "interpretation",
    ]
    return df_out[[c for c in ordered_cols if c in df_out.columns]]
