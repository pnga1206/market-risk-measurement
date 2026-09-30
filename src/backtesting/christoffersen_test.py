"""
christoffersen_test.py – Kiểm định Christoffersen (Conditional Coverage Test)
===============================================================================
Mục đích:
    Kiểm tra liệu các vi phạm VaR có tập trung cụm (clustering) theo thời gian
    hay phân bổ độc lập (independence). Đây là kiểm định toàn diện hơn Kupiec vì
    kiểm tra đồng thời cả mức phủ bao phủ vô điều kiện (UC) lẫn tính độc lập (IND).

    Có 3 thống kê kiểm định:
        1. LR_uc   – Unconditional Coverage (= LR Kupiec)
        2. LR_ind  – Independence (vi phạm có cụm không?)
        3. LR_cc   – Conditional Coverage = LR_uc + LR_ind  (df=2)

    H0 (IND): Xác suất vi phạm hôm nay KHÔNG phụ thuộc vào vi phạm hôm qua.
    H0 (CC) : Mô hình VaR vừa hiệu chỉnh đúng vừa có tính độc lập.

Input:
    hit_series   : pd.Series nhị phân (1 = vi phạm, 0 = không vi phạm)
    conf_level   : float, mức độ tin cậy (vd: 0.975)
    alpha        : float, mức ý nghĩa thống kê (mặc định 0.05)

Output:
    dict với các khóa kiểm định UC, IND, CC và các ma trận chuyển tiếp
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _transition_counts(hits: np.ndarray) -> tuple:
    """
    Đếm ma trận chuyển tiếp nhị phân (0→0, 0→1, 1→0, 1→1).

    Parameters
    ----------
    hits : np.ndarray
        Mảng nhị phân 0/1.

    Returns
    -------
    tuple: (n00, n01, n10, n11)
        n_ij = số lần trạng thái i → trạng thái j.
    """
    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))
    return n00, n01, n10, n11


def christoffersen_test(
    hit_series: pd.Series,
    conf_level: float = 0.975,
    alpha: float = 0.05,
) -> dict:
    """
    Christoffersen Conditional Coverage (CC) Test.

    Gồm 3 kiểm định liên quan:
        - UC  (Unconditional Coverage)  : giống Kupiec, df=1
        - IND (Independence)            : tính độc lập chuỗi vi phạm, df=1
        - CC  (Conditional Coverage)    : tổng hợp, df=2

    Parameters
    ----------
    hit_series : pd.Series
        Chuỗi nhị phân: 1 nếu vi phạm, 0 nếu không.
    conf_level : float
        Mức độ tin cậy của VaR.
    alpha : float
        Mức ý nghĩa thống kê.

    Returns
    -------
    dict
        Kết quả đầy đủ của cả 3 kiểm định (LR, p-value, kết luận).
    """
    hits = hit_series.dropna().astype(int).values
    n    = len(hits)
    x    = int(hits.sum())

    p_hat = x / n if n > 0 else 0.0
    p_exp = 1.0 - conf_level
    eps   = 1e-10

    # ── 1. Kiểm định UC (giống Kupiec) ──────────────────────────────────────
    p_hat_c = np.clip(p_hat, eps, 1 - eps)
    p_exp_c = np.clip(p_exp, eps, 1 - eps)

    if x == 0:
        lr_uc = -2.0 * n * np.log(1.0 - p_exp_c)
    elif x == n:
        lr_uc = -2.0 * n * np.log(p_exp_c)
    else:
        ll_null_uc = x * np.log(p_exp_c) + (n - x) * np.log(1 - p_exp_c)
        ll_alt_uc  = x * np.log(p_hat_c) + (n - x) * np.log(1 - p_hat_c)
        lr_uc = -2.0 * (ll_null_uc - ll_alt_uc)

    p_val_uc  = chi2.sf(lr_uc, df=1)
    reject_uc = bool(p_val_uc < alpha)

    # ── 2. Kiểm định IND (Independence) ─────────────────────────────────────
    n00, n01, n10, n11 = _transition_counts(hits)

    # Xác suất chuyển tiếp có điều kiện
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_2  = (n01 + n11) / (n00 + n01 + n10 + n11) if n > 1 else p_hat

    # Clip để tránh log(0)
    pi_01_c = np.clip(pi_01, eps, 1 - eps)
    pi_11_c = np.clip(pi_11, eps, 1 - eps)
    pi_2_c  = np.clip(pi_2,  eps, 1 - eps)

    # Log-likelihood dưới H0 (tất cả xác suất = pi_2, không phụ thuộc lịch sử)
    ll_null_ind = (
        (n00 + n10) * np.log(1 - pi_2_c)
        + (n01 + n11) * np.log(pi_2_c)
    )

    # Log-likelihood dưới H1 (xác suất phụ thuộc trạng thái trước)
    ll_alt_ind = (
        n00 * np.log(1 - pi_01_c)
        + n01 * np.log(pi_01_c)
        + n10 * np.log(1 - pi_11_c)
        + n11 * np.log(pi_11_c)
    )

    lr_ind    = -2.0 * (ll_null_ind - ll_alt_ind)
    p_val_ind = chi2.sf(lr_ind, df=1)
    reject_ind = bool(p_val_ind < alpha)

    # ── 3. Kiểm định CC (Conditional Coverage = UC + IND) ────────────────────
    lr_cc     = lr_uc + lr_ind
    p_val_cc  = chi2.sf(lr_cc, df=2)
    reject_cc = bool(p_val_cc < alpha)

    return {
        # Thông tin cơ bản
        "n_obs"           : n,
        "n_violations"    : x,
        "violation_rate"  : round(p_hat * 100, 4),
        "expected_rate"   : round(p_exp * 100, 4),

        # Ma trận chuyển tiếp
        "n00" : n00, "n01" : n01,
        "n10" : n10, "n11" : n11,
        "pi_01"           : round(pi_01 * 100, 4),   # P(violation | no viol prev day)
        "pi_11"           : round(pi_11 * 100, 4),   # P(violation | viol prev day)

        # Kiểm định UC
        "LR_uc"           : round(float(lr_uc),  6),
        "p_value_uc"      : round(float(p_val_uc), 6),
        "reject_uc"       : reject_uc,

        # Kiểm định IND
        "LR_ind"          : round(float(lr_ind), 6),
        "p_value_ind"     : round(float(p_val_ind), 6),
        "reject_ind"      : reject_ind,

        # Kiểm định CC (tổng hợp)
        "LR_cc"           : round(float(lr_cc),  6),
        "p_value_cc"      : round(float(p_val_cc), 6),
        "reject_cc"       : reject_cc,
        "alpha"           : alpha,

        # Diễn giải kết quả
        "interpretation_ind": (
            "Bác bỏ H0 IND: Vi phạm CÓ phân cụm theo thời gian (VaR chưa bắt được cấu trúc biến động)."
            if reject_ind else
            "Không bác bỏ H0 IND: Vi phạm độc lập theo thời gian."
        ),
        "interpretation_cc": (
            "Bác bỏ H0 CC: Mô hình VaR KHÔNG đạt kiểm định bao phủ có điều kiện."
            if reject_cc else
            "Không bác bỏ H0 CC: Mô hình VaR đạt kiểm định bao phủ có điều kiện."
        ),
    }


def run_christoffersen_all(
    violations_df: pd.DataFrame,
    conf_level: float = 0.975,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Chạy Christoffersen Test cho tất cả các phương pháp VaR.

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
        Bảng kết quả đầy đủ (UC, IND, CC) cho từng phương pháp.
    """
    hit_cols = [c for c in violations_df.columns if c.endswith("_violation")]
    rows = []
    for col in hit_cols:
        method = col.replace("_violation", "")
        result = christoffersen_test(violations_df[col], conf_level=conf_level, alpha=alpha)
        result["Phương pháp"] = method
        rows.append(result)

    df_out = pd.DataFrame(rows).set_index("Phương pháp")

    # Sắp xếp cột theo nhóm để dễ đọc
    ordered_cols = [
        "n_obs", "n_violations", "violation_rate", "expected_rate",
        "n00", "n01", "n10", "n11", "pi_01", "pi_11",
        "LR_uc",  "p_value_uc",  "reject_uc",
        "LR_ind", "p_value_ind", "reject_ind",
        "LR_cc",  "p_value_cc",  "reject_cc",
        "alpha", "interpretation_ind", "interpretation_cc",
    ]
    return df_out[[c for c in ordered_cols if c in df_out.columns]]
