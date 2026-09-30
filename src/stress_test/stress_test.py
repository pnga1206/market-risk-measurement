"""
stress_test.py – Kiểm định Stress Test VaR cho 3 giai đoạn khủng hoảng
========================================================================
Giai đoạn phân tích:
    - 2008: Khủng hoảng tài chính toàn cầu (2007-2009) → data/stress_2008/
    - 2020: Đại dịch COVID-19                           → data/main/ (lọc 2020)
    - 2022: Khủng hoảng lạm phát / thắt chặt tiền tệ   → data/main/ (lọc 2022)

Với mỗi giai đoạn, stress test tính:
    1. Thống kê lợi suất (min, max, std, VaR thực tế trung bình)
    2. VaR Historical, Parametric, Monte Carlo (full-window trên dữ liệu giai đoạn)
    3. GARCH-VaR trên dữ liệu giai đoạn
    4. Tỷ lệ vi phạm trong giai đoạn căng thẳng

Input:
    Chuỗi portfolio_return từ data/main/ và data/stress_2008/

Output:
    dict chứa DataFrame kết quả cho từng giai đoạn + bảng so sánh tổng hợp
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model
from src.var.var_models import (
    calculate_historical_var,
    calculate_historical_es,
    calculate_parametric_var,
    calculate_monte_carlo_var_es,
)
from src.config import CONF_LEVEL, RANDOM_SEED, MC_SIMULATIONS

warnings.filterwarnings("ignore")

# ── Định nghĩa các giai đoạn stress ──────────────────────────────────────────
STRESS_PERIODS = {
    "2008": ("2007-01-01", "2009-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}

# Năm nào dùng data riêng (stress_2008), còn lại lấy từ main
_DATA_SOURCE = {
    "2008": "stress_2008",
    "2020": "main",
    "2022": "main",
}

_BASE_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
_TABLE_DIR  = os.path.join(_OUTPUT_DIR, "tables")
_FIGURE_DIR = os.path.join(_OUTPUT_DIR, "figures")


def _load_returns(source: str, start: str, end: str) -> pd.Series:
    """
    Đọc portfolio_return từ thư mục nguồn và lọc theo khoảng thời gian.

    Parameters
    ----------
    source : str
        "main" hoặc "stress_2008"
    start, end : str
        Ngày bắt đầu và kết thúc (định dạng "YYYY-MM-DD").

    Returns
    -------
    pd.Series – Log-return của danh mục, index là DatetimeIndex.
    """
    csv_path = os.path.join(_BASE_DIR, source, "portfolio_returns.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.index.name = "Date"
    series = df["portfolio_return"].sort_index()
    return series.loc[start:end].dropna()


def _calculate_var_simple(
    returns: pd.Series,
    conf_level: float,
    n_sims: int,
    seed: int,
) -> dict:
    """
    Tính VaR Historical, Parametric, Monte Carlo và ES Historical
    """
    historical_var = calculate_historical_var(returns, conf_level)
    historical_es = calculate_historical_es(returns, conf_level)
    parametric_var = calculate_parametric_var(returns, conf_level)

    monte_carlo_var, _ = calculate_monte_carlo_var_es(
        returns,
        conf_level=conf_level,
        n_sims=n_sims,
        seed=seed,
    )

    return {
        "VaR_Historical": round(historical_var, 6),
        "VaR_Parametric": round(parametric_var, 6),
        "VaR_MonteCarlo": round(monte_carlo_var, 6),
        "ES_Historical": round(historical_es, 6),
    }


def _calculate_garch_var(returns: pd.Series, conf_level: float) -> dict:
    """Ước lượng GARCH(1,1), tạo VaR và ES theo từng ngày trong giai đoạn."""
    z = norm.ppf(conf_level)
    es_factor = norm.pdf(z) / (1.0 - conf_level)

    scaled = returns * 100.0
    am = arch_model(
        scaled,
        vol="Garch",
        p=1,
        q=1,
        dist="normal",
        mean="Constant",
    )
    res = am.fit(disp="off")

    # conditional_volatility là độ lệch chuẩn có điều kiện, không lấy căn bậc hai.
    sigma = res.conditional_volatility / 100.0
    sigma = pd.Series(sigma, index=returns.index).reindex(returns.index)

    # res.params["mu"] đang ở đơn vị phần trăm vì dữ liệu được nhân 100.
    mu = float(res.params.get("mu", 0.0)) / 100.0

    var_series = z * sigma - mu
    es_series = es_factor * sigma - mu

    return {
        "GARCH_Volatility": round(float(sigma.mean()), 8),
        "VaR_GARCH": round(float(var_series.mean()), 6),
        "ES_GARCH": round(float(es_series.mean()), 6),
        "VaR_GARCH_series": var_series,
        "ES_GARCH_series": es_series,
    }


def _descriptive_stats(returns: pd.Series) -> dict:
    """
    Tính thống kê mô tả cơ bản cho một giai đoạn căng thẳng.

    Parameters
    ----------
    returns : pd.Series
        Chuỗi log-return của giai đoạn.

    Returns
    -------
    dict: n_obs, mean, std, min, max, skewness, kurtosis, worst_5_days
    """
    return {
        "n_obs"       : len(returns),
        "mean (%)"    : round(returns.mean()  * 100, 4),
        "std (%)"     : round(returns.std()   * 100, 4),
        "min (%)"     : round(returns.min()   * 100, 4),
        "max (%)"     : round(returns.max()   * 100, 4),
        "skewness"    : round(float(returns.skew()), 4),
        "kurtosis"    : round(float(returns.kurtosis()), 4),
        "worst_5_days": returns.nsmallest(5).round(6).tolist(),
    }


def run_period_stress_test(
    period_name: str,
    conf_level: float = CONF_LEVEL,
    n_sims: int = MC_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Chạy stress test cho một giai đoạn cụ thể.

    Các tỷ lệ vi phạm được tính trên chính mẫu dùng để ước lượng VaR,
    vì vậy đây là kết quả in-sample.
    """
    if period_name not in STRESS_PERIODS:
        raise ValueError(
            f"Giai đoạn không hợp lệ: {period_name}. "
            f"Chọn từ {list(STRESS_PERIODS.keys())}"
        )

    start, end = STRESS_PERIODS[period_name]
    source = _DATA_SOURCE[period_name]

    returns = _load_returns(source, start, end)
    if len(returns) < 30:
        raise ValueError(
            f"Không đủ dữ liệu cho giai đoạn {period_name} "
            f"(chỉ có {len(returns)} ngày)."
        )

    stats = _descriptive_stats(returns)
    var = _calculate_var_simple(returns, conf_level, n_sims, seed)
    garch = _calculate_garch_var(returns, conf_level)

    # Tính vi phạm in-sample cho các VaR dạng scalar.
    violations = {}
    var_methods = {
        "Historical": var["VaR_Historical"],
        "Parametric": var["VaR_Parametric"],
        "MonteCarlo": var["VaR_MonteCarlo"],
    }

    for method, var_value in var_methods.items():
        n_violations = int((returns < -var_value).sum())
        violations[method] = {
            "n_violations": n_violations,
            "violation_rate (%)": round(
                n_violations / len(returns) * 100, 4
            ),
            "expected_rate (%)": round(
                (1 - conf_level) * 100, 4
            ),
        }

    # GARCH dùng ngưỡng VaR riêng cho từng ngày.
    garch_var_series = garch["VaR_GARCH_series"].reindex(returns.index)
    valid_garch_dates = garch_var_series.notna()

    garch_hits = (
        returns.loc[valid_garch_dates]
        < -garch_var_series.loc[valid_garch_dates]
    )
    n_garch_violations = int(garch_hits.sum())
    n_garch_observations = int(valid_garch_dates.sum())

    violations["GARCH"] = {
        "n_violations": n_garch_violations,
        "violation_rate (%)": (
            round(
                n_garch_violations / n_garch_observations * 100,
                4,
            )
            if n_garch_observations > 0
            else 0.0
        ),
        "expected_rate (%)": round(
            (1 - conf_level) * 100, 4
        ),
    }

    return {
        "returns": returns,
        "stats": stats,
        "var": var,
        "garch": garch,
        "violations": violations,
    }


def run_all_stress_tests(
    conf_level: float = CONF_LEVEL,
    n_sims: int = MC_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Chạy stress test cho cả 3 giai đoạn: 2008, 2020, 2022.

    Returns
    -------
    dict với key là tên giai đoạn ("2008", "2020", "2022"),
    mỗi value là kết quả từ run_period_stress_test().
    """
    results = {}
    for period in ["2008", "2020", "2022"]:
        print(f"  → Đang xử lý giai đoạn {period}...")
        try:
            results[period] = run_period_stress_test(period, conf_level, n_sims, seed)
        except Exception as e:
            print(f"  [CẢNH BÁO] Giai đoạn {period} lỗi: {e}")
            results[period] = None
    return results


def build_comparison_table(stress_results: dict) -> pd.DataFrame:
    """
    Tổng hợp kết quả của các giai đoạn thành một bảng so sánh.

    Parameters
    ----------
    stress_results : dict
        Đầu ra từ run_all_stress_tests().

    Returns
    -------
    pd.DataFrame – Bảng so sánh VaR, ES, và vi phạm qua các giai đoạn.
    """
    rows = []
    for period, res in stress_results.items():
        if res is None:
            continue
        row = {
            "Giai đoạn"              : period,
            "Số ngày QS"             : res["stats"]["n_obs"],
            "Return TB (%)"          : res["stats"]["mean (%)"],
            "Std (%)"                : res["stats"]["std (%)"],
            "Return thấp nhất (%)"   : res["stats"]["min (%)"],
            "VaR_Historical (%)"     : round(res["var"]["VaR_Historical"] * 100, 4),
            "VaR_Parametric (%)"     : round(res["var"]["VaR_Parametric"] * 100, 4),
            "VaR_MonteCarlo (%)"     : round(res["var"]["VaR_MonteCarlo"] * 100, 4),
            "VaR_GARCH (%)"          : round(res["garch"]["VaR_GARCH"]    * 100, 4),
            "ES_Historical (%)"      : round(res["var"]["ES_Historical"]   * 100, 4),
            "ES_GARCH (%)"           : round(res["garch"]["ES_GARCH"]      * 100, 4),
            "Vi phạm HVar (%)"       : res["violations"]["Historical"]["violation_rate (%)"],
            "Vi phạm PVar (%)"       : res["violations"]["Parametric"]["violation_rate (%)"],
            "Vi phạm GARCH-VaR (%)"  : res["violations"]["GARCH"]["violation_rate (%)"],
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("Giai đoạn")
