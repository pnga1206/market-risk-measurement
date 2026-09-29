"""
run_stress_test.py – Pipeline Stress Test hoàn chỉnh (Coder 3)
===============================================================
Orchestrator chạy toàn bộ quy trình stress test cho 3 giai đoạn khủng hoảng:
    - 2008: Khủng hoảng tài chính toàn cầu
    - 2020: Đại dịch COVID-19
    - 2022: Khủng hoảng lạm phát / thắt chặt tiền tệ

Usage (chạy từ thư mục gốc dự án):
    python -m src.stress_test.run_stress_test

    # Hoặc gọi từ Python:
    from src.stress_test.run_stress_test import run_full_stress_test
    results = run_full_stress_test()
"""

import os
import warnings
import pandas as pd
from src.stress_test.plot_stress import (
    plot_return_distribution_by_period,
    plot_var_comparison,
    plot_timeseries_by_period,
    plot_violation_rate,
    plot_oos_violation_rate,
)
from src.backtesting.run_backtest import run_full_backtest
from src.stress_test.stress_oos import run_stress_oos
from src.stress_test.stress_test import run_all_stress_tests, build_comparison_table
from src.stress_test.plot_stress import (
    plot_return_distribution_by_period,
    plot_var_comparison,
    plot_timeseries_by_period,
    plot_violation_rate,
)
from src.config import CONF_LEVEL, MC_SIMULATIONS, RANDOM_SEED

warnings.filterwarnings("ignore")

_TABLE_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "tables")
_FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "figures")


def _ensure_dirs():
    os.makedirs(_TABLE_DIR,  exist_ok=True)
    os.makedirs(_FIGURE_DIR, exist_ok=True)


def run_full_stress_test(
    conf_level: float = CONF_LEVEL,
    n_sims: int = MC_SIMULATIONS,
    seed: int = RANDOM_SEED,
    save_outputs: bool = True,
    backtest_results: dict | None = None,
) -> dict:
    """
    Chạy toàn bộ pipeline Stress Test cho 3 giai đoạn.

    Parameters
    ----------
    conf_level : float
        Mức độ tin cậy VaR (mặc định 0.975).
    n_sims : int
        Số kịch bản Monte Carlo (mặc định 10,000).
    seed : int
        Random seed (mặc định 42).
    save_outputs : bool
        Có lưu bảng CSV và biểu đồ ra outputs/ không.

    Returns
    -------
    dict với các khóa:
        - "stress_results"   : dict  – kết quả từng giai đoạn (2008, 2020, 2022)
        - "comparison_table" : pd.DataFrame – bảng so sánh tổng hợp
    """
    _ensure_dirs()
    print("=" * 60)
    print("STRESS TEST VaR – CODER 3")
    print("=" * 60)

    # ── Bước 1: Chạy stress test cho từng giai đoạn ─────────────────────────
    print("\n[1] Chạy stress test cho 3 giai đoạn khủng hoảng...")
    stress_results = run_all_stress_tests(conf_level=conf_level, n_sims=n_sims, seed=seed)
    if backtest_results is None:

        backtest_results = run_full_backtest(
        conf_level=conf_level,
        save_outputs=False,
    )

    stress_oos_results = run_stress_oos(
    backtest_results=backtest_results,
    conf_level=conf_level,
)

    # ── Bước 2: In kết quả chi tiết từng giai đoạn ──────────────────────────
    for period, res in stress_results.items():
        if res is None:
            print(f"\n  [SKIP] Giai đoạn {period}: không có dữ liệu.")
            continue
        print(f"\n{'─' * 40}")
        print(f"GIAI ĐOẠN {period}")
        print(f"{'─' * 40}")

        stats = res["stats"]
        print(f"  Số ngày quan sát : {stats['n_obs']}")
        print(f"  Return TB        : {stats['mean (%)']:.4f}%")
        print(f"  Std              : {stats['std (%)']:.4f}%")
        print(f"  Return thấp nhất : {stats['min (%)']:.4f}%")
        print(f"  Skewness         : {stats['skewness']}")
        print(f"  Kurtosis         : {stats['kurtosis']}")

        print(f"\n  VaR (97.5%):          |  Tỷ lệ vi phạm:")
        for method, var_val in [
            ("Historical", res["var"]["VaR_Historical"]),
            ("Parametric", res["var"]["VaR_Parametric"]),
            ("MonteCarlo", res["var"]["VaR_MonteCarlo"]),
            ("GARCH"     , res["garch"]["VaR_GARCH"]),
        ]:
            viol_rate = res["violations"][method]["violation_rate (%)"]
            exp_rate  = res["violations"][method]["expected_rate (%)"]
            print(f"    {method:<12}: {var_val*100:7.4f}%  |  {viol_rate:.2f}% (kỳ vọng: {exp_rate:.1f}%)")

    # ── Bước 3: Tổng hợp bảng so sánh ──────────────────────────────────────
    print(f"\n[2] Bảng so sánh tổng hợp 3 giai đoạn:")
    comparison_df = build_comparison_table(stress_results)
    print(comparison_df.to_string())

    # ── Bước 4: Lưu kết quả ─────────────────────────────────────────────────
    if save_outputs:
        comparison_df.to_csv(os.path.join(_TABLE_DIR, "stress_test_comparison.csv"))
        
        # Lưu chi tiết từng giai đoạn
        for period, res in stress_results.items():
            if res is None:
                continue
            detail_rows = []
            for method in ["Historical", "Parametric", "MonteCarlo", "GARCH"]:
                var_val   = res["var"].get(f"VaR_{method}", res["garch"].get("VaR_GARCH"))
                es_val = (res["garch"].get("ES_GARCH") if method == "GARCH"else res["var"].get("ES_Historical"))
                viol_info = res["violations"].get(method, {})
                detail_rows.append({
                    "Phương pháp"        : method,
                    "VaR (%)"            : round(var_val * 100, 4) if var_val else None,
                    "ES (%)"             : round(es_val  * 100, 4) if es_val  else None,
                    "Số vi phạm"         : viol_info.get("n_violations"),
                    "Tỷ lệ vi phạm (%)"  : viol_info.get("violation_rate (%)"),
                    "Tỷ lệ kỳ vọng (%)"  : viol_info.get("expected_rate (%)"),
                })
            pd.DataFrame(detail_rows).set_index("Phương pháp").to_csv(
                os.path.join(_TABLE_DIR, f"stress_{period}_detail.csv")
            )
        for period, result in stress_oos_results.items():
            result["violations"].to_csv(
                os.path.join(_TABLE_DIR, f"stress_oos_{period}_violations.csv")
            )
            result["kupiec"].to_csv(
                os.path.join(_TABLE_DIR, f"stress_oos_{period}_kupiec.csv")
            )
            result["christoffersen"].to_csv(
                os.path.join(_TABLE_DIR,f"stress_oos_{period}_christoffersen.csv",)
            )

        print(f"\n  [OK] Bảng kết quả đã lưu vào: outputs/tables/")

        # ── Bước 5: Vẽ biểu đồ ──────────────────────────────────────────────
        print(f"\n[3] Vẽ biểu đồ stress test...")
        plot_timeseries_by_period(stress_results, save=True)
        plot_return_distribution_by_period(stress_results, save=True)
        plot_var_comparison(comparison_df, save=True)
        plot_violation_rate(comparison_df, conf_level=conf_level, save=True)
        plot_oos_violation_rate(stress_oos_results,conf_level=conf_level,save=True)

    print("\n[DONE] Stress Test hoàn tất.")
    print("=" * 60)

    return {
        "stress_results": stress_results,
        "comparison_table": comparison_df,
        "stress_oos_results": stress_oos_results,
    }


if __name__ == "__main__":
    run_full_stress_test()
