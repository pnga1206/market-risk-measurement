"""
plot_stress.py – Vẽ biểu đồ Stress Test cho 3 giai đoạn khủng hoảng
=====================================================================
Cung cấp các hàm visualize kết quả stress test để phục vụ báo cáo.

Output:
    - Biểu đồ phân phối lợi suất + ngưỡng VaR cho từng giai đoạn
    - Biểu đồ so sánh VaR / ES qua 3 giai đoạn
    - Biểu đồ chuỗi thời gian lợi suất của các giai đoạn căng thẳng
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy.stats import norm

warnings.filterwarnings("ignore")

_FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "figures")
_PERIOD_LABELS = {"2008": "2008 – Khủng hoảng tài chính", "2020": "2020 – Đại dịch COVID-19", "2022": "2022 – Khủng hoảng lạm phát"}
_COLORS        = {"2008": "#E53935", "2020": "#FB8C00", "2022": "#8E24AA"}


def _ensure_fig_dir():
    os.makedirs(_FIGURE_DIR, exist_ok=True)


def plot_return_distribution_by_period(stress_results: dict, save: bool = True):
    """
    Vẽ phân phối log-return và đánh dấu ngưỡng VaR cho từng giai đoạn.

    Parameters
    ----------
    stress_results : dict
        Đầu ra từ stress_test.run_all_stress_tests().
    save : bool
        Lưu file PNG nếu True.
    """
    _ensure_fig_dir()
    n_periods = sum(1 for v in stress_results.values() if v is not None)
    fig, axes = plt.subplots(1, n_periods, figsize=(6 * n_periods, 5), sharey=False)
    if n_periods == 1:
        axes = [axes]

    ax_iter = iter(axes)
    for period, res in stress_results.items():
        if res is None:
            continue
        ax     = next(ax_iter)
        color  = _COLORS.get(period, "#1565C0")
        rets   = res["returns"]
        h_var  = res["var"]["VaR_Historical"]
        p_var  = res["var"]["VaR_Parametric"]
        g_var  = res["garch"]["VaR_GARCH"]

        ax.hist(rets, bins=40, color=color, alpha=0.55, edgecolor="white", density=True, label="Log-return")

        # Đường phân phối chuẩn
        xs = np.linspace(rets.min(), rets.max(), 300)
        ax.plot(xs, norm.pdf(xs, rets.mean(), rets.std()), "k--", linewidth=1.3, label="Normal fit")

        # Đường ngưỡng VaR
        ax.axvline(-h_var, color="#1E88E5", linewidth=1.8, linestyle="-",  label=f"HVar={h_var*100:.2f}%")
        ax.axvline(-p_var, color="#43A047", linewidth=1.8, linestyle="--", label=f"PVar={p_var*100:.2f}%")
        ax.axvline(-g_var, color="#E53935", linewidth=1.8, linestyle=":", label=f"GVar={g_var*100:.2f}%")

        ax.set_title(_PERIOD_LABELS.get(period, period), fontsize=11, fontweight="bold")
        ax.set_xlabel("Log-return")
        ax.set_ylabel("Mật độ")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Phân phối lợi suất trong các giai đoạn khủng hoảng", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(_FIGURE_DIR, "stress_return_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Lưu: {path}")
    plt.close(fig)

def plot_var_comparison(comparison_df: pd.DataFrame, save: bool = True):
    """
    Biểu đồ cột so sánh VaR (%) và ES (%) qua 3 giai đoạn khủng hoảng.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Đầu ra từ stress_test.build_comparison_table().
    save : bool
        Lưu file PNG nếu True.
    """
    _ensure_fig_dir()
    cols_var  = ["VaR_Historical (%)", "VaR_Parametric (%)", "VaR_MonteCarlo (%)", "VaR_GARCH (%)"]
    cols_es   = ["ES_Historical (%)", "ES_GARCH (%)"]

    df_var = comparison_df[[c for c in cols_var if c in comparison_df.columns]]
    df_es  = comparison_df[[c for c in cols_es  if c in comparison_df.columns]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # VaR
    x    = np.arange(len(df_var))
    w    = 0.2
    bars = df_var.columns
    for i, col in enumerate(bars):
        ax1.bar(x + i * w, df_var[col], width=w, label=col.replace(" (%)", ""), alpha=0.85)
    ax1.set_xticks(x + w * (len(bars) - 1) / 2)
    ax1.set_xticklabels(df_var.index)
    ax1.set_title("So sánh VaR (%) – 3 giai đoạn khủng hoảng", fontsize=11, fontweight="bold")
    ax1.set_ylabel("VaR (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=2))

    # ES
    x2   = np.arange(len(df_es))
    bars2 = df_es.columns
    w2    = 0.3
    for i, col in enumerate(bars2):
        ax2.bar(x2 + i * w2, df_es[col], width=w2, label=col.replace(" (%)", ""), alpha=0.85)
    ax2.set_xticks(x2 + w2 * (len(bars2) - 1) / 2)
    ax2.set_xticklabels(df_es.index)
    ax2.set_title("So sánh ES (%) – 3 giai đoạn khủng hoảng", fontsize=11, fontweight="bold")
    ax2.set_ylabel("ES (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=2))

    plt.suptitle("Stress Test – VaR và ES qua các giai đoạn khủng hoảng", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(_FIGURE_DIR, "stress_var_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Lưu: {path}")
    plt.close(fig)


def plot_timeseries_by_period(stress_results: dict, save: bool = True):
    """
    Vẽ chuỗi thời gian lợi suất trong từng giai đoạn và đường ngưỡng VaR.

    Parameters
    ----------
    stress_results : dict
        Đầu ra từ stress_test.run_all_stress_tests().
    save : bool
        Lưu file PNG nếu True.
    """
    _ensure_fig_dir()
    valid = {k: v for k, v in stress_results.items() if v is not None}
    n     = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (period, res) in zip(axes, valid.items()):
        color = _COLORS.get(period, "#1565C0")
        rets  = res["returns"]
        h_var = res["var"]["VaR_Historical"]
        g_var_series = res["garch"]["VaR_GARCH_series"].reindex(rets.index)

        ax.fill_between(rets.index, rets, 0, where=(rets < 0), color="#FFCDD2", alpha=0.5)
        ax.plot(rets.index, rets, color=color, linewidth=0.8, label="Log-return")
        ax.axhline(-h_var, color="#1E88E5", linewidth=1.5, linestyle="--", label=f"−VaR Historical ({h_var*100:.2f}%)")
        ax.plot(g_var_series.index,-g_var_series,color="#E53935",linewidth=1.5,linestyle=":",label="−VaR GARCH theo ngày",)
        ax.axhline(0,      color="black",  linewidth=0.5, linestyle="-")

        ax.set_title(_PERIOD_LABELS.get(period, period), fontsize=11, fontweight="bold")
        ax.set_ylabel("Log-return")
        ax.legend(fontsize=8, loc="lower right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Lợi suất danh mục trong các giai đoạn khủng hoảng", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(_FIGURE_DIR, "stress_timeseries.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Lưu: {path}")
    plt.close(fig)


def plot_violation_rate(comparison_df: pd.DataFrame, conf_level: float = 0.975, save: bool = True):
    """
    Biểu đồ tỷ lệ vi phạm VaR trong giai đoạn khủng hoảng so với mức kỳ vọng.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Đầu ra từ stress_test.build_comparison_table().
    conf_level : float
        Mức độ tin cậy (để vẽ đường kỳ vọng).
    save : bool
        Lưu file PNG nếu True.
    """
    _ensure_fig_dir()
    viol_cols = [c for c in comparison_df.columns if "Vi phạm" in c]
    if not viol_cols:
        print("[CẢNH BÁO] Không tìm thấy cột vi phạm trong comparison_df.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x   = np.arange(len(comparison_df))
    w   = 0.25

    for i, col in enumerate(viol_cols):
        label = col.replace("Vi phạm ", "").replace(" (%)", "")
        ax.bar(x + i * w, comparison_df[col], width=w, alpha=0.8, label=label)

    # Đường kỳ vọng (2.5%)
    ax.axhline((1 - conf_level) * 100, color="red", linewidth=2, linestyle="--",
               label=f"Kỳ vọng ({(1-conf_level)*100:.1f}%)")

    ax.set_xticks(x + w * (len(viol_cols) - 1) / 2)
    ax.set_xticklabels(comparison_df.index)
    ax.set_title("Tỷ lệ vi phạm VaR trong các giai đoạn khủng hoảng", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tỷ lệ vi phạm (%)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    plt.tight_layout()

    if save:
        path = os.path.join(_FIGURE_DIR, "stress_violation_rate.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Lưu: {path}")
    plt.close(fig)

def plot_oos_violation_rate(
    stress_oos_results: dict,
    conf_level: float = 0.975,
    save: bool = True,
):
    """Vẽ tỷ lệ vi phạm VaR ngoài mẫu theo giai đoạn stress."""
    _ensure_fig_dir()

    periods = [
        period
        for period, result in stress_oos_results.items()
        if result["violations"] is not None
        and not result["violations"].empty
    ]
    if not periods:
        print("[CẢNH BÁO] Không có kết quả OOS để vẽ.")
        return

    methods = sorted({
        col.removesuffix("_violation")
        for period in periods
        for col in stress_oos_results[period]["violations"].columns
        if col.endswith("_violation")
    })
    if not methods:
        print("[CẢNH BÁO] Không tìm thấy cột vi phạm OOS.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(periods))
    width = 0.8 / len(methods)

    for method_index, method in enumerate(methods):
        rates = []
        for period in periods:
            violations = stress_oos_results[period]["violations"]
            column = f"{method}_violation"

            if column in violations.columns:
                rates.append(violations[column].mean() * 100)
            else:
                rates.append(np.nan)

        offsets = x - 0.4 + width / 2 + method_index * width
        ax.bar(offsets, rates, width=width, label=method)

    expected_rate = (1 - conf_level) * 100
    ax.axhline(
        expected_rate,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Mức kỳ vọng ({expected_rate:.1f}%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_title("Tỷ lệ vi phạm VaR ngoài mẫu")
    ax.set_ylabel("Tỷ lệ vi phạm (%)")
    ax.set_xlabel("Giai đoạn stress")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    plt.tight_layout()

    if save:
        path = os.path.join(_FIGURE_DIR, "stress_oos_violation_rate.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Lưu: {path}")

    plt.close(fig)
