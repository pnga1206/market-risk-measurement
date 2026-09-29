"""
run_backtest.py – Pipeline Backtesting hoàn chỉnh (Coder 3)
=============================================================
Orchestrator chạy toàn bộ quy trình backtesting VaR ngoài mẫu:
    1. Load dữ liệu portfolio return (Coder 1)
    2. Chạy rolling VaR + GARCH (Coder 2)
    3. Tính violations
    4. Kupiec Test
    5. Christoffersen Test
    6. Xuất bảng kết quả + biểu đồ

Usage (chạy từ thư mục gốc dự án):
    python -m src.backtesting.run_backtest
    
    # Hoặc gọi từ Python:
    from src.backtesting.run_backtest import run_full_backtest
    results = run_full_backtest()
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Coder 2 – Mô hình VaR & GARCH
from src.var.var_models   import run_rolling_var
from src.garch.garch_model import calculate_garch_var_es_rolling

# Coder 3 – Backtesting
from src.backtesting.violations        import compute_violations, violation_summary
from src.backtesting.kupiec_test       import run_kupiec_all
from src.backtesting.christoffersen_test import run_christoffersen_all

# Cấu hình chung
from src.config import CONF_LEVEL, ROLLING_WINDOW

warnings.filterwarnings("ignore")

# ── Đường dẫn mặc định ────────────────────────────────────────────────────────
_BASE_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "main")
_OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
_TABLE_DIR   = os.path.join(_OUTPUT_DIR, "tables")
_FIGURE_DIR  = os.path.join(_OUTPUT_DIR, "figures")


def _ensure_dirs():
    """Tạo thư mục đầu ra nếu chưa tồn tại."""
    os.makedirs(_TABLE_DIR, exist_ok=True)
    os.makedirs(_FIGURE_DIR, exist_ok=True)


def load_portfolio_returns(csv_path: str | None = None) -> pd.Series:
    """
    Đọc file portfolio_returns.csv từ Coder 1.

    Input:
        csv_path : Đường dẫn tới file CSV (mặc định data/main/portfolio_returns.csv)

    Output:
        pd.Series – Chuỗi log-return của danh mục, index là pd.DatetimeIndex.
    """
    if csv_path is None:
        csv_path = os.path.join(_BASE_DIR, "portfolio_returns.csv")

    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.index.name = "Date"
    return df["portfolio_return"].sort_index()


def plot_violations(violations_df: pd.DataFrame, save_path: str | None = None):
    """
    Vẽ biểu đồ lợi suất thực tế so với VaR dự báo và đánh dấu vi phạm.

    Parameters
    ----------
    violations_df : pd.DataFrame
        Đầu ra từ compute_violations().
    save_path : str | None
        Nếu được cung cấp, lưu biểu đồ vào file.
    """
    var_cols = [c for c in violations_df.columns if c.startswith("VaR_")]
    n_plots  = len(var_cols)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for ax, col, color in zip(axes, var_cols, colors):
        method = col.replace("VaR_", "")
        hit_col = method + "_violation"

        actual = violations_df["actual_return"]
        var_neg = -violations_df[col]   # Đường VaR âm (ngưỡng lỗ)

        ax.fill_between(
            violations_df.index, actual, 0,
            where=(actual < 0), color="#FFCDD2", alpha=0.4, label="Lỗ thực tế"
        )
        ax.plot(violations_df.index, actual,  color="#424242", linewidth=0.6, alpha=0.7, label="Return thực tế")
        ax.plot(violations_df.index, var_neg, color=color,    linewidth=1.4, label=f"Ngưỡng -{col}")

        # Đánh dấu vi phạm
        if hit_col in violations_df.columns:
            viol_dates = violations_df.index[violations_df[hit_col] == 1]
            viol_vals  = actual.loc[viol_dates]
            ax.scatter(viol_dates, viol_vals, color="red", s=18, zorder=5, label=f"Vi phạm ({len(viol_dates)})")

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"Backtesting VaR – {method}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Log-return")
        ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)

    plt.xlabel("Ngày", fontsize=10)
    plt.suptitle("Backtesting VaR – So sánh dự báo với thực tế", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Biểu đồ violations lưu tại: {save_path}")
    plt.close(fig)


def run_full_backtest(
    portfolio_csv: str | None = None,
    conf_level: float = CONF_LEVEL,
    window: int = ROLLING_WINDOW,
    save_outputs: bool = True,
) -> dict:
    """
    Chạy toàn bộ pipeline backtesting VaR ngoài mẫu.

    Parameters
    ----------
    portfolio_csv : str | None
        Đường dẫn tới portfolio_returns.csv (Coder 1).
    conf_level : float
        Mức độ tin cậy (mặc định 0.975).
    window : int
        Cửa sổ trượt rolling (mặc định 250 ngày).
    save_outputs : bool
        Có lưu bảng CSV và biểu đồ ra thư mục outputs/ không.

    Returns
    -------
    dict với các khóa:
        - "var_forecasts"   : pd.DataFrame  (VaR rolling từ Coder 2)
        - "garch_forecasts" : pd.DataFrame  (GARCH-VaR từ Coder 2)
        - "violations"      : pd.DataFrame  (vi phạm)
        - "summary"         : pd.DataFrame  (tóm tắt violations)
        - "kupiec"          : pd.DataFrame  (kết quả Kupiec Test)
        - "christoffersen"  : pd.DataFrame  (kết quả Christoffersen Test)
    """
    _ensure_dirs()
    print("=" * 60)
    print("BACKTESTING VaR – CODER 3")
    print("=" * 60)

    # ── Bước 1: Load dữ liệu ─────────────────────────────────────────────────
    print(f"\n[1] Đọc dữ liệu portfolio_returns...")
    port_returns = load_portfolio_returns(portfolio_csv)
    print(f"    Số ngày dữ liệu: {len(port_returns):,}  |  {port_returns.index[0].date()} → {port_returns.index[-1].date()}")

    # ── Bước 2: Chạy mô hình VaR & GARCH (Coder 2) ──────────────────────────
    print(f"\n[2] Tính rolling VaR (window={window} ngày)...")
    var_df   = run_rolling_var(port_returns, window=window, conf_level=conf_level)

    print(f"[3] Tính GARCH-VaR và ES 97.5%...")
    garch_df = calculate_garch_var_es_rolling(port_returns, window=window, conf_level=conf_level)

    # Gộp tất cả VaR dự báo
    all_var_df = pd.concat([var_df, garch_df[["VaR_GARCH", "ES_97_5"]]], axis=1)
    print(f"    Số ngày VaR dự báo: {len(all_var_df):,}  |  {all_var_df.index[0].date()} → {all_var_df.index[-1].date()}")

    # ── Bước 3: Tính violations ───────────────────────────────────────────────
    print(f"\n[4] Xác định vi phạm VaR...")
    violations_df = compute_violations(port_returns, all_var_df)
    summary_df    = violation_summary(violations_df)
    print(summary_df.to_string())

    # ── Bước 4: Kupiec Test ───────────────────────────────────────────────────
    print(f"\n[5] Kupiec POF Test (α={0.05})...")
    kupiec_df = run_kupiec_all(violations_df, conf_level=conf_level)
    print(kupiec_df[["n_violations", "violation_rate", "expected_rate", "LR_statistic", "p_value", "reject_H0"]].to_string())

    # ── Bước 5: Christoffersen Test ───────────────────────────────────────────
    print(f"\n[6] Christoffersen Conditional Coverage Test (α={0.05})...")
    chris_df = run_christoffersen_all(violations_df, conf_level=conf_level)
    print(chris_df[["LR_uc", "p_value_uc", "LR_ind", "p_value_ind", "LR_cc", "p_value_cc",
                     "reject_uc", "reject_ind", "reject_cc"]].to_string())

    # ── Bước 6: Lưu kết quả ──────────────────────────────────────────────────
    if save_outputs:
        violations_df.to_csv(os.path.join(_TABLE_DIR, "backtest_violations.csv"))
        summary_df.to_csv(os.path.join(_TABLE_DIR,    "backtest_summary.csv"))
        kupiec_df.to_csv(os.path.join(_TABLE_DIR,     "kupiec_results.csv"))
        chris_df.to_csv(os.path.join(_TABLE_DIR,      "christoffersen_results.csv"))
        print(f"\n  [OK] Các bảng kết quả đã lưu vào: outputs/tables/")

        plot_violations(
            violations_df,
            save_path=os.path.join(_FIGURE_DIR, "backtest_violations.png")
        )

    print("\n[DONE] Backtesting hoàn tất.")
    print("=" * 60)

    return {
        "var_forecasts"   : var_df,
        "garch_forecasts" : garch_df,
        "violations"      : violations_df,
        "summary"         : summary_df,
        "kupiec"          : kupiec_df,
        "christoffersen"  : chris_df,
    }


if __name__ == "__main__":
    run_full_backtest()
