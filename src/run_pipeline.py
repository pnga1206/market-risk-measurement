"""
Chạy quy trình phân tích rủi ro thị trường hoàn chỉnh.

Quy trình gồm:
    1. Backtest các mô hình VaR.
    2. Stress test các giai đoạn 2008, 2020 và 2022.

Chạy từ thư mục gốc dự án:
    python -m src.run_pipeline
"""

from src.backtesting.run_backtest import run_full_backtest
from src.stress_test.run_stress_test import run_full_stress_test


def run_full_pipeline(save_outputs: bool = True) -> dict:
    """Chạy backtest và stress test, trả về kết quả của cả hai."""
    backtest_results = run_full_backtest(save_outputs=save_outputs)
    stress_results = run_full_stress_test(save_outputs=save_outputs, backtest_results=backtest_results)

    return {
        "backtest": backtest_results,
        "stress_test": stress_results,
    }


if __name__ == "__main__":
    run_full_pipeline()