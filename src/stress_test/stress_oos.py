"""Tạo kết quả backtest ngoài mẫu cho các giai đoạn stress."""

import os

import pandas as pd

from src.backtesting.christoffersen_test import run_christoffersen_all
from src.backtesting.kupiec_test import run_kupiec_all
from src.backtesting.violations import compute_violations
from src.config import CONF_LEVEL, ROLLING_WINDOW
from src.garch.garch_model import calculate_garch_var_es_rolling
from src.var.var_models import run_rolling_var


_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _run_tests(violations_df: pd.DataFrame, conf_level: float) -> dict:
    """Tạo các kiểm định cho một bảng vi phạm."""
    if violations_df.empty:
        raise ValueError("Không có dự báo ngoài mẫu trong khoảng đánh giá.")

    return {
        "violations": violations_df,
        "kupiec": run_kupiec_all(
            violations_df,
            conf_level=conf_level,
        ),
        "christoffersen": run_christoffersen_all(
            violations_df,
            conf_level=conf_level,
        ),
    }


def _load_stress_2008_returns() -> pd.Series:
    """Đọc portfolio returns đã xây dựng cho giai đoạn stress 2008."""
    csv_path = os.path.join(
        _DATA_DIR,
        "stress_2008",
        "portfolio_returns.csv",
    )
    df = pd.read_csv(
        csv_path,
        parse_dates=["date"],
        index_col="date",
    )
    returns = df["portfolio_return"].sort_index().dropna()
    returns.index.name = "Date"
    return returns


def _run_2008_oos(
    conf_level: float,
    window: int,
) -> dict:
    """Dùng dữ liệu warm-up để đánh giá ngoài mẫu trong 2008–2009."""
    returns = _load_stress_2008_returns()
    eval_start = pd.Timestamp("2008-01-01")
    eval_end = pd.Timestamp("2009-12-31")

    # Rolling chỉ tạo dự báo khi đã có đủ window quan sát trước ngày dự báo.
    # Với dữ liệu hiện có, một số ngày đầu năm 2008 sẽ chưa có dự báo.

    # Chỉ dùng dữ liệu đến hết kỳ đánh giá.
    returns = returns.loc[:eval_end]

    var_df = run_rolling_var(
        returns,
        window=window,
        conf_level=conf_level,
    )
    garch_df = calculate_garch_var_es_rolling(
        returns,
        window=window,
        conf_level=conf_level,
    )

    forecasts = pd.concat(
        [
            var_df,
            garch_df[["VaR_GARCH"]],
        ],
        axis=1,
    )

    eval_returns = returns.loc[eval_start:eval_end]
    eval_forecasts = forecasts.loc[eval_start:eval_end]

    violations_df = compute_violations(
        eval_returns,
        eval_forecasts,
    )
    return _run_tests(violations_df, conf_level)


def _filter_backtest_year(
    backtest_results: dict,
    year: int,
    conf_level: float,
) -> dict:
    """Lọc dự báo rolling đã có trong backtest chính theo năm."""
    violations_df = backtest_results["violations"]
    dates = pd.to_datetime(violations_df.index)
    year_violations = violations_df.loc[dates.year == year].copy()

    return _run_tests(year_violations, conf_level)


def run_stress_oos(
    backtest_results: dict,
    conf_level: float = CONF_LEVEL,
    window: int = ROLLING_WINDOW,
) -> dict:
    """Trả kết quả OOS cho 2008–2009, 2020 và 2022."""
    return {
        "2008-2009": _run_2008_oos(conf_level, window),
        "2020": _filter_backtest_year(
            backtest_results,
            2020,
            conf_level,
        ),
        "2022": _filter_backtest_year(
            backtest_results,
            2022,
            conf_level,
        ),
    }