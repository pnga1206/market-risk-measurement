import yfinance as yf
import pandas as pd
from pathlib import Path


TICKER = "^GSPC"

START_DATE = "2007-01-01"
END_DATE = "2010-01-01"

OUTPUT_FILE = Path("data/stress_2008/sp500_raw.csv")


def download_sp500_stress():

    print("Downloading S&P 500 stress data from Yahoo Finance...")
    print(f"Requested from: {START_DATE}")
    print(f"Requested to:   2009-12-31")

    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            "Không tải được dữ liệu S&P 500 stress từ Yahoo Finance."
        )

    # yfinance phiên bản mới có thể trả MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"][TICKER]
    else:
        close = df["Close"]

    result = pd.DataFrame({
        "date": pd.to_datetime(close.index),
        "close": close.values,
    })

    result = result.dropna(subset=["close"])
    result = result.sort_values("date")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(result))
    print("First date:", result["date"].min().date())
    print("Last date:", result["date"].max().date())
    print("Missing close:", result["close"].isna().sum())


if __name__ == "__main__":
    download_sp500_stress()