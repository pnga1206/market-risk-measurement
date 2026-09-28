import yfinance as yf
import pandas as pd
from pathlib import Path


TICKER = "VND=X"

START_DATE = "2016-01-01"
END_DATE = "2026-09-26"

OUTPUT_FILE = Path("data/main/fx_raw.csv")


def download_fx():

    print("Downloading USD/VND from Yahoo Finance...")
    print(f"Requested from: {START_DATE}")
    print(f"Requested to:   2026-09-25")

    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            "Không tải được dữ liệu USD/VND từ Yahoo Finance."
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
    download_fx()