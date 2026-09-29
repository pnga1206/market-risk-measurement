import pandas as pd
from pathlib import Path


URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"

START_DATE = "2016-01-01"
END_DATE = "2026-09-25"

OUTPUT_FILE = Path("data/main/sp500_raw.csv")


def download_sp500():

    print("Downloading S&P 500 from FRED...")
    print(f"Requested from: {START_DATE}")
    print(f"Requested to:   {END_DATE}")

    df = pd.read_csv(URL)

    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )

    df = df.rename(
        columns={
            "observation_date": "date",
            "SP500": "close",
        }
    )

    df = df[
        (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
    ].copy()

    df = df.sort_values("date")

    # Không tự ý fill dữ liệu thiếu
    df = df.dropna(subset=["close"])

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(df))
    print("First date:", df["date"].min().date())
    print("Last date:", df["date"].max().date())
    print("Missing close:", df["close"].isna().sum())


if __name__ == "__main__":
    download_sp500()