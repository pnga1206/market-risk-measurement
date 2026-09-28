import pandas as pd
from pathlib import Path
import numpy as np


# =========================
# 1. Đường dẫn dữ liệu
# =========================
INPUT_FILE = Path("data/main/portfolio_aligned.csv")
OUTPUT_FILE = Path("data/main/portfolio_returns.csv")


# =========================
# 2. Đọc dữ liệu
# =========================
df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)


# =========================
# 3. Tính log return
# =========================
df["vnindex_return"] = np.log(
    df["vnindex_close"] / df["vnindex_close"].shift(1)
)

df["sp500_vnd_return"] = np.log(
    df["sp500_close_vnd"] / df["sp500_close_vnd"].shift(1)
)


# =========================
# 4. Tính lợi suất danh mục 50/50
# =========================
df["portfolio_return"] = (
    0.5 * df["vnindex_return"]
    + 0.5 * df["sp500_vnd_return"]
)


# =========================
# 5. Loại dòng đầu tiên
# =========================
df = df.dropna(
    subset=[
        "vnindex_return",
        "sp500_vnd_return",
        "portfolio_return"
    ]
).reset_index(drop=True)


# =========================
# 6. Chỉ giữ biến cần thiết
# =========================
df = df[
    [
        "date",
        "vnindex_close",
        "sp500_close_usd",
        "usd_vnd",
        "sp500_close_vnd",
        "vnindex_return",
        "sp500_vnd_return",
        "portfolio_return"
    ]
]


# =========================
# 7. Lưu kết quả
# =========================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# 8. Kiểm tra
# =========================
print("Saved:", OUTPUT_FILE)
print("Rows:", len(df))
print("First date:", df["date"].min().date())
print("Last date:", df["date"].max().date())

print("\nMissing values:")
print(df.isna().sum())

print("\nReturn summary:")
print(
    df[
        [
            "vnindex_return",
            "sp500_vnd_return",
            "portfolio_return"
        ]
    ].describe()
)

print("\nSample:")
print(df.head())