import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# 1. Đường dẫn dữ liệu
# =========================
VN_FILE = Path("data/stress_2008/vnindex_raw.csv")
SP_FILE = Path("data/stress_2008/sp500_raw.csv")
FX_FILE = Path("data/stress_2008/fx_raw.csv")

OUTPUT_FILE = Path("data/stress_2008/portfolio_returns.csv")


# =========================
# 2. Đọc dữ liệu
# =========================
vn = pd.read_csv(
    VN_FILE,
    parse_dates=["date"]
)

sp = pd.read_csv(
    SP_FILE,
    parse_dates=["date"]
)

fx = pd.read_csv(
    FX_FILE,
    parse_dates=["date"]
)


# =========================
# 3. Giữ các biến cần thiết
# =========================
vn = vn[["date", "close"]].rename(columns={
    "close": "vnindex_close"
})

sp = sp[["date", "close"]].rename(columns={
    "close": "sp500_close_usd"
})

fx = fx[["date", "close"]].rename(columns={
    "close": "usd_vnd"
})


# =========================
# 4. Inner Join theo ngày
# =========================
portfolio = vn.merge(
    sp,
    on="date",
    how="inner"
)

portfolio = portfolio.merge(
    fx,
    on="date",
    how="inner"
)


# =========================
# 5. Quy đổi S&P 500 sang VND
# =========================
portfolio["sp500_close_vnd"] = (
    portfolio["sp500_close_usd"] * portfolio["usd_vnd"]
)


# =========================
# 6. Sắp xếp theo ngày
# =========================
portfolio = portfolio.sort_values(
    "date"
).reset_index(drop=True)


# =========================
# 7. Tính log return
# =========================
portfolio["vnindex_return"] = np.log(
    portfolio["vnindex_close"]
    / portfolio["vnindex_close"].shift(1)
)

portfolio["sp500_vnd_return"] = np.log(
    portfolio["sp500_close_vnd"]
    / portfolio["sp500_close_vnd"].shift(1)
)


# =========================
# 8. Portfolio return 50/50
# =========================
portfolio["portfolio_return"] = (
    0.5 * portfolio["vnindex_return"]
    + 0.5 * portfolio["sp500_vnd_return"]
)


# =========================
# 9. Loại dòng đầu tiên
# =========================
portfolio = portfolio.dropna(
    subset=[
        "vnindex_return",
        "sp500_vnd_return",
        "portfolio_return"
    ]
).reset_index(drop=True)


# =========================
# 10. Chỉ giữ biến cần thiết
# =========================
portfolio = portfolio[
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
# 11. Lưu kết quả
# =========================
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

portfolio.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# 12. Kiểm tra
# =========================
print("Saved:", OUTPUT_FILE)
print("Rows:", len(portfolio))
print("First date:", portfolio["date"].min().date())
print("Last date:", portfolio["date"].max().date())

print("\nMissing values:")
print(portfolio.isna().sum())

print("\nReturn summary:")
print(
    portfolio[
        [
            "vnindex_return",
            "sp500_vnd_return",
            "portfolio_return"
        ]
    ].describe()
)

print("\nSample:")
print(portfolio.head())