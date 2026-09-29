import pandas as pd
from pathlib import Path


# =========================
# 1. Đường dẫn dữ liệu
# =========================
VN_FILE = Path("data/main/vnindex_raw.csv")
SP_FILE = Path("data/main/sp500_raw.csv")
FX_FILE = Path("data/main/fx_raw.csv")

OUTPUT_FILE = Path("data/main/portfolio_aligned.csv")


# =========================
# 2. Đọc dữ liệu
# =========================
vn = pd.read_csv(VN_FILE, parse_dates=["date"])
sp = pd.read_csv(SP_FILE, parse_dates=["date"])
fx = pd.read_csv(FX_FILE, parse_dates=["date"])


# =========================
# 3. Chỉ giữ các biến cần thiết
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
# 5. S&P 500 quy đổi sang VND
# =========================
portfolio["sp500_close_vnd"] = (
    portfolio["sp500_close_usd"] * portfolio["usd_vnd"]
)


# =========================
# 6. Sắp xếp theo thời gian
# =========================
portfolio = portfolio.sort_values("date").reset_index(drop=True)


# =========================
# 7. Lưu dữ liệu
# =========================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

portfolio.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================
# 8. Kiểm tra kết quả
# =========================
print("Saved:", OUTPUT_FILE)
print("Rows:", len(portfolio))
print("First date:", portfolio["date"].min().date())
print("Last date:", portfolio["date"].max().date())
print("Missing values:")
print(portfolio.isna().sum())

print("\nSample:")
print(portfolio.head())