import html as _html
import re
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ============================================================
# 1. CẤU HÌNH & HẰNG SỐ
# ============================================================
st.set_page_config(
    page_title="Đo lường rủi ro thị trường",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

C = dict(
    ink="#0F2A43", text="#1E293B", muted="#64748B", line="#DCE7F3",
    blue="#2563EB", sky="#0EA5E9", teal="#14B8A6", amber="#F59E0B",
    coral="#EF4444", green="#16A34A", violet="#7C3AED",
)
MODEL_COLORS = {
    "Historical": "#64748B", "Parametric": C["blue"], "MonteCarlo": C["teal"], "Monte Carlo": C["teal"],
    "GARCH": C["amber"], "ES_97_5": C["coral"], "Actual": "#94A3B8",
}
DATA_FREEZE_DATE = "25/09/2026"
REPO_URL = "https://github.com/pnga1206/market-risk-measurement"
CONF = 0.975
P0_PCT = (1 - CONF) * 100   # tỷ lệ vi phạm kỳ vọng (%)
WINDOW = 250                # cửa sổ trượt (phiên)
MC_SIMULATIONS = 10_000
RANDOM_SEED = 42
W_VN, W_US = 50, 50
LOG_RETURNS = True
SCALE = 1.15                # hệ số phóng chữ toàn trang
PLOT_SCALE = SCALE
OOS_KEYS = {"2008": "2008-2009", "2020": "2020", "2022": "2022"}
VAR_COLS = [("VaR_Historical", "Historical"), ("VaR_Parametric", "Parametric"),
            ("VaR_MonteCarlo", "Monte Carlo"), ("VaR_GARCH", "GARCH")]

# Tỷ lệ ES/VaR lý thuyết của phân phối chuẩn ở mức CONF (≈ 1,19)
_Z = NormalDist().inv_cdf(CONF)
ES_VAR_NORMAL = NormalDist().pdf(_Z) / (1 - CONF) / _Z

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html, body, .stApp, p, li, label, h1, h2, h3, h4, td, th, input, button { font-family: 'Be Vietnam Pro', 'Segoe UI', Arial, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-icons"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: linear-gradient(180deg, #EAF4FF 0%, #F8FBFF 320px, #FFFFFF 100%); color: #1E293B; }
[data-testid="stSidebar"], [data-testid="stHeader"] { display: none; }
.block-container { max-width: 1600px; padding: 1.2rem 2rem 3rem 2rem; }
.hero { background: linear-gradient(120deg, #1D4ED8 0%, #0EA5E9 55%, #14B8A6 100%);
        border-radius: 24px; padding: 28px 34px; color: #fff; margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, .22); }
.hero h1 { color: #fff; font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; padding: 0; line-height: 1.25; }
.hero p { color: #EAF6FF; font-size: 1.05rem; margin: 0; }
.sec-title { font-size: 1.45rem; font-weight: 800; color: #0F2A43; margin: 1.6rem 0 .2rem 0; }
.sec-desc { color: #64748B; font-size: 1rem; margin: 0 0 .9rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #fff; padding: 8px; border-radius: 999px;
        border: 1px solid #DCE7F3; box-shadow: 0 4px 14px rgba(15, 42, 67, .06); flex-wrap: wrap; margin-bottom: 8px; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 22px; border-radius: 999px; background: transparent; }
.stTabs [data-baseweb="tab"] p { font-size: 1rem; font-weight: 700; color: #475569; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, #2563EB, #14B8A6); }
.stTabs [aria-selected="true"] p { color: #fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.card { background: #fff; border: 1px solid #DCE7F3; border-left: 6px solid var(--accent, #2563EB);
        border-radius: 18px; padding: 16px 20px; box-shadow: 0 4px 14px rgba(15, 42, 67, .05);
        display: flex; flex-direction: column; justify-content: flex-start; overflow: hidden; margin-bottom: 14px; }
.card.s { height: 10rem; } .card.m { height: 13.5rem; } .card.l { height: 17rem; } .card.xl { height: 22rem; }
.card .t { font-size: .95rem; font-weight: 700; color: #64748B; }
.card .v { font-size: 1.9rem; font-weight: 800; color: #0F2A43; margin: 4px 0 2px 0; line-height: 1.2; }
.card .h { font-size: 1.15rem; font-weight: 800; color: #0F2A43; margin: 4px 0 6px 0; }
.card .s2, .card .b { font-size: .95rem; color: #334155; line-height: 1.5; }
.card ul { margin: 6px 0 0 0; padding-left: 18px; font-size: .95rem; color: #334155; line-height: 1.5; }
.howto { background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 14px; padding: 12px 18px;
         color: #0C4A6E; font-size: .97rem; margin: 4px 0 14px 0; }
.warn { background: #FFF7E6; border: 1px solid #FDE7B0; border-radius: 14px; padding: 12px 18px;
        color: #92400E; font-size: .97rem; margin: 4px 0 14px 0; }
[data-testid="stDataFrame"] { border: 1px solid #DCE7F3; border-radius: 14px; overflow: hidden; }
.stExpander { border-radius: 14px !important; border: 1px solid #DCE7F3 !important; background: #fff; }
.footer { border-top: 1px solid #DCE7F3; margin-top: 40px; padding-top: 16px; text-align: center;
          color: #64748B; font-size: .95rem; }
.pt { width: 100%; border-collapse: separate; border-spacing: 0; background: #fff; border: 1px solid #DCE7F3;
      border-radius: 16px; overflow: hidden; font-size: 1.05rem; margin-bottom: 14px; }
.pt th { background: #EAF4FF; color: #0F2A43; font-weight: 700; text-align: left; padding: .7rem .9rem; border-bottom: 1px solid #DCE7F3; }
.pt td { padding: .65rem .9rem; border-bottom: 1px solid #EEF3F9; color: #1E293B; }
.pt tr:last-child td { border-bottom: none; }
.pill { display: inline-block; padding: .12rem .75rem; border-radius: 999px; font-weight: 700; }
.pill.good { background: #DCFCE7; color: #15803D; } .pill.bad { background: #FEE2E2; color: #B91C1C; }
.pill.warn { background: #FFF3D6; color: #B45309; }
.take { background: linear-gradient(90deg, #FFFFFF, #ECFDF8); border: 1px solid #CDEFE9; border-left: 6px solid #14B8A6;
        border-radius: 16px; padding: 14px 22px; font-size: 1.2rem; font-weight: 600; color: #0F2A43; margin: 6px 0 14px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"<style>html {{ font-size: {16 * SCALE:.1f}px; }}</style>", unsafe_allow_html=True)

# st.fragment chỉ có từ Streamlit 1.37; nếu bản cũ thì bỏ qua (chạy như bình thường)
_fragment = getattr(st, "fragment", lambda f: f)


# ============================================================
# 2. HÀM HỖ TRỢ GIAO DIỆN
# ============================================================
_VN_SWAP = str.maketrans(",.", ".,")


def vn(s):
    """Đổi sang định dạng số Việt Nam: 1,234.56 -> 1.234,56."""
    return str(s).translate(_VN_SWAP)


def d2(x, d=2):
    """Số thực -> chuỗi kiểu Việt Nam với d chữ số thập phân."""
    return vn(f"{x:.{d}f}")


def card(title, value, sub="", color=C["blue"], size="s"):
    st.markdown(
        f'<div class="card {size}" style="--accent:{color}"><div class="t">{title}</div>'
        f'<div class="v">{value}</div><div class="s2">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def text_card(title, head, body, color=C["blue"], size="l"):
    st.markdown(
        f'<div class="card {size}" style="--accent:{color}"><div class="t">{title}</div>'
        f'<div class="h">{head}</div><div class="b">{body}</div></div>',
        unsafe_allow_html=True,
    )


def list_card(head, items, color=C["blue"], size="l"):
    li = "".join(f"<li>{i}</li>" for i in items)
    st.markdown(
        f'<div class="card {size}" style="--accent:{color}"><div class="h">{head}</div><ul>{li}</ul></div>',
        unsafe_allow_html=True,
    )


def section(title, desc=""):
    st.markdown(f'<div class="sec-title">{title}</div><div class="sec-desc">{desc}</div>', unsafe_allow_html=True)


def howto(text):
    st.markdown(f'<div class="howto"><b>Cách đọc:</b> {text}</div>', unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="warn">⚠ {text}</div>', unsafe_allow_html=True)


def takeaway(text):
    st.markdown(f'<div class="take"><b>Thông điệp chính:</b> {text}</div>', unsafe_allow_html=True)


def missing_file(filename):
    warn(f"Chưa có file <b>{filename}</b> trong <code>outputs/tables/</code>. Hãy chạy pipeline tương ứng trước.")


def html_table(df, tones=None):
    """Bảng HTML cỡ chữ lớn; tones = {tên cột: [good/bad/warn/None theo từng dòng]}."""
    df = df.reset_index(drop=True)
    tones = tones or {}
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for i in range(len(df)):
        tds = []
        for c in df.columns:
            v = _html.escape(str(df.iloc[i][c]))
            t = tones[c][i] if c in tones else None
            tds.append(f'<td><span class="pill {t}">{v}</span></td>' if t else f"<td>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    st.markdown(f'<div style="overflow-x:auto"><table class="pt"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


def cum_index(r):
    return np.exp(r.cumsum()) * 100 if LOG_RETURNS else (1 + r).cumprod() * 100


def pct(x, d=2):
    """Số thập phân -> chuỗi %, ví dụ 0.0173 -> 1,73%."""
    return "—" if pd.isna(x) else vn(f"{x:.{d}%}")


def pu(x, d=2):
    """Số đã ở đơn vị % -> chuỗi %."""
    return "—" if pd.isna(x) else vn(f"{x:.{d}f}%")


def as_pct_units(s):
    """Đưa cột tỷ lệ về đơn vị % (nếu đang là thập phân)."""
    s = pd.to_numeric(s, errors="coerce")
    return s * 100 if s.abs().max() <= 1 else s


def to_bool(series):
    """Chuyển Boolean an toàn (tránh bool('False') == True)."""
    if pd.api.types.is_bool_dtype(series):
        return series
    m = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    return series.astype(str).str.strip().str.lower().map(m)


_PCT_NAME = re.compile(r"(^|[_\s])(var|es)($|[_\s\d])|loss|return|lợi suất|tổn thất|volat|biến động", re.I)
_COUNT_NAME = re.compile(r"n_|số|count|viol|vi phạm|obs|p_value|lr_", re.I)


def fmt_metric(v, col="", series=None):
    """Định dạng theo tên cột (kết quả đã ở kiểu số Việt Nam); cột có '(%)' hoặc giá trị lớn >= 1 được hiểu là đã ở đơn vị %."""
    if pd.isna(v):
        return "—"
    if "(%)" in str(col):
        return vn(f"{v:.2f}%")
    if _PCT_NAME.search(str(col)) and not _COUNT_NAME.search(str(col)):
        big = series is not None and series.abs().max() >= 1
        return vn(f"{v:.2f}%") if big else vn(f"{v:.2%}")
    return vn(f"{v:,.4g}")


def style_fig(fig, title="", height=430):
    fig.update_layout(
        template="plotly_white", height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=int(18 * PLOT_SCALE), color=C["ink"])),
        font=dict(family="Be Vietnam Pro, Arial", size=int(14 * PLOT_SCALE), color=C["text"]),
        margin=dict(l=20, r=20, t=70, b=20), hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        separators=",.",  # dấu thập phân "," và dấu nghìn "." trên trục và tooltip
    )
    fig.update_xaxes(showgrid=False, linecolor=C["line"])
    fig.update_yaxes(gridcolor="#E8F0F8", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 3. ĐỌC DỮ LIỆU
# ============================================================
@st.cache_data(ttl=3600)
def load_table(filename: str) -> pd.DataFrame:
    path = ROOT_DIR / "outputs" / "tables" / filename
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_raw() -> pd.DataFrame:
    for name in ("portfolio_returns.csv", "portfolio_aligned.csv"):
        path = ROOT_DIR / "data" / "main" / name
        if path.exists():
            try:
                df = pd.read_csv(path)
                dcols = [c for c in df.columns if "date" in c.lower() or c == "Unnamed: 0"]
                if dcols:
                    df = df.rename(columns={dcols[0]: "Date"})
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                return df
            except Exception:
                pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_prices() -> pd.DataFrame:
    frames = []
    for name in ("portfolio_aligned.csv", "portfolio_returns.csv"):
        path = ROOT_DIR / "data" / "main" / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            dc = [c for c in df.columns if "date" in c.lower() or c == "Unnamed: 0"]
            if dc:
                df = df.rename(columns={dc[0]: "Date"})
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f[[c for c in f.columns if c == "Date" or c not in out.columns]], on="Date", how="outer")
    return out.sort_values("Date")


def return_col(df):
    if df.empty:
        return None
    if "portfolio_return" in df.columns:
        return "portfolio_return"
    cands = [c for c in df.columns if "return" in c.lower()]
    return cands[0] if cands else df.columns[-1]


def with_date(df):
    if df.empty:
        return df
    df = df.copy()
    col = "Date" if "Date" in df.columns else df.columns[0]
    df["Date"] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def series_tests(r: pd.Series) -> pd.DataFrame:
    rows = []
    try:
        from scipy import stats
        jb = stats.jarque_bera(r)
        rows.append(("Jarque-Bera", "Lợi suất có phân phối chuẩn không?", jb.statistic, jb.pvalue,
                     "Không chuẩn (đuôi dày)" if jb.pvalue < 0.05 else "Xấp xỉ chuẩn", jb.pvalue < 0.05))
    except Exception:
        pass
    try:
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.stats.diagnostic import het_arch
        adf = adfuller(r, autolag="AIC")
        rows.append(("ADF", "Chuỗi lợi suất có ổn định không?", adf[0], adf[1],
                     "Chuỗi dừng" if adf[1] < 0.05 else "Chuỗi không dừng", adf[1] >= 0.05))
        lm = het_arch(r, nlags=10)
        rows.append(("ARCH-LM (10 độ trễ)", "Biến động có xu hướng co cụm không?", lm[0], lm[1],
                     "Có hiệu ứng ARCH" if lm[1] < 0.05 else "Không có hiệu ứng ARCH", lm[1] < 0.05))
    except Exception:
        pass
    return pd.DataFrame(rows, columns=["Kiểm định", "Câu hỏi kiểm tra", "Thống kê", "p-value", "Kết luận", "flag"])


def hist_var_es(r: pd.Series):
    """VaR và ES lịch sử toàn mẫu ở mức 97,5% (dương = mức lỗ)."""
    q = np.quantile(r, 1 - CONF)
    return -q, -r[r <= q].mean()


def _verdict(x):
    return "✖ Không đạt" if bool(x) else "✔ Đạt"


def ind_txt(r):
    """Diễn đạt kết quả kiểm định độc lập của một dòng backtest."""
    return "bị bác bỏ" if r["reject_ind"] else "không bị bác bỏ"


def _remark(r):
    if not r["reject_H0"] and not r["reject_ind"] and not r["reject_cc"]:
        return "Đạt cả hai tiêu chí"
    if not r["reject_H0"] and r["reject_ind"]:
        return "Đủ tần suất nhưng vi phạm dồn cụm"
    if r["reject_H0"] and not r["reject_ind"]:
        return "Ít dồn cụm nhưng vi phạm quá nhiều"
    return "Lệch cả tần suất lẫn tính độc lập"


def _merge_bt(k, c):
    """Ghép bảng Kupiec và Christoffersen theo 'Phương pháp'; chấp nhận thiếu một số cột."""
    m_col = "Phương pháp"
    if k.empty or c.empty or m_col not in k.columns or m_col not in c.columns:
        return pd.DataFrame()
    k, c = k.copy(), c.copy()
    for d_ in (k, c):
        for col in [x for x in d_.columns if x.startswith("reject")]:
            d_[col] = to_bool(d_[col]).fillna(False).astype(bool)
    kc = [x for x in ["n_obs", "n_violations", "violation_rate", "LR_statistic", "p_value", "reject_H0"] if x in k.columns]
    cc = [x for x in ["p_value_uc", "LR_ind", "p_value_ind", "reject_ind", "LR_cc", "p_value_cc", "reject_cc"] if x in c.columns]
    m = k[[m_col] + kc].merge(c[[m_col] + cc], on=m_col)
    if "violation_rate" in m.columns:
        m["rate_pct"] = as_pct_units(m["violation_rate"])
    return m


def backtest_table():
    return _merge_bt(load_table("kupiec_results.csv"), load_table("christoffersen_results.csv"))


def oos_table(key):
    return _merge_bt(load_table(f"stress_oos_{key}_kupiec.csv"), load_table(f"stress_oos_{key}_christoffersen.csv"))


def html_bt(bt, remark=False):
    """Bảng backtest: số vi phạm, p-value và kết luận của Kupiec, Độc lập, Conditional Coverage.
    remark=True thêm cột Nhận xét (cần đủ ba cột reject_*)."""
    d = pd.DataFrame({"Mô hình": bt["Phương pháp"].astype(str)})
    tones = {}
    if "n_obs" in bt.columns:
        d["Số ngày"] = bt["n_obs"].astype(int)
    if "n_violations" in bt.columns:
        d["Số vi phạm"] = bt["n_violations"].astype(int)
    if "rate_pct" in bt.columns:
        d["Tỷ lệ vi phạm (%)"] = bt["rate_pct"].map(lambda v: d2(v))
    for p_col, r_col, p_name, v_name in [
        ("p_value", "reject_H0", "p-value Kupiec", "Kupiec"),
        ("p_value_ind", "reject_ind", "p-value IND", "Độc lập (IND)"),
        ("p_value_cc", "reject_cc", "p-value CC", "Christoffersen (CC)"),
    ]:
        if p_col in bt.columns:
            d[p_name] = bt[p_col].map(lambda v: d2(v, 4))
        if r_col in bt.columns:
            d[v_name] = bt[r_col].map(_verdict)
            tones[v_name] = ["bad" if bool(x) else "good" for x in bt[r_col]]
    if remark and {"reject_H0", "reject_ind", "reject_cc"}.issubset(bt.columns):
        d["Nhận xét"] = bt.apply(_remark, axis=1)
    html_table(d, tones)


def worst_stress():
    """Giai đoạn stress nặng nhất. Ưu tiên cột ES Historical (khớp báo cáo), sau đó ES khác, sau cùng cột số đầu tiên.
    Nhãn giai đoạn lấy từ cột chữ (bỏ qua cột chỉ số 'Unnamed')."""
    comp = load_table("stress_test_comparison.csv")
    if comp.empty:
        return None
    num = [c for c in comp.select_dtypes("number").columns if "unnamed" not in c.lower()]
    if not num:
        return None
    cat = [c for c in comp.columns if c not in num and "unnamed" not in c.lower()]
    es = [c for c in num if re.search(r"(^|_)es($|_|\d)|shortfall", c.lower())]
    hist = [c for c in es if "hist" in c.lower()]
    col = (hist or es or num)[0]
    i = comp[col].abs().idxmax()
    if cat:
        label = str(comp.loc[i, cat[0]])
    else:
        periods = ["2008–2009", "2020", "2022"]
        label = periods[i] if len(comp) == len(periods) and isinstance(i, (int, np.integer)) else str(i)
    return label, col, float(comp.loc[i, col]), comp[col]


# ============================================================
# 4. NẠP DỮ LIỆU CHÍNH
# ============================================================
rolling_var = with_date(load_table("rolling_var_975.csv"))
garch_out = with_date(load_table("garch_var_es_rolling_975.csv"))
vio = load_table("backtest_violations.csv")
vio = with_date(vio).dropna(subset=["Date"]) if not vio.empty else vio
stress_comp = load_table("stress_test_comparison.csv")
bt = backtest_table()
raw = load_raw()
rcol = return_col(raw)
returns = pd.to_numeric(raw[rcol], errors="coerce").dropna() if rcol else pd.Series(dtype=float)
oos_all = {y: oos_table(k) for y, k in OOS_KEYS.items()}
oos_all = {y: t for y, t in oos_all.items() if not t.empty and {"reject_H0", "rate_pct"}.issubset(t.columns)}
BT_NEED = {"reject_H0", "reject_ind", "reject_cc", "n_obs", "rate_pct"}

if not raw.empty and "Date" in raw.columns and raw["Date"].notna().any():
    DATA_SPAN = f"{raw['Date'].min():%d/%m/%Y} – {raw['Date'].max():%d/%m/%Y}"
else:
    DATA_SPAN = f"2016 – {DATA_FREEZE_DATE}"

# ============================================================
# 5. TIÊU ĐỀ
# ============================================================
st.markdown(
    """<div class="hero"><h1>Đo lường rủi ro thị trường: VaR, ES và kiểm định ngược</h1>
    <p>Danh mục """ + f"{W_VN}% VN-Index và {W_US}% S&amp;P 500 (quy đổi USD/VND)" + """ &nbsp;|&nbsp; Độ tin cậy """
    + vn(f"{CONF:.1%}") + """ &nbsp;|&nbsp; Cửa sổ trượt """ + str(WINDOW)
    + """ phiên &nbsp;|&nbsp; Dữ liệu chốt ngày """ + DATA_FREEZE_DATE + """</p></div>""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Tổng quan", "Dữ liệu", "Mô hình VaR", "GARCH & ES", "Backtesting", "Stress test", "Kết luận"])

# ============================================================
# TAB 1: TỔNG QUAN
# ============================================================
with tabs[0]:
    section("1. Tổng quan nghiên cứu", "Nghiên cứu đo mức lỗ có thể xảy ra của danh mục hai thị trường và kiểm tra mô hình nào dự báo đáng tin.")
    text_card("Mục tiêu", "Đo rủi ro, rồi kiểm tra độ tin cậy của phép đo",
              "Nghiên cứu đo lường rủi ro thị trường của danh mục VN-Index và S&amp;P 500 bằng bốn mô hình VaR (Historical, Parametric, Monte Carlo, GARCH), "
              "kết hợp Expected Shortfall 97,5%, sau đó kiểm định khả năng dự báo bằng Kupiec và Christoffersen và thử lại trong ba giai đoạn khủng hoảng.",
              C["blue"], "m")
    section("2. Cấu hình nghiên cứu", "Toàn bộ mô hình dùng chung một bộ tham số để so sánh công bằng.")
    html_table(pd.DataFrame({
        "Thành phần": ["Danh mục", "Dữ liệu chính", "Dữ liệu stress 2008", "Lợi suất", "Mức tin cậy", "Kỳ hạn",
                       "Cửa sổ trượt", "Monte Carlo", "Hạt giống ngẫu nhiên"],
        "Thiết lập": [f"{W_VN}% VN-Index + {W_US}% S&P 500 (tỷ trọng cố định)", DATA_SPAN,
                      "01/01/2007 – 31/12/2009", "Lợi suất logarit", vn(f"{CONF:.1%}"), "1 ngày",
                      f"{WINDOW} phiên", f"{MC_SIMULATIONS:,} kịch bản mỗi ngày".replace(",", "."), f"{RANDOM_SEED} (cộng chỉ số ngày)"],
    }))

    section("3. Ba câu hỏi nghiên cứu", "Ba câu hỏi nghiên cứu (RQ1–RQ3) của đề án; kiểm định Kupiec – Christoffersen là công cụ kiểm chứng dùng chung cho cả ba.")
    q1, q2, q3 = st.columns(3)
    with q1: text_card("RQ1 — Ba phương pháp VaR", "Historical, Parametric và Monte Carlo khác nhau thế nào?",
                       "So sánh mức độ bảo thủ và độ chính xác của ba phương pháp VaR, kiểm chứng bằng Kupiec và Christoffersen. "
                       "Xem tab Mô hình VaR và Backtesting.", C["blue"], "m")
    with q2: text_card("RQ2 — GARCH và ES", "GARCH có cải thiện backtesting, và ES phản ánh rủi ro đuôi ra sao?",
                       "Đánh giá GARCH(1,1) so với biến động không điều kiện, và ES 97,5% so với VaR thông thường. "
                       "Xem tab GARCH &amp; ES và Backtesting.", C["coral"], "m")
    with q3: text_card("RQ3 — Stress test", "Hiệu năng mô hình thay đổi thế nào khi khủng hoảng?",
                       "Thử lại trong giai đoạn 2008, 2020 và 2022, xem mô hình nào bền vững nhất. Xem tab Stress test.", C["amber"], "m")

    section("4. Giải thích ngắn các khái niệm")
    html_table(pd.DataFrame({
        "Khái niệm": ["VaR", "Historical VaR", "Parametric VaR", "Monte Carlo VaR", "GARCH VaR", "Expected Shortfall", "Kupiec", "Christoffersen"],
        "Ý nghĩa": [
            "Ngưỡng tổn thất mà xác suất bị vượt qua trong một ngày chỉ là 2,5%.",
            "Lấy phân vị thực nghiệm của 250 phiên gần nhất, không giả định phân phối.",
            "Dùng trung bình, độ lệch chuẩn và giả định phân phối chuẩn.",
            "Mô phỏng 10.000 kịch bản từ cùng phân phối chuẩn nên gần như trùng Parametric.",
            "Cho độ biến động thay đổi theo thời gian qua GARCH(1,1), nên VaR tự co giãn.",
            "Mức lỗ trung bình khi lỗ đã vượt VaR. Luôn lớn hơn VaR.",
            "Kiểm tra tỷ lệ vi phạm thực tế có sát 2,5% không.",
            "Kiểm tra các vi phạm có độc lập hay dồn thành cụm, và kết hợp thành độ phủ có điều kiện.",
        ],
    }))
    howto("Trong các bảng và biểu đồ, VaR là độ lớn khoản lỗ (số dương), ví dụ VaR 1,80% nghĩa là ngưỡng lỗ một ngày là 1,80%. "
          "Khi vẽ cùng lợi suất thực tế, ngưỡng VaR được đổi sang phía âm để thấy ranh giới tổn thất.")

# ============================================================
# TAB 2: DỮ LIỆU
# ============================================================
with tabs[1]:
    section("1. Nguồn và phạm vi dữ liệu", "Ba chuỗi dữ liệu ngày tạo nên danh mục nhìn từ góc độ nhà đầu tư Việt Nam.")
    html_table(pd.DataFrame({
        "Dữ liệu": ["VN-Index", "S&P 500", "USD/VND", "Danh mục"],
        "Nguồn (mẫu chính)": ["DNSE OpenAPI", "FRED (mã SP500)", "Yahoo Finance (VND=X)", "Tính từ hai tài sản theo tỷ trọng 50/50"],
        "Nguồn (stress 2008)": ["DNSE OpenAPI", "Yahoo Finance (^GSPC)", "Yahoo Finance (VND=X)", "Tính từ hai tài sản theo tỷ trọng 50/50"],
        "Vai trò": ["Đại diện thị trường cổ phiếu Việt Nam", "Đại diện thị trường cổ phiếu Mỹ",
                    "Quy đổi lợi suất S&P 500 sang VND", "Đối tượng đo lường rủi ro"],
    }))
    howto(f"Dữ liệu chính được cố định đến <b>{DATA_FREEZE_DATE}</b>. Giai đoạn stress 2008 dùng mẫu riêng 01/01/2007 – 31/12/2009 "
          "vì nằm ngoài khoảng dữ liệu chính và thị trường Việt Nam thời đó có cấu trúc khác nhiều so với hiện nay.")

    section("2. Dữ liệu danh mục", "Hai chỉ số cấu thành và diễn biến của danh mục kết hợp.")
    if raw.empty:
        warn("Chưa tìm thấy file dữ liệu trong <code>data/main/</code>. Dashboard vẫn hiển thị được các kết quả trong <code>outputs/tables/</code>.")
    else:
        pr = load_prices()
        ret_cols = [c for c in pr.columns if "return" in c.lower()]
        px_cols = [c for c in pr.columns if c != "Date" and c not in ret_cols and "unnamed" not in c.lower()
                   and "weight" not in c.lower() and pd.api.types.is_numeric_dtype(pr[c])]
        palette = [C["blue"], C["teal"], C["amber"], C["violet"], C["coral"]]

        def nice(c):
            lc = c.lower()
            if "sp" in lc or "s&p" in lc:
                return "S&P 500 (VND)" if "vnd" in lc else "S&P 500 (USD)"
            if "usd" in lc and "vnd" in lc: return "Tỷ giá USD/VND"
            if lc.startswith("vn"): return "VN-Index"
            return c

        labels = {c: nice(c) for c in px_cols}
        for lab in set(labels.values()):
            dup = [c for c, l in labels.items() if l == lab]
            if len(dup) > 1:
                for c in dup: labels[c] = f"{lab} [{c}]"
        fig = go.Figure()
        in_port = [c for c in px_cols if labels[c] in ("VN-Index", "S&P 500 (VND)")]
        for i, c in enumerate(in_port or px_cols):
            s_ = pr[["Date", c]].dropna()
            if s_.empty: continue
            fig.add_trace(go.Scatter(x=s_["Date"], y=s_[c] / s_[c].iloc[0] * 100, name=labels[c],
                                     line=dict(color=palette[i % 5], width=2.4)))
        if rcol in pr.columns:
            s_ = pr[["Date", rcol]].dropna()
            fig.add_trace(go.Scatter(x=s_["Date"], y=cum_index(s_[rcol]), name="Danh mục",
                                     line=dict(color=C["ink"], width=3, dash="dot")))
        if fig.data:
            fig.update_yaxes(title="Chỉ số (ngày đầu = 100)")
            show(style_fig(fig, "Diễn biến các chỉ số cấu thành danh mục", 440))
            howto("Đường nào nằm cao hơn nghĩa là tài sản đó tăng nhiều hơn so với ngày đầu kỳ. Đường chấm đen là danh mục kết hợp. "
                  "S&P 500 (VND) đã cộng thêm phần USD tăng giá so với VND.")
        if "Date" in raw.columns and rcol in raw.columns:
            rr = raw[["Date", rcol]].dropna()
            fr = go.Figure(go.Scatter(x=rr["Date"], y=rr[rcol] * 100, name="Lợi suất ngày",
                                      line=dict(color=C["blue"], width=1.1)))
            fr.update_yaxes(title="Lợi suất ngày (%)", ticksuffix="%")
            show(style_fig(fr, "Lợi suất ngày của danh mục", 340))
            howto("Biên độ dao động không đều: các phiên lãi/lỗ lớn đi liền nhau thành cụm (rõ nhất quanh tháng 3/2020) rồi mới dịu dần. "
                  "Đây là hiện tượng co cụm biến động, cơ sở để đưa GARCH(1,1) vào mô hình.")

    section("3. Kiểm tra phân phối lợi suất", "Đây là cơ sở để chọn mô hình: lợi suất thực tế có giống phân phối chuẩn không?")
    if len(returns) == 0:
        warn("Không có dữ liệu thô để thực hiện thống kê mô tả.")
    else:
        ek = returns.kurtosis()
        takeaway(f"Lợi suất danh mục {'có đuôi dày' if ek > 0 else 'gần phân phối chuẩn'} (excess kurtosis {d2(ek)}, tức kurtosis {d2(ek + 3)} so với 3 của phân phối chuẩn); "
                 f"ngày tệ nhất lỗ {pct(abs(returns.min()))}, ngày tốt nhất lãi {pct(returns.max())}.")
        left, right = st.columns([1, 1.5])
        with left:
            html_table(pd.DataFrame({
                "Chỉ tiêu": ["Số quan sát", "Khoảng thời gian", "Trung bình", "Độ lệch chuẩn", "Nhỏ nhất", "Lớn nhất", "Phân vị 2,5%",
                             "Độ lệch (Skewness)", "Excess kurtosis"],
                "Giá trị": [vn(f"{len(returns):,}"), DATA_SPAN, pct(returns.mean(), 4), pct(returns.std(), 4), pct(returns.min(), 4),
                            pct(returns.max(), 4), pct(returns.quantile(0.025), 4), d2(returns.skew(), 4), d2(ek, 4)],
            }))
        with right:
            x = np.linspace(returns.min(), returns.max(), 200)
            pdf = np.exp(-0.5 * ((x - returns.mean()) / returns.std()) ** 2) / (returns.std() * np.sqrt(2 * np.pi))
            fh = go.Figure()
            fh.add_trace(go.Histogram(x=returns * 100, histnorm="probability density", nbinsx=80,
                                      name="Lợi suất thực tế", marker_color=C["sky"], opacity=.75))
            fh.add_trace(go.Scatter(x=x * 100, y=pdf / 100, name="Phân phối chuẩn", line=dict(color=C["coral"], width=2.5)))
            fh.update_xaxes(title="Lợi suất ngày (%)", ticksuffix="%")
            fh.update_layout(hovermode="closest", bargap=.03)
            show(style_fig(fh, "Phân phối lợi suất so với phân phối chuẩn", 340))
        howto("Nếu các cột cao hơn đường đỏ ở giữa và dày hơn ở hai đầu, lợi suất có đuôi dày: những ngày lãi hoặc lỗ cực đoan xảy ra thường xuyên hơn mô hình chuẩn giả định.")
        tests = series_tests(returns)
        if tests.empty:
            warn("Không chạy được các kiểm định Jarque-Bera, ADF, ARCH-LM. Kiểm tra đã cài <code>scipy</code> và <code>statsmodels</code> trong <code>requirements.txt</code> chưa.")
        else:
            if len(tests) < 3:
                warn("Một số kiểm định chưa chạy được (có thể thiếu <code>statsmodels</code>), bảng dưới chỉ hiển thị các kiểm định đã tính.")
            t = tests.copy()
            t["Thống kê"] = t["Thống kê"].map(lambda v: vn(f"{v:,.2f}"))
            t["p-value"] = t["p-value"].map(lambda v: "< 0,001" if v < 0.001 else d2(v, 3))
            flags = tests["flag"].tolist()
            html_table(t.drop(columns="flag"), {"Kết luận": ["warn" if f else "good" for f in flags]})

    section("4. Nguyên tắc xử lý dữ liệu")
    html_table(pd.DataFrame({
        "Bước": ["Đồng bộ ngày", "Quy đổi tỷ giá", "Tính lợi suất", "Xây dựng danh mục", "Ngoại lệ", "Dữ liệu stress"],
        "Thiết lập": ["Inner Join", "S&P 500 x USD/VND", "Lợi suất logarit", f"{W_VN}% VN-Index + {W_US}% S&P 500",
                      "Giữ nguyên, không loại bỏ", "2008–2009, 2020, 2022"],
        "Mục đích": ["Chỉ giữ các ngày cả hai thị trường cùng giao dịch, không tạo giá giả.",
                     "Hai lợi suất chỉ cộng được khi cùng đồng tiền VND.",
                     "Có tính cộng theo thời gian, phù hợp mô hình tham số và GARCH.",
                     "Đo rủi ro nội tại của cấu trúc tài sản, không phải một chiến lược phân bổ.",
                     "Các phiên giảm mạnh chính là rủi ro đuôi cần đo; cắt bỏ sẽ làm mô hình đánh giá thấp rủi ro.",
                     "Đánh giá mô hình khi thị trường biến động mạnh."],
    }))

# ============================================================
# TAB 3: MÔ HÌNH VAR
# ============================================================
@_fragment
def violation_section():
    """Mục 3 của tab Mô hình VaR: đặt trong fragment để đổi lựa chọn không chạy lại cả trang."""
    if vio.empty:
        missing_file("backtest_violations.csv")
        return
    if "actual_return" in vio.columns:
        act = vio[["Date", "actual_return"]].rename(columns={"actual_return": "ret"})
    elif not raw.empty and rcol:
        act = raw[["Date", rcol]].rename(columns={rcol: "ret"})
        act = act[act["Date"] >= vio["Date"].min()]
    else:
        act = pd.DataFrame(columns=["Date", "ret"])
    fig = go.Figure()
    if len(act):
        fig.add_trace(go.Scatter(x=act["Date"], y=act["ret"] * 100, name="Lợi suất thực tế",
                                 line=dict(color=MODEL_COLORS["Actual"], width=1)))
    for c, l in VAR_COLS:
        if c in vio.columns:
            fig.add_trace(go.Scatter(x=vio["Date"], y=-vio[c].abs() * 100, name=f"VaR {l}",
                                     line=dict(color=MODEL_COLORS.get(l, "#000"), width=1.6)))
    vcols = [c for c in vio.columns if c.endswith("_violation")]
    names = [c.replace("_violation", "") for c in vcols]
    pick_m = st.radio("Đánh dấu ngày vi phạm của mô hình", ["Không đánh dấu"] + names, horizontal=True)
    if pick_m != "Không đánh dấu" and len(act):
        d = vio.loc[vio[pick_m + "_violation"] == 1, ["Date"]].merge(act, on="Date", how="left")
        fig.add_trace(go.Scatter(x=d["Date"], y=d["ret"] * 100, mode="markers", name=f"Vi phạm ({pick_m})",
                                 marker=dict(color=MODEL_COLORS.get(pick_m, C["coral"]), size=10, line=dict(color="#fff", width=1.5))))
    fig.update_yaxes(title="Lợi suất / ngưỡng VaR (%)", ticksuffix="%")
    show(style_fig(fig, "Lợi suất thực tế và ngưỡng VaR", 500))
    howto("Mỗi lần đường xám cắt xuống dưới một đường VaR là một vi phạm. Chấm màu mọc sát nhau thành đám cho thấy mô hình phản ứng chậm khi thị trường chuyển sang giai đoạn biến động.")


with tabs[2]:
    section("1. VaR theo thời gian", "VaR được tính lại mỗi ngày từ 250 phiên gần nhất.")
    mv = {}
    if rolling_var.empty:
        missing_file("rolling_var_975.csv")
    else:
        df = rolling_var.sort_values("Date")
        stat_models = [(c, l) for c, l in VAR_COLS[:3] if c in df.columns]
        mv = {l: pd.to_numeric(df[c], errors="coerce").abs().mean() for c, l in stat_models}
        if mv:
            lo, hi = min(mv, key=mv.get), max(mv, key=mv.get)
            takeaway(f"VaR {vn(f'{CONF:.1%}')} trung bình dao động từ {pct(mv[lo])} ({lo}) đến {pct(mv[hi])} ({hi}): "
                     "mỗi phương pháp cho một mức rủi ro khác nhau nên cần backtest để biết mô hình nào đáng tin.")
        fig = go.Figure()
        for c, l in stat_models:
            fig.add_trace(go.Scatter(x=df["Date"], y=pd.to_numeric(df[c], errors="coerce").abs() * 100, name=l,
                                     line=dict(color=MODEL_COLORS.get(l, C["blue"]), width=2.2)))
        fig.update_yaxes(title="VaR (%)", ticksuffix="%")
        show(style_fig(fig, "VaR cuốn 97,5% (độ lớn khoản lỗ)", 440))
        howto("Giá trị tăng nghĩa là ngưỡng lỗ dự báo đang lớn hơn. Historical VaR có dạng bậc thang và giữ mức cao khoảng 250 phiên sau một cú sốc (hiệu ứng bóng ma).")

    section("2. So sánh phương pháp", "Cùng một danh mục, cùng một cửa sổ, chỉ khác cách xử lý phân phối.")
    if not rolling_var.empty:
        rows = []
        for c, l in VAR_COLS[:3]:
            if c in rolling_var.columns:
                s = pd.to_numeric(rolling_var[c], errors="coerce").dropna().abs()
                if len(s):
                    rows.append({"Phương pháp": l, "VaR trung bình": pct(s.mean(), 3), "Trung vị": pct(s.median(), 3),
                                 "Nhỏ nhất": pct(s.min(), 3), "Lớn nhất": pct(s.max(), 3),
                                 "Điểm cần lưu ý": {"Historical": "Phụ thuộc mạnh vào mẫu quá khứ; dạng bậc thang, phản ứng chậm.",
                                                    "Parametric": "Giả định chuẩn nên có xu hướng đánh giá thấp đuôi phân phối.",
                                                    "Monte Carlo": "Không thêm thông tin đuôi so với Parametric; giá trị nằm ở khả năng mở rộng sang phân phối khác."}.get(l, "")})
        if rows:
            html_table(pd.DataFrame(rows))
        if "Parametric" in mv and "Monte Carlo" in mv:
            warn(f"Monte Carlo mô phỏng từ cùng phân phối chuẩn với tham số của Parametric nên hai mô hình gần như trùng nhau "
                 f"(chênh lệch trung bình chỉ {d2(abs(mv['Monte Carlo'] - mv['Parametric']) * 100, 3)} điểm phần trăm, do sai số lấy mẫu). "
                 "Về thực chất chỉ có hai quan điểm độc lập: Historical và phân phối chuẩn.")

    section("3. Kiểm tra vi phạm VaR", "Một vi phạm xảy ra khi lợi suất thực tế nằm thấp hơn ngưỡng VaR của ngày đó.")
    violation_section()

# ============================================================
# TAB 4: GARCH & ES
# ============================================================
with tabs[3]:
    section("1. GARCH(1,1) và Expected Shortfall", "GARCH cho VaR co giãn theo biến động của thị trường; ES cho biết mức lỗ trung bình khi VaR bị vượt.")
    howto("Với mỗi cửa sổ 250 phiên, GARCH(1,1) dự báo độ biến động cho ngày kế tiếp, từ đó tính GARCH-VaR và ES ở mức 97,5%. "
          "Dashboard đọc trực tiếp kết quả từ <code>garch_var_es_rolling_975.csv</code>, không ước lượng lại GARCH để tránh lệch với pipeline nghiên cứu.")
    v = e = np.nan
    hv = he = None
    if garch_out.empty:
        missing_file("garch_var_es_rolling_975.csv")
    else:
        df = garch_out.sort_values("Date")
        v = df["VaR_GARCH"].abs().mean() if "VaR_GARCH" in df.columns else np.nan
        e = df["ES_97_5"].abs().mean() if "ES_97_5" in df.columns else np.nan
        if len(returns):
            hv, he = hist_var_es(returns)
        if v and v > 0 and not pd.isna(e):
            msg = f"ES của GARCH chỉ gấp {d2(e / v)} lần GARCH-VaR do giả định phân phối chuẩn."
            if hv and he and (he / hv) > (e / v):
                msg += (f" Trong khi ES lịch sử gấp {d2(he / hv)} lần VaR lịch sử, cho thấy giả định phân phối chuẩn của GARCH "
                        "có thể chưa phản ánh đầy đủ rủi ro ở phần đuôi.")
            takeaway(msg)
        fig = go.Figure()
        if "VaR_GARCH" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["VaR_GARCH"].abs() * 100, name="GARCH-VaR 97,5%",
                                     line=dict(color=MODEL_COLORS["GARCH"], width=2.4)))
        if "ES_97_5" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["ES_97_5"].abs() * 100, name="ES 97,5% (GARCH, chuẩn)",
                                     line=dict(color=MODEL_COLORS["ES_97_5"], width=2.4, dash="dash")))
        fig.update_yaxes(title="Độ lớn khoản lỗ (%)", ticksuffix="%")
        show(style_fig(fig, "GARCH-VaR và ES 97,5% theo thời gian", 440))
        howto("VaR (cam) tự nâng lên khi thị trường biến động mạnh và hạ xuống khi yên ắng. ES (đỏ nét đứt) luôn nằm trên VaR vì phản ánh mức lỗ trung bình ở vùng đuôi.")
        if "GARCH_Volatility" in df.columns:
            fv = go.Figure(go.Scatter(x=df["Date"], y=df["GARCH_Volatility"] * 100, name="Độ biến động có điều kiện",
                                      line=dict(color=C["violet"], width=2)))
            fv.update_yaxes(title="Độ biến động (%/ngày)", ticksuffix="%")
            show(style_fig(fv, "Độ biến động có điều kiện từ GARCH(1,1)", 340))
            howto("Các đỉnh tập trung theo cụm rồi giảm dần chứ không về ngay mức nền: đó là hiện tượng co cụm biến động mà GARCH nắm bắt được.")

        section("2. Tóm tắt GARCH và ES", "Tỷ lệ ES/VaR của GARCH (giả định chuẩn) đặt cạnh ES/VaR lịch sử toàn mẫu (không giả định chuẩn).")
        m1, m2, m3, m4 = st.columns(4)
        with m1: card("ES / VaR của GARCH", f"{d2(e / v)} lần" if v and v > 0 else "—",
                      f"Cố định ≈ {d2(ES_VAR_NORMAL)} do giả định chuẩn", C["violet"])
        with m2: card("ES / VaR lịch sử", f"{d2(he / hv)} lần" if hv and he else "—", "Toàn mẫu, không giả định chuẩn", C["blue"])
        with m3: card("VaR lịch sử toàn mẫu", pct(hv) if hv else "—", "Phân vị 2,5% của lợi suất", C["sky"])
        with m4: card("ES lịch sử toàn mẫu", pct(he) if he else "—", "Trung bình phần đuôi 2,5%", C["teal"])

    section("3. GARCH so với biến động không điều kiện", "RQ2: GARCH có cải thiện kết quả backtesting so với Parametric (độ lệch chuẩn cửa sổ 250 phiên, không điều kiện) không?")
    need_g = {"reject_H0", "reject_ind", "n_obs", "rate_pct"}
    if bt.empty or not need_g.issubset(bt.columns):
        warn("Chưa có kết quả backtest để so sánh GARCH với Parametric.")
    else:
        nm_ = bt["Phương pháp"].astype(str).str.lower()
        sub = bt[nm_.str.contains("parametric|garch")].reset_index(drop=True)
        g_row = bt[nm_.str.contains("garch")]
        p_row = bt[nm_.str.contains("parametric")]
        if len(g_row) and len(p_row):
            html_bt(sub)
            g_, p_ = g_row.iloc[0], p_row.iloc[0]
            howto(f"Tỷ lệ vi phạm: GARCH {d2(g_['rate_pct'])}% so với Parametric {d2(p_['rate_pct'])}% (kỳ vọng 2,5%). "
                  f"Tính độc lập: GARCH {ind_txt(g_)}, Parametric {ind_txt(p_)}. "
                  "Cần đọc đồng thời tần suất và tính độc lập: cải thiện chỉ tiêu này không đồng nghĩa với cải thiện chỉ tiêu kia.")

    warn("GARCH hiện dùng phân phối chuẩn, hằng số trung bình và có cơ chế dự phòng EWMA khi ước lượng lỗi. GARCH phản ứng nhanh hơn "
         "không có nghĩa là ít vi phạm hơn các mô hình tĩnh: xem tab Backtesting. ES ở đây chưa được backtest.")

# ============================================================
# TAB 5: BACKTESTING
# ============================================================
with tabs[4]:
    need = BT_NEED
    section("1. Kết quả kiểm định ngược",
            "Kupiec kiểm tra tỷ lệ vi phạm có phù hợp với mức kỳ vọng 2,5% hay không; Christoffersen kiểm tra tính độc lập của các vi phạm "
            "và kết hợp hai điều kiện thành kiểm định CC.")
    if bt.empty or not need.issubset(bt.columns):
        warn("Chưa có kupiec_results.csv hoặc christoffersen_results.csv đầy đủ cột trong <code>outputs/tables/</code>.")
    else:
        n = len(bt)
        ok_k = bt.loc[~bt["reject_H0"], "Phương pháp"].tolist()
        ok_i = bt.loc[~bt["reject_ind"], "Phương pháp"].tolist()
        ok_c = bt.loc[~bt["reject_cc"], "Phương pháp"].tolist()
        n_obs_max = int(bt["n_obs"].max())
        names_ = lambda lst: ", ".join(lst) or "không mô hình nào"
        takeaway(f"Đạt Kupiec: {names_(ok_k)}. Đạt kiểm định độc lập: {names_(ok_i)}. "
                 f"{len(ok_c)}/{n} mô hình đạt độ phủ có điều kiện (CC).")
        b1, b2, b3, b4 = st.columns(4)
        with b1: card("Số ngày kiểm định", vn(f"{n_obs_max:,}"), f"Tỷ lệ vi phạm kỳ vọng {d2(P0_PCT, 1)}%", C["blue"])
        with b2: card("Đạt Kupiec", f"{len(ok_k)} / {n}", "Tỷ lệ vi phạm phù hợp 2,5%", C["teal"])
        with b3: card("Đạt độc lập (IND)", f"{len(ok_i)} / {n}", "Vi phạm không dồn cụm", C["amber"])
        with b4: card("Đạt cả hai (CC)", f"{len(ok_c)} / {n}", "Độ phủ có điều kiện", C["green"])
        html_bt(bt, remark=True)
        howto("p-value < 0,05: bác bỏ giả thuyết H₀ của kiểm định ở mức ý nghĩa 5%; p-value ≥ 0,05: chưa đủ bằng chứng để bác bỏ H₀. "
              "Một mô hình có thể đạt Kupiec nhưng vi phạm dồn cụm, hoặc ít dồn cụm nhưng vi phạm quá nhiều. Chỉ đạt CC khi vượt cả hai.")
        warn(f"Với khoảng {round(n_obs_max * (1 - CONF))} vi phạm kỳ vọng, các kiểm định có sức mạnh hạn chế. Không bị bác bỏ không có nghĩa là mô hình chắc chắn chính xác.")

    section("2. Tỷ lệ vi phạm", "So với mức kỳ vọng 2,5% và theo từng năm để thấy vi phạm có tập trung vào năm biến động không.")
    if not bt.empty and "rate_pct" in bt.columns:
        fig = go.Figure(go.Bar(x=bt["Phương pháp"], y=bt["rate_pct"], text=bt["rate_pct"].map(lambda v: f"{d2(v)}%"),
                               textposition="outside",
                               marker_color=[C["coral"] if r else C["teal"] for r in bt.get("reject_H0", [False] * len(bt))]))
        fig.add_hline(y=P0_PCT, line_dash="dash", line_color=C["ink"], annotation_text=f"Kỳ vọng {d2(P0_PCT, 1)}%")
        fig.update_yaxes(title="Tỷ lệ vi phạm (%)", ticksuffix="%")
        fig.update_layout(hovermode="closest")
        show(style_fig(fig, "Tỷ lệ vi phạm VaR so với mức kỳ vọng (đỏ: bị Kupiec bác bỏ)", 400))
    if not vio.empty:
        vnames = [c.replace("_violation", "") for c in vio.columns if c.endswith("_violation")]
        if vnames:
            exp = vio["Date"].dt.year.value_counts().sort_index() * (1 - CONF)
            fig2 = go.Figure()
            for n_ in vnames:
                cnt = vio.loc[vio[n_ + "_violation"] == 1, "Date"].dt.year.value_counts().reindex(exp.index, fill_value=0)
                fig2.add_trace(go.Bar(x=exp.index.astype(str), y=cnt.values, name=n_, marker_color=MODEL_COLORS.get(n_, C["blue"])))
            fig2.add_trace(go.Scatter(x=exp.index.astype(str), y=exp.values, name=f"Mức kỳ vọng ({d2(P0_PCT, 1)}%)",
                                      mode="lines+markers", line=dict(color=C["ink"], dash="dash", width=2.5)))
            fig2.update_layout(barmode="group")
            fig2.update_yaxes(title="Số ngày vi phạm")
            show(style_fig(fig2, "Số ngày vi phạm theo năm so với mức kỳ vọng", 400))
            howto("Cột vượt xa đường đứt nét ở năm nào nghĩa là mô hình đánh giá thấp rủi ro ở năm đó; năm biến động mạnh thường là năm vi phạm tập trung.")

    chr_ = load_table("christoffersen_results.csv")
    section("3. Thống kê LR", "Giá trị thống kê của ba kiểm định; kết luận đạt/không đạt và p-value đã có ở bảng mục 1.")
    with st.expander("Xem thống kê LR của Kupiec và Christoffersen"):
        lr_cols = [("LR_statistic", "LR Kupiec (UC)"), ("LR_ind", "LR độc lập (IND)"), ("LR_cc", "LR có điều kiện (CC)")]
        if bt.empty or not any(c in bt.columns for c, _ in lr_cols):
            warn("Chưa có thống kê LR trong kết quả kiểm định.")
        else:
            lr_df = pd.DataFrame({"Mô hình": bt["Phương pháp"].astype(str)})
            for c_, name_ in lr_cols:
                if c_ in bt.columns:
                    lr_df[name_] = bt[c_].map(lambda v: d2(v, 3))
            html_table(lr_df)

    section("4. So sánh p-value", "Cột nào nằm dưới đường 5% nghĩa là mô hình bị bác bỏ ở kiểm định đó.")
    if not chr_.empty and "Phương pháp" in chr_.columns:
        fig = go.Figure()
        for col, label in [("p_value_uc", "Unconditional Coverage"), ("p_value_ind", "Independence"), ("p_value_cc", "Conditional Coverage")]:
            if col in chr_.columns:
                fig.add_trace(go.Bar(x=chr_["Phương pháp"], y=chr_[col], name=label))
        fig.add_hline(y=0.05, line_dash="dash", line_color=C["ink"], annotation_text="α = 5%")
        fig.update_layout(barmode="group", hovermode="closest")
        fig.update_yaxes(title="p-value")
        show(style_fig(fig, "p-value của các kiểm định Christoffersen", 420))
    else:
        missing_file("christoffersen_results.csv")
    warn("Không nên kết luận một mô hình là “tốt nhất” chỉ từ một chỉ tiêu. Kupiec xét tỷ lệ vi phạm, Christoffersen bổ sung tính độc lập và độ phủ có điều kiện.")

# ============================================================
# TAB 6: STRESS TEST
# ============================================================
with tabs[5]:
    section("1. Thiết kế stress test", "Ba giai đoạn đại diện cho ba kiểu thị trường căng thẳng.")
    html_table(pd.DataFrame({
        "Giai đoạn": ["2008–2009", "2020", "2022"],
        "Bối cảnh": ["Khủng hoảng tài chính toàn cầu", "Cú sốc COVID-19", "Tăng lãi suất và cú sốc trái phiếu"],
        "Thời gian": ["01/01/2007 – 31/12/2009 (ước lượng từ 2007, OOS từ 2008)", "01/01/2020 – 31/12/2020", "01/01/2022 – 31/12/2022"],
        "Nguồn dữ liệu": ["data/stress_2008/ (mẫu riêng)", "data/main/", "data/main/"],
    }))
    howto("Mỗi giai đoạn trình bày hai lớp: kết quả trong mẫu (chỉ mô tả, vì VaR Historical được ước lượng từ chính giai đoạn đó) "
          "và backtest ngoài mẫu. Backtest ngoài mẫu phản ánh khả năng dự báo của mô hình trên dữ liệu không được dùng để ước lượng, "
          "nên phù hợp hơn để đánh giá khả năng bao phủ VaR.")

    section("2. So sánh các giai đoạn")
    ws = worst_stress()
    if stress_comp.empty:
        missing_file("stress_test_comparison.csv")
    else:
        if ws:
            takeaway(f"{ws[0]} là giai đoạn có {ws[1].replace(' (%)', '')} cao nhất ({fmt_metric(ws[2], ws[1], ws[3])}).")
        cn = [c for c in stress_comp.select_dtypes("number").columns if "unnamed" not in c.lower()]
        cc_ = [c for c in stress_comp.columns if c not in cn and "unnamed" not in c.lower()]
        xcol = "Giai đoạn" if "Giai đoạn" in stress_comp.columns else (cc_[0] if cc_ else None)
        if xcol:
            xs = stress_comp[xcol].astype(str)
        elif len(stress_comp) == 3:
            xs = pd.Series(["2008–2009", "2020", "2022"])
        else:
            xs = stress_comp.index.astype(str)
        disp = stress_comp.drop(columns=[c for c in stress_comp.columns if "unnamed" in c.lower()])
        if not xcol and len(stress_comp) == 3:
            disp.insert(0, "Giai đoạn", list(xs))
        for c_ in cn:
            disp[c_] = [fmt_metric(x, c_, stress_comp[c_]) for x in stress_comp[c_]]
        html_table(disp)
        warn("Các chỉ tiêu trong bảng tính trên chính dữ liệu của từng giai đoạn (trong mẫu), chỉ có giá trị mô tả. Tổn thất thực tế là lợi suất ngày thấp nhất, không phải mức lỗ tích lũy cả giai đoạn.")
        fv = go.Figure()
        for c, l in VAR_COLS:
            col = next((x for x in cn if x.startswith(c)), None)
            if col:
                fv.add_trace(go.Bar(x=xs, y=stress_comp[col], name=f"VaR {l}", marker_color=MODEL_COLORS.get(l, C["blue"])))
        es_cols = [x for x in cn if re.match(r"^es", x, re.I)]
        for col in es_cols:
            fv.add_trace(go.Bar(x=xs, y=stress_comp[col], name=col, marker_color=C["coral"], opacity=.6))
        loss_cols = [x for x in cn if re.search(r"thấp nhất|tổn thất|min|worst|actual", x, re.I)]
        if loss_cols:
            fv.add_trace(go.Scatter(x=xs, y=as_pct_units(stress_comp[loss_cols[0]]).abs(), mode="markers",
                                    name="Lợi suất ngày thấp nhất (độ lớn)",
                                    marker=dict(symbol="diamond", size=13, color=C["ink"], line=dict(color="#fff", width=1.5))))
        if fv.data:
            fv.update_layout(barmode="group", hovermode="closest")
            fv.update_yaxes(title="%")
            show(style_fig(fv, "So sánh VaR và ES giữa các giai đoạn stress", 440))
            txt_ = "Cột càng cao nghĩa là mô hình dự báo mức lỗ càng lớn trong giai đoạn đó. "
            if loss_cols:
                txt_ += "Hình thoi là độ lớn lợi suất ngày thấp nhất của giai đoạn. "
            txt_ += ("So sánh VaR với mức lợi suất thấp nhất giúp minh họa khoảng cách giữa mức lỗ VaR và một cú sốc cực đoan trong từng giai đoạn. "
                     "Khả năng bao phủ VaR được đánh giá chính thức qua backtest ngoài mẫu.")
            howto(txt_)

    def stress_block(year, filename):
        detail = load_table(filename)
        if detail.empty:
            missing_file(filename)
        else:
            dd = detail.copy()
            for c_ in dd.select_dtypes("number").columns:
                dd[c_] = [fmt_metric(x, c_, detail[c_]) for x in detail[c_]]
            html_table(dd)
            rate_cols = [c for c in detail.columns if "vi phạm" in c.lower() and "%" in c]
            name_cols = [c for c in detail.columns if detail[c].dtype == object]
            if rate_cols and name_cols:
                r_ = detail[rate_cols[0]]
                hi_i, lo_i = r_.idxmax(), r_.idxmin()
                howto(f"Trong mẫu, tỷ lệ vi phạm cao nhất là {d2(r_[hi_i])}% ({detail.loc[hi_i, name_cols[0]]}) và thấp nhất là {d2(r_[lo_i])}% ({detail.loc[lo_i, name_cols[0]]}), "
                      f"so với mức kỳ vọng {d2(P0_PCT, 1)}%. Con số của Historical gần kỳ vọng một phần vì VaR được ước lượng từ chính giai đoạn này.")
        st.markdown(f"##### Backtest ngoài mẫu — {year}")
        oos = oos_all.get(year)
        if oos is None:
            warn(f"Chưa có file <code>stress_oos_{OOS_KEYS[year]}_kupiec.csv</code> và <code>stress_oos_{OOS_KEYS[year]}_christoffersen.csv</code>.")
        else:
            html_bt(oos)
            if "n_obs" in oos.columns:
                n_days = int(oos["n_obs"].max())
                howto(f"{n_days} ngày đánh giá, kỳ vọng khoảng {d2(n_days * (1 - CONF), 1)} vi phạm mỗi mô hình. Mẫu nhỏ nên kiểm định có sức mạnh hạn chế: không bị bác bỏ không có nghĩa là mô hình đạt.")

    section("3. Stress 2008–2009", "Khủng hoảng tài chính toàn cầu, dùng bộ dữ liệu stress riêng.")
    st.caption("Giai đoạn 2008 dùng bộ dữ liệu 2007–2009 tách biệt với mẫu chính; thị trường Việt Nam thời đó có quy mô và cấu trúc khác nhiều, nên kết quả cần diễn giải thận trọng.")
    stress_block("2008", "stress_2008_detail.csv")

    section("4. Stress 2020", "Cú sốc COVID-19, giai đoạn biến động cao nhất của mẫu chính.")
    stress_block("2020", "stress_2020_detail.csv")

    section("5. Stress 2022", "Giai đoạn tăng lãi suất và căng thẳng thị trường.")
    stress_block("2022", "stress_2022_detail.csv")

    section("6. Ý nghĩa stress test", "Tổng hợp kết quả ngoài mẫu: tỷ lệ vi phạm và kết luận Kupiec (✖ bị bác bỏ, ✔ không bị bác bỏ).")
    if oos_all:
        periods_ = list(oos_all)
        plabel_ = [{"2008": "2008–2009"}.get(y, y) for y in periods_]
        models_ = list(dict.fromkeys(m for t in oos_all.values() for m in t["Phương pháp"].astype(str)))
        fo = go.Figure()
        for m in models_:
            ys_ = []
            for y in periods_:
                t_ = oos_all[y]
                r_ = t_[t_["Phương pháp"].astype(str) == m]
                ys_.append(float(r_["rate_pct"].iloc[0]) if len(r_) else None)
            fo.add_trace(go.Bar(x=plabel_, y=ys_, name=m, marker_color=MODEL_COLORS.get(m, C["blue"]),
                                text=[f"{d2(v)}%" if v is not None else "" for v in ys_], textposition="outside"))
        fo.add_hline(y=P0_PCT, line_dash="dash", line_color=C["ink"], annotation_text=f"Kỳ vọng {d2(P0_PCT, 1)}%")
        fo.update_layout(barmode="group", hovermode="closest")
        fo.update_yaxes(title="Tỷ lệ vi phạm (%)", ticksuffix="%")
        show(style_fig(fo, "Tỷ lệ vi phạm VaR ngoài mẫu theo giai đoạn và mô hình", 420))
        howto("Cột nằm trên đường nét đứt nghĩa là tỷ lệ vi phạm cao hơn mức kỳ vọng 2,5% trong giai đoạn đó.")
        rows_ = {}
        for y, t in oos_all.items():
            for _, r in t.iterrows():
                rows_.setdefault(str(r["Phương pháp"]), {})[y] = f"{d2(r['rate_pct'])}% {'✖' if r['reject_H0'] else '✔'}"
        summ = pd.DataFrame(rows_).T.reindex(columns=list(oos_all)).fillna("—").reset_index().rename(columns={"index": "Mô hình"})
        html_table(summ)
    howto("Stress test không nhằm chứng minh mô hình nào “tốt nhất”. Mục đích là cho thấy mức rủi ro và tỷ lệ vi phạm thay đổi thế nào khi thị trường chuyển sang trạng thái bất thường; "
          "mẫu mỗi giai đoạn nhỏ nên mọi kết luận cần thận trọng.")

# ============================================================
# TAB 7: KẾT LUẬN
# ============================================================
with tabs[6]:
    section("1. Kết quả chính", "Mỗi thẻ tóm tắt một khối kết quả gắn với ba câu hỏi nghiên cứu; số liệu tính trực tiếp từ kết quả.")
    # VaR
    if not rolling_var.empty:
        vm = {l: pd.to_numeric(rolling_var[c], errors="coerce").abs().mean() for c, l in VAR_COLS[:3] if c in rolling_var.columns}
        var_body = "VaR trung bình: " + "; ".join(f"{l} {pct(x)}" for l, x in vm.items()) + ". "
        if "Historical" in vm and "Parametric" in vm and vm["Historical"] > vm["Parametric"]:
            var_body += "Historical cao hơn Parametric, phù hợp với đuôi dày. "
        var_body += "Parametric và Monte Carlo gần như trùng nhau vì cùng giả định chuẩn."
    else:
        var_body = "Cần file rolling_var_975.csv."
    # GARCH & ES
    es_body = "Cần file garch_var_es_rolling_975.csv."
    if not garch_out.empty and {"VaR_GARCH", "ES_97_5"}.issubset(garch_out.columns):
        gv, ge = garch_out["VaR_GARCH"].abs().mean(), garch_out["ES_97_5"].abs().mean()
        es_body = f"GARCH-VaR trung bình {pct(gv)}, ES {pct(ge)} (gấp {d2(ge / gv)} lần do giả định chuẩn). "
        if len(returns):
            hv_, he_ = hist_var_es(returns)
            es_body += (f"ES lịch sử gấp {d2(he_ / hv_)} lần VaR lịch sử, "
                        "cho thấy mức tổn thất ở phần đuôi cao hơn đáng kể so với ngưỡng VaR. ")
        es_body += "ES chưa được backtest."
    # Backtesting
    bt_head, bt_body, bt_col = "Chưa có kết quả backtest", "Cần file kupiec_results.csv và christoffersen_results.csv.", C["muted"]
    sel_txt = "Kết hợp Historical VaR và GARCH-VaR để giám sát, bổ sung ES và stress test để đánh giá rủi ro đuôi."
    if not bt.empty and BT_NEED.issubset(bt.columns):
        ok_k = bt.loc[~bt["reject_H0"], "Phương pháp"].tolist()
        ok_i = bt.loc[~bt["reject_ind"], "Phương pháp"].tolist()
        ok_c = bt.loc[~bt["reject_cc"], "Phương pháp"].tolist()
        bt_head = "Không mô hình nào đạt cả hai tiêu chí" if not ok_c else f"{len(ok_c)}/{len(bt)} mô hình đạt cả hai tiêu chí"
        bt_col = C["coral"] if not ok_c else C["amber"]
        bt_body = (f"Đạt Kupiec: {', '.join(ok_k) or 'không mô hình nào'}. Đạt độc lập: {', '.join(ok_i) or 'không mô hình nào'}. "
                   "Mô hình có tỷ lệ vi phạm gần 2,5% vẫn có thể có vi phạm dồn cụm.")
        best_rate = bt.loc[(bt["rate_pct"] - P0_PCT).abs().idxmin(), "Phương pháp"]
        sel_txt = (f"{best_rate} có tỷ lệ vi phạm gần kỳ vọng nhất; đạt kiểm định độc lập: {', '.join(ok_i) or 'không mô hình nào'}; "
                   f"đạt CC: {len(ok_c)}/{len(bt)}. Nên kết hợp các mô hình bổ sung nhau (VaR lịch sử kèm VaR có điều kiện) "
                   "thay vì chọn một mô hình duy nhất, và bổ sung ES, stress test.")
    # Stress
    st_head, st_body, st_col = "Chưa có kết quả stress test", "Cần file stress_oos_* và stress_test_comparison.csv.", C["muted"]
    buffer_txt = "Chưa có kết quả stress ngoài mẫu; vẫn nên dành đệm vốn cho rủi ro mô hình."
    ws = worst_stress()
    if oos_all:
        total = sum(len(t) for t in oos_all.values())
        fail = sum(int(t["reject_H0"].sum()) for t in oos_all.values())
        st_head = "Tỷ lệ vi phạm tăng trong các giai đoạn stress" if fail == total else "Độ bền vững khác nhau theo giai đoạn"
        st_col = C["coral"] if fail == total else C["amber"]
        st_body = f"Ngoài mẫu, {fail}/{total} lượt kiểm định (mô hình x giai đoạn) bị Kupiec bác bỏ. "
        if ws:
            st_body += f"{ws[0]} là giai đoạn có {ws[1].replace(' (%)', '')} cao nhất ({fmt_metric(ws[2], ws[1], ws[3])}). "
        st_body += "Mẫu nhỏ nên cần diễn giải thận trọng."
        buffer_txt = (f"Ngoài mẫu, {fail}/{total} lượt kiểm định (mô hình x giai đoạn) bị Kupiec bác bỏ"
                      + (": mọi mô hình đều bị vượt thường xuyên hơn kỳ vọng khi stress, cần đệm vốn cho rủi ro mô hình."
                         if fail == total else "; mức vi phạm khác nhau theo giai đoạn nhưng vẫn cần đệm vốn cho rủi ro mô hình."))
    elif ws:
        st_head, st_col = f"{ws[0]} là cú sốc nặng nhất", C["amber"]
        st_body = f"Theo {ws[1]} ({fmt_metric(ws[2], ws[1], ws[3])})."
    rq1_body, rq2_body = var_body, es_body
    if not bt.empty and BT_NEED.issubset(bt.columns):
        rq1_body += f" Backtest: {bt_head[0].lower() + bt_head[1:]}. {bt_body}"
        nl_ = bt["Phương pháp"].astype(str).str.lower()
        g_, p_ = bt[nl_.str.contains("garch")], bt[nl_.str.contains("parametric")]
        if len(g_) and len(p_):
            g_, p_ = g_.iloc[0], p_.iloc[0]
            rq2_body += (f" So với Parametric (biến động không điều kiện): tỷ lệ vi phạm GARCH {d2(g_['rate_pct'])}% so với {d2(p_['rate_pct'])}%; "
                         f"tính độc lập GARCH {ind_txt(g_)}, Parametric {ind_txt(p_)}.")
    r1, r2, r3 = st.columns(3)
    with r1: text_card("RQ1 — Ba phương pháp VaR", "Ba cách tính, ba mức rủi ro", rq1_body, C["blue"], "xl")
    with r2: text_card("RQ2 — GARCH và ES", "Phản ứng nhanh với biến động; ES còn dựa trên giả định chuẩn", rq2_body, C["teal"], "xl")
    with r3: text_card("RQ3 — Stress test", st_head, st_body, st_col, "xl")

    section("2. Hàm ý quản trị rủi ro")
    html_table(pd.DataFrame({
        "Đối tượng": ["Ngân hàng"] * 5 + ["Nhà đầu tư"] * 3,
        "Nội dung": ["Không dựa vào một mô hình", "Lựa chọn mô hình", "Bổ sung ES và stress test", "Đệm vốn cho rủi ro mô hình", "Backtesting định kỳ",
                     "Không dùng giả định chuẩn", "Theo dõi biến động có điều kiện", "Rủi ro tỷ giá và đa dạng hóa"],
        "Hàm ý": ["Historical VaR cung cấp mức tham chiếu theo phân phối thực nghiệm; GARCH-VaR phản ánh biến động có điều kiện; ES bổ sung thông tin về mức độ tổn thất vượt VaR.",
                  sel_txt,
                  "ES 97,5% theo tinh thần FRTB (nghiên cứu không xây dựng hệ thống tính vốn FRTB đầy đủ); stress test để xác định vốn và thanh khoản.",
                  buffer_txt,
                  "Theo dõi đồng thời số vi phạm, tính độc lập và quy mô vi phạm, không chỉ làm một lần.",
                  "So với Historical VaR và ES; ngày lỗ nặng thường lớn hơn nhiều so với VaR.",
                  "Khi biến động tăng đột biến, đánh giá lại tỷ trọng và đòn bẩy thay vì chờ VaR cửa sổ trượt cập nhật.",
                  "Cân nhắc phòng ngừa tỷ giá tùy chi phí; kiểm tra tương quan và đóng góp rủi ro trước khi dựa vào tỷ trọng 50/50."],
    }))

    section("3. Hạn chế của nghiên cứu")
    l1, l2, l3 = st.columns(3)
    with l1: list_card("Dữ liệu", ["VN-Index và S&amp;P 500 đóng cửa lệch múi giờ nên lợi suất cùng ngày không cùng tập thông tin.",
                                    "Inner Join loại các ngày không có đủ dữ liệu; chuỗi tỷ giá có nhiều ngày giá không đổi.",
                                    "Mẫu stress ngắn nên kiểm định có sức mạnh hạn chế."], C["blue"], "xl")
    with l2: list_card("Mô hình", ["Parametric và Monte Carlo dùng phân phối chuẩn; Monte Carlo không thêm thông tin đuôi.",
                                    "GARCH(1,1) đối xứng, nhiễu chuẩn, chưa thử Student-t hay EGARCH/GJR.",
                                    "ES tính theo phân phối chuẩn và chưa được backtest.",
                                    "Chưa kiểm tra độ nhạy theo cửa sổ và mức tin cậy."], C["teal"], "xl")
    with l3: list_card("Danh mục và stress", ["Tỷ trọng cố định 50:50, chưa xét tái cân bằng, chi phí giao dịch, thanh khoản.",
                                               "Stress 2008 dùng mẫu riêng, không so sánh trực tiếp với hai giai đoạn còn lại.",
                                               "Stress dựa trên giai đoạn lịch sử, chưa có kịch bản giả định."], C["amber"], "xl")

    section("4. Hướng nghiên cứu tiếp theo")
    list_card("Mở rộng mô hình và kiểm định", [
        "Dùng phân phối Student-t hoặc Skewed-t cho phần dư GARCH để phản ánh đuôi dày.",
        "Mở rộng sang GJR-GARCH hoặc EGARCH để nắm bắt tính bất đối xứng của biến động.",
        "Bổ sung Filtered Historical Simulation làm phương pháp đối chứng.",
        "Backtest ES và kiểm tra độ nhạy theo độ dài cửa sổ và mức tin cậy.",
    ], C["violet"], "m")

    section("5. Tái lập kết quả")
    text_card("Quy trình", "Chạy lại toàn bộ trong năm bước",
              "<b>Bước 1:</b> clone repository. <b>Bước 2:</b> cài thư viện trong <code>requirements.txt</code>. "
              f"<b>Bước 3:</b> dùng dữ liệu đã chốt ngày {DATA_FREEZE_DATE} trong <code>data/</code>. "
              "<b>Bước 4:</b> mở notebook <code>notebooks/market_risk_analysis.ipynb</code> và chạy toàn bộ để tạo các bảng và biểu đồ trong <code>outputs/</code>. "
              "<b>Bước 5:</b> chạy dashboard để đọc các kết quả trong <code>outputs/</code>.<br>"
              f"Repository: <b>{REPO_URL.replace('https://', '')}</b>", C["blue"], "m")

    section("6. Kết luận")
    takeaway("Đo lường rủi ro thị trường nên theo nhiều lớp: VaR xác định ngưỡng tổn thất, ES bổ sung thông tin về phần đuôi, "
             "backtesting kiểm tra khả năng dự báo và stress test đánh giá phản ứng khi thị trường căng thẳng.")
    warn("Kết quả phản ánh cấu hình nghiên cứu hiện tại của nhóm. Không nên dùng riêng một chỉ tiêu hoặc một mô hình để kết luận về rủi ro của danh mục.")

st.markdown(
    f'<div class="footer">Dữ liệu chốt ngày {DATA_FREEZE_DATE} &nbsp;|&nbsp; '
    f'<a href="{REPO_URL}" target="_blank">market-risk-measurement</a> '
    f'&nbsp;|&nbsp; Đề án 03: Đo lường rủi ro thị trường</div>',
    unsafe_allow_html=True,
)