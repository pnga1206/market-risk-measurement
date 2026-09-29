# ĐO LƯỜNG RỦI RO THỊ TRƯỜNG: VaR, ES VÀ KIỂM ĐỊNH NGƯỢC

## 1. Tổng quan

Dự án thực hiện đo lường và đánh giá rủi ro thị trường đối với danh mục kết hợp **VN-Index và S&P 500**.

Mục tiêu chính:

* Đo lường tổn thất tiềm ẩn bằng các phương pháp VaR.
* Ước lượng biến động và rủi ro bằng GARCH(1,1) và Expected Shortfall (ES).
* Đánh giá chất lượng dự báo VaR thông qua backtesting.
* Phân tích khả năng dự báo của các mô hình trong những giai đoạn thị trường căng thẳng.

Toàn bộ quy trình được thực hiện bằng Python và Jupyter Notebook, với mã nguồn được tổ chức theo từng nhóm chức năng nhằm đảm bảo khả năng tái lập kết quả.

---

## 2. Dữ liệu

### Dữ liệu chính

* **Thời gian:** 01/01/2016 – 25/09/2026
* **Tần suất:** dữ liệu ngày
* **Danh mục:** VN-Index và S&P 500
* **Tỷ trọng:** 50% VN-Index và 50% S&P 500
* **Lợi suất:** log return
* **Phương pháp ghép dữ liệu:** Inner Join
* **Ngày đóng băng dữ liệu:** 25/09/2026

| Dữ liệu                 | Nguồn         |
| ----------------------- | ------------- |
| VN-Index                | DNSE OpenAPI  |
| S&P 500                 | FRED          |
| USD/VND                 | Yahoo Finance |
| S&P 500 cho stress 2008 | Yahoo Finance |

### Dữ liệu stress 2008

Đối với phân tích khủng hoảng tài chính toàn cầu, dự án sử dụng bộ dữ liệu riêng từ **01/01/2007 đến 31/12/2009**.

Dữ liệu năm 2007 được giữ lại để tạo rolling window trước khi đánh giá dự báo trong giai đoạn 2008–2009.

---

## 3. Phương pháp

Các thiết lập chính:

* **Mức tin cậy:** 97,5%
* **Horizon:** 1 ngày
* **Rolling window:** 250 phiên
* **Monte Carlo:** 10.000 mô phỏng
* **Random seed:** 42

### VaR

Ba phương pháp VaR được sử dụng:

* Historical VaR
* Parametric VaR
* Monte Carlo VaR

### GARCH và ES

Mô hình **GARCH(1,1)** được sử dụng để dự báo biến động có điều kiện, từ đó tính **GARCH VaR** và **ES 97,5%**.

### Backtesting

Khả năng dự báo VaR được đánh giá bằng:

* **Kupiec Test:** kiểm định tỷ lệ vi phạm VaR.
* **Christoffersen Test:** kiểm định tính độc lập của các vi phạm và conditional coverage.

Tại mức tin cậy 97,5%, tỷ lệ vi phạm kỳ vọng là **2,5%**.

### Stress Test

Stress test gồm hai phần:

**In-sample stress analysis**

* 2008–2009
* 2020
* 2022

**Out-of-sample stress backtesting**

* 2008–2009
* 2020
* 2022

Phần in-sample được sử dụng để mô tả mức độ nghiêm trọng của rủi ro trong các giai đoạn stress. Phần OOS được sử dụng để đánh giá khả năng dự báo VaR của các mô hình trong các giai đoạn này.

---

## 4. Cấu trúc project

```text
market-risk-measurement/
├── data/
│   ├── main/
│   └── stress_2008/
│
├── notebooks/
│   └── market_risk_analysis.ipynb
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── backtesting/
│   ├── garch/
│   ├── stress_test/
│   ├── var/
│   └── config.py
│
├── requirements.txt
└── README.md
```

---

## 5. Hướng dẫn tái lập

### Cách 1: Google Colab

Mở trực tiếp notebook trên Google Colab:

**[▶ Mở notebook trên Google Colab](https://colab.research.google.com/github/pnga1206/market-risk-measurement/blob/main/notebooks/market_risk_analysis.ipynb)**

Sau khi notebook được mở trên Google Colab, chọn:

**Runtime → Run all**

Notebook sẽ tự động:

1. Clone repository từ GitHub.
2. Cài đặt các thư viện từ `requirements.txt`.
3. Đọc dữ liệu local trong repository.
4. Kiểm tra và chuẩn bị dữ liệu.
5. Chuẩn bị chuỗi lợi suất danh mục.
6. Tính Historical, Parametric và Monte Carlo VaR.
7. Tính Rolling VaR.
8. Ước lượng GARCH(1,1) và ES.
9. Thực hiện backtesting bằng Kupiec và Christoffersen.
10. Thực hiện stress test và stress OOS backtesting.
11. Lưu các kết quả vào thư mục `outputs/`.

Repository ở chế độ public nên không cần GitHub Personal Access Token để clone.

### Cách 2: VS Code / Jupyter

Yêu cầu: **Python và Git**.

Clone repository:

```bash
git clone https://github.com/pnga1206/market-risk-measurement.git
cd market-risk-measurement
```

Cài đặt thư viện:

```bash
python -m pip install -r requirements.txt
```

Mở notebook:

```text
notebooks/market_risk_analysis.ipynb
```

Sau đó chọn **Run All** để chạy toàn bộ quy trình.

---

## 6. Kết quả đầu ra

Kết quả phân tích được lưu trong thư mục `outputs/`. Một số kết quả tổng hợp và biểu đồ được lưu cùng repository để thuận tiện cho việc kiểm tra, trong khi các bảng dữ liệu chi tiết được tạo lại khi chạy notebook.

### Figures

Các biểu đồ được tạo trong quá trình phân tích gồm:

* Rolling VaR 97,5%
* Phân phối lợi suất trong các giai đoạn stress
* Diễn biến lợi suất trong các giai đoạn stress
* So sánh VaR giữa các giai đoạn
* Tỷ lệ vi phạm VaR
* Tỷ lệ vi phạm trong stress OOS

### Tables

Các bảng kết quả gồm:

* Backtest summary
* Kupiec results
* Christoffersen results
* Stress comparison
* Stress detail cho 2008–2009, 2020 và 2022
* OOS backtesting results cho từng giai đoạn stress

Các bảng dữ liệu chi tiết có kích thước lớn như rolling VaR, GARCH VaR/ES và dữ liệu vi phạm VaR được tạo lại khi chạy notebook và không được lưu vào repository.


## 7. Tái lập và quản lý tham số

Các tham số chính được tập trung tại `src/config.py` để đảm bảo toàn bộ phân tích sử dụng cùng một thiết lập.

Notebook mặc định sử dụng dữ liệu local nhằm đảm bảo kết quả có thể tái lập ổn định. Khi cần cập nhật dữ liệu, có thể chuyển sang chế độ lấy dữ liệu trực tiếp từ các nguồn tương ứng.

Các tham số, ngày đóng băng dữ liệu và random seed được cố định trong phạm vi nghiên cứu nhằm hạn chế khác biệt khi chạy lại.
