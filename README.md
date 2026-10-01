# ĐO LƯỜNG RỦI RO THỊ TRƯỜNG: VaR, ES VÀ KIỂM ĐỊNH NGƯỢC

## 1. Tổng quan

Dự án thực hiện đo lường và đánh giá **rủi ro thị trường** đối với danh mục kết hợp **VN-Index và S&P 500**.

Mục tiêu chính:

* Đo lường tổn thất tiềm ẩn bằng các phương pháp VaR.
* Ước lượng biến động và rủi ro bằng GARCH(1,1) và Expected Shortfall (ES).
* Đánh giá chất lượng dự báo VaR thông qua backtesting.
* Phân tích khả năng dự báo của các mô hình trong những giai đoạn thị trường căng thẳng.

Toàn bộ quy trình được thực hiện bằng **Python và Jupyter Notebook**. Dữ liệu, mã nguồn và kết quả nghiên cứu được lưu trữ trong repository nhằm phục vụ kiểm tra và tái lập.

---

## 2. Dữ liệu

### 2.1. Dữ liệu chính

Bộ dữ liệu chính được sử dụng trong giai đoạn **01/01/2016 – 25/09/2026**, với tần suất dữ liệu ngày.

| Thiết lập    | Giá trị             |
| ------------ | ------------------- |
| Danh mục     | VN-Index và S&P 500 |
| Tỷ trọng     | 50% – 50%           |
| Lợi suất     | Log return          |
| Ghép dữ liệu | Inner Join          |
| Data Freeze  | 25/09/2026          |

| Dữ liệu                 | Nguồn         |
| ----------------------- | ------------- |
| VN-Index                | DNSE OpenAPI  |
| S&P 500                 | FRED          |
| USD/VND                 | Yahoo Finance |
| S&P 500 cho stress 2008 | Yahoo Finance |

### 2.2. Dữ liệu stress 2008–2009

Đối với phân tích khủng hoảng tài chính toàn cầu, dự án sử dụng bộ dữ liệu riêng từ **01/01/2007 đến 31/12/2009**.

Dữ liệu năm 2007 được giữ lại để tạo rolling window trước khi đánh giá dự báo trong giai đoạn 2008–2009.

---

## 3. Phương pháp

### 3.1. VaR

Ba phương pháp VaR được sử dụng:

* Historical VaR
* Parametric VaR
* Monte Carlo VaR

### 3.2. GARCH và ES

Mô hình **GARCH(1,1)** được sử dụng để dự báo biến động có điều kiện, từ đó tính **GARCH VaR** và **ES 97,5%**.

### 3.3. Backtesting

Khả năng dự báo VaR được đánh giá bằng:

* **Kupiec Test:** kiểm định tỷ lệ vi phạm VaR.
* **Christoffersen Test:** kiểm định tính độc lập của các vi phạm và conditional coverage.

Tại mức tin cậy 97,5%, tỷ lệ vi phạm kỳ vọng là **2,5%**.

### 3.4. Stress Test

Stress test được thực hiện đối với ba giai đoạn:

* **2008–2009**
* **2020**
* **2022**

Phân tích gồm:

* **In-sample stress analysis:** đánh giá mức độ rủi ro trong các giai đoạn thị trường căng thẳng.
* **Out-of-sample stress backtesting:** đánh giá khả năng dự báo VaR của các mô hình trong các giai đoạn stress.

### 3.5. Thiết lập chính

| Tham số        | Giá trị         |
| -------------- | --------------- |
| Mức tin cậy    | 97,5%           |
| Horizon        | 1 ngày          |
| Rolling window | 250 phiên       |
| Monte Carlo    | 10.000 mô phỏng |
| Random seed    | 42              |

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
│   ├── config.py
│   ├── backtesting/
│   ├── data/
│   ├── garch/
│   ├── stress_test/
│   └── var/
│
├── requirements.txt
└── README.md
```

---

## 5. Kết quả nghiên cứu

Các kết quả nghiên cứu được trình bày trực tiếp trong notebook và được lưu vào thư mục `outputs/`.

### 5.1. Figures

Các biểu đồ chính gồm:

* Rolling VaR 97,5%
* Phân phối lợi suất trong các giai đoạn stress
* Diễn biến lợi suất trong các giai đoạn stress
* So sánh VaR giữa các giai đoạn
* Tỷ lệ vi phạm VaR
* Tỷ lệ vi phạm trong stress OOS

### 5.2. Tables

Các bảng kết quả gồm:

* Backtest summary
* Backtest violations
* Kupiec results
* Christoffersen results
* Stress test comparison
* Stress detail cho 2008–2009, 2020 và 2022
* OOS backtesting results cho từng giai đoạn stress

Các file trong `outputs/` được lưu để phục vụ kiểm tra, đối chiếu và sử dụng trong báo cáo nghiên cứu.

---

## 6. Hướng dẫn tái lập nghiên cứu

### Bước 1. Clone repository

```bash
git clone https://github.com/pnga1206/market-risk-measurement.git
cd market-risk-measurement
```

### Bước 2. Cài đặt thư viện

```bash
python -m pip install -r requirements.txt
```

### Bước 3. Mở notebook

Mở file:

```text
notebooks/market_risk_analysis.ipynb
```

### Bước 4. Xóa output cũ

Để chạy lại toàn bộ nghiên cứu từ đầu, chọn:

**Clear All Outputs**

Bước này chỉ xóa các kết quả đang hiển thị trong notebook, không xóa dữ liệu hoặc các file kết quả trong thư mục `outputs/`.

### Bước 5. Chạy toàn bộ notebook

Chọn:

**Run All**

Notebook sẽ thực hiện toàn bộ quy trình phân tích từ chuẩn bị dữ liệu, xây dựng danh mục, tính VaR, GARCH và ES, backtesting đến stress test.

Các kết quả sẽ:

* **Hiển thị trực tiếp trong notebook** trong quá trình chạy.
* **Được lưu vào `outputs/`** theo các file kết quả tương ứng.

### Bước 6. Kiểm tra kết quả

Sau khi notebook chạy hoàn tất, kiểm tra:

```text
outputs/
├── figures/
└── tables/
```

Các kết quả tạo ra có thể được đối chiếu với các kết quả nghiên cứu đã được lưu trong repository.

---

