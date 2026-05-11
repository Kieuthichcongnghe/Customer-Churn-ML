
# Phân Tích Customer Churn — Telco

> **Sinh viên:** Phạm Ngọc Thế Kiều  
> **MSSV:** 2321003593  
> **Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Mô tả bài toán

Dự đoán khách hàng nào có khả năng **rời bỏ dịch vụ viễn thông (churn)** trong kỳ tới, nhằm triển khai can thiệp giữ chân có mục tiêu trước khi quyết định rời bỏ xảy ra.

- **Churn:** Khách hàng ngừng sử dụng dịch vụ trong kỳ quan sát (`Churn = "Yes"`).
- **Ý nghĩa kinh doanh:** Chi phí giữ chân khách hàng cũ thấp hơn 5–7 lần so với thu hút khách mới.

---

## Dataset

| Thông tin | Chi tiết |
|-----------|---------|
| Nguồn | Telco Customer Churn (Kaggle) |
| Kích thước | 7.043 dòng × 21 cột |
| Sau làm sạch | 7.032 dòng × 30 features (sau encoding) |
| Tỷ lệ Churn | ~26.5% (mất cân bằng lớp) |

**Các nhóm đặc trưng:**
- Nhân khẩu học: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Dịch vụ: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- Hợp đồng & thanh toán: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- Thời gian gắn bó: `tenure`

---

## Cấu trúc dự án

```
├── Source_code.ipynb                  # Notebook chính
├── PhamNgocTheKieu_2321003503.ipynb   # Notebook cá nhân
├── Telco-Customer-Churn.csv           # Dataset gốc
├── test.csv                           # Dữ liệu test
├── File py tạo dữ liệu test.py        # Script tạo dữ liệu test
├── Báo cáo Máy học và trí tuệ nhân tạo.pdf  # Báo cáo PDF
└── README.md
```

---

## Quy trình thực hiện

```
1. Import thư viện
2. Đọc dữ liệu
3. Kiểm tra cấu trúc & chất lượng dữ liệu
4. Phân tích phân bố biến mục tiêu (Churn)
5. Làm sạch dữ liệu
6. EDA (Exploratory Data Analysis)
7. Tiền xử lý & chia tập Train/Test (80/20, stratify)
8. Huấn luyện & tối ưu siêu tham số (GridSearchCV)
9. So sánh & lựa chọn mô hình
10. Phân tích ROC, Confusion Matrix, Feature Importance
11. Tổng kết insight & đề xuất kinh doanh
```

---

## Các mô hình sử dụng

| Mô hình | CV Recall | Recall (Churn) | F1 (Churn) | AUC |
|---------|-----------|----------------|------------|-----|
| Logistic Regression | 0.8033 | 0.7968 | 0.6075 | 0.8350 |
| KNN | 0.5318 | 0.5535 | 0.5565 | 0.7924 |
| SVM | 0.8368 | 0.8155 | 0.5804 | 0.8115 |
| **Random Forest** | **0.8074** | **0.7995** | **0.6115** | **0.8364** |

> **Metric chính:** Recall và AUC — không phải Accuracy. Mô hình luôn đoán "Không churn" vẫn đạt ~74% accuracy nhưng hoàn toàn vô dụng trong thực tế.

**Mô hình được chọn: Random Forest** — Recall và AUC cao nhất, overfit gap thấp, hỗ trợ feature importance.

**Hiệu năng trên tập test (Random Forest):**
- Recall (Churn): ~0.80
- Precision (Churn): ~0.50
- F1-score (Churn): ~0.61
- AUC: ~0.836

---

## Thư viện sử dụng

```python
numpy, pandas, matplotlib, seaborn
scikit-learn:
  - LogisticRegression, SVC, RandomForestClassifier, KNeighborsClassifier
  - GridSearchCV, train_test_split, learning_curve
  - StandardScaler
  - confusion_matrix, classification_report, roc_auc_score, roc_curve
```

---

## Cách chạy

1. Cài đặt thư viện:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Đặt file `Telco-Customer-Churn.csv` cùng thư mục với notebook.

3. Mở và chạy `Source_code.ipynb` theo thứ tự từ trên xuống.

---

## Insight chính

| # | Insight |
|---|---------|
| 1 | **`tenure`** là yếu tố quan trọng nhất — khách hàng mới (0–12 tháng) có nguy cơ churn cao nhất |
| 2 | **Hợp đồng dài hạn** (1 năm, 2 năm) gắn với tỷ lệ churn thấp hơn rõ rệt so với hợp đồng tháng |
| 3 | **Fiber optic** + phí cao + tenure thấp = tổ hợp rủi ro churn mạnh nhất |
| 4 | Khách dùng **Electronic check** có tỷ lệ churn cao hơn các phương thức tự động |
| 5 | **OnlineSecurity, TechSupport** giúp giữ chân khách hàng — vừa là cơ hội upsell |

## Đề xuất hành động

| # | Hành động | Nhóm mục tiêu |
|---|-----------|--------------|
| 1 | Chương trình onboarding & chăm sóc sớm (3–6 tháng đầu) | Khách tenure thấp |
| 2 | Ưu đãi chuyển sang hợp đồng dài hạn | Khách hợp đồng tháng |
| 3 | Tối ưu chiến lược giá & value proposition | Khách MonthlyCharges cao |
| 4 | Theo dõi riêng nhóm Fiber optic | Khách Fiber optic |
| 5 | Khuyến khích chuyển phương thức thanh toán tự động | Khách dùng Electronic check |
| 6 | Bundle dịch vụ TechSupport & OnlineSecurity | Khách chưa dùng add-on |
