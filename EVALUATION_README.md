# 📊 HƯỚNG DẪN ĐÁNH GIÁ ĐỘ CHÍNH XÁC CHI TIẾT

## 🎯 Tổng quan

Code đã được bổ sung các tính năng đánh giá độ chính xác chi tiết cho few-shot learning, bao gồm:

### ✅ **Các Metrics Đã Có:**
- **Accuracy cơ bản**: Độ chính xác tổng thể
- **Loss**: Loss function cho query và validation sets
- **Thống kê**: Mean, std, min, max cho accuracy và loss

### 🆕 **Các Metrics Mới Được Bổ Sung:**

#### 1. **Precision, Recall, F1-Score**
- **Macro Average**: Trung bình đơn giản của tất cả classes
- **Weighted Average**: Trung bình có trọng số theo số lượng samples
- **Per-Class**: Metrics riêng cho từng class

#### 2. **Confusion Matrix**
- Ma trận nhầm lẫn chi tiết
- Hiển thị số lượng dự đoán đúng/sai cho từng class
- Visualization với heatmap

#### 3. **Accuracy by Class**
- Phân tích độ chính xác theo từng class riêng biệt
- Xác định classes nào dễ/hard nhất để phân loại
- Visualization với bar chart

#### 4. **Classification Report**
- Báo cáo chi tiết từ scikit-learn
- Precision, recall, F1-score cho từng class
- Support (số lượng samples) cho mỗi class

## 📈 Cách Sử Dụng

### 1. **Chạy với đánh giá chi tiết:**
```bash
python main.py
```

### 2. **Cấu hình trong config.py:**
```python
# Bật/tắt validation
USE_VALIDATION = True
Q_VALID = 3

# Bật/tắt so sánh với/không augmentation
COMPARE_WITHOUT_AUG = True

# Lưu kết quả
SAVE_RESULTS = True
```

## 📊 Output Files

### **📁 Output Folder Structure:**
```
few_shot_results_YYYYMMDD_HHMMSS/
├── 📊 Dataset Analysis
│   ├── dataset_analysis.png              # Phân tích cơ bản dataset
│   ├── detailed_analysis.png             # Phân tích chi tiết dataset
│   └── file_formats_analysis.png         # Phân tích định dạng file
├── 📈 Episode Results
│   ├── episode_results_single.png        # Kết quả episodes (có augmentation)
│   └── episode_results.png               # So sánh có/không augmentation
├── 🔍 Query Set Evaluation
│   ├── query_confusion_matrix.png        # Confusion matrix query set
│   ├── query_accuracy_by_class.png       # Accuracy theo class query set
│   ├── query_imbalance_analysis.png      # Phân tích ảnh hưởng imbalance
│   ├── query_confusion_matrix_no_aug.png # Confusion matrix không augmentation
│   ├── query_accuracy_by_class_no_aug.png # Accuracy theo class không augmentation
│   └── query_imbalance_analysis_no_aug.png # Imbalance analysis không augmentation
└── ✅ Validation Set Evaluation
    ├── valid_confusion_matrix.png        # Confusion matrix validation set
    ├── valid_accuracy_by_class.png       # Accuracy theo class validation set
    ├── valid_imbalance_analysis.png      # Imbalance analysis validation set
    ├── valid_confusion_matrix_no_aug.png # Confusion matrix validation không augmentation
    ├── valid_accuracy_by_class_no_aug.png # Accuracy theo class validation không augmentation
    └── valid_imbalance_analysis_no_aug.png # Imbalance analysis validation không augmentation
```

### **📋 Chi Tiết Từng File:**

#### **1. Dataset Analysis Files:**
- **`dataset_analysis.png`**: 
  - Bar chart số lượng ảnh theo class
  - Pie chart tỷ lệ phân bố
  - Horizontal bar chart top 10 classes
  - Statistics table tổng quan

- **`detailed_analysis.png`**:
  - 6 loại đồ thị phân tích chi tiết
  - Distribution histogram
  - Box plot
  - Statistics table nâng cao

- **`file_formats_analysis.png`**:
  - Phân bố định dạng file ảnh
  - Pie chart và bar chart định dạng

#### **2. Episode Results Files:**
- **`episode_results_single.png`**:
  - Accuracy/Loss theo episodes
  - Histogram phân bố accuracy
  - Statistics table kết quả

- **`episode_results.png`**:
  - So sánh có/không data augmentation
  - Box plot phân bố accuracy
  - Statistics table so sánh

#### **3. Query Set Evaluation Files:**
- **`query_confusion_matrix.png`**:
  - Ma trận nhầm lẫn cho query set
  - Heatmap với tên class thực tế
  - Hiển thị TP, FP, FN, TN

- **`query_accuracy_by_class.png`**:
  - Bar chart accuracy theo từng class
  - Đường trung bình accuracy
  - Xác định class dễ/khó phân loại

- **`query_imbalance_analysis.png`**:
  - Phân tích ảnh hưởng của dataset imbalance
  - F1-Score vs số lượng samples
  - So sánh minority vs majority classes
  - Precision vs Recall scatter plot

#### **4. Validation Set Evaluation Files:**
- **`valid_confusion_matrix.png`**:
  - Ma trận nhầm lẫn cho validation set
  - So sánh với query set để phát hiện overfitting

- **`valid_accuracy_by_class.png`**:
  - Accuracy theo class cho validation set
  - So sánh hiệu suất với query set

- **`valid_imbalance_analysis.png`**:
  - Imbalance analysis cho validation set
  - So sánh với query set để đánh giá stability

### **🎯 Điều Kiện Tạo Files:**

#### **Luôn được tạo:**
- `dataset_analysis.png` hoặc `detailed_analysis.png`
- `episode_results_single.png`
- `query_confusion_matrix.png`
- `query_accuracy_by_class.png`
- `query_imbalance_analysis.png`

#### **Khi COMPARE_WITHOUT_AUG = True:**
- `episode_results.png`
- `query_confusion_matrix_no_aug.png`
- `query_accuracy_by_class_no_aug.png`

#### **Khi USE_VALIDATION = True:**
- `valid_confusion_matrix.png`
- `valid_accuracy_by_class.png`
- `valid_imbalance_analysis.png`

#### **Khi cả hai đều True:**
- Tất cả files trên + validation versions

## 🔍 Cách Đọc và Phân Tích Output Files

### **📊 Dataset Analysis - Cách Đọc:**

#### **1. dataset_analysis.png:**
- **Bar Chart**: Classes có bar cao = nhiều ảnh, bar thấp = ít ảnh
- **Pie Chart**: Tỷ lệ % của từng class trong tổng dataset
- **Horizontal Bar**: Top 10 classes có nhiều ảnh nhất
- **Statistics Table**: 
  - Tổng số class và ảnh
  - Trung bình, min, max ảnh/class
  - Độ lệch chuẩn và tỷ lệ cân bằng

#### **2. detailed_analysis.png:**
- **Distribution Histogram**: Phân bố số lượng ảnh theo class
- **Box Plot**: Quartiles và outliers của số lượng ảnh
- **Statistics Table**: Thống kê chi tiết hơn với quartiles

### **📈 Episode Results - Cách Đọc:**

#### **1. episode_results_single.png:**
- **Accuracy Line**: Đường tăng = cải thiện, giảm = overfitting
- **Loss Line**: Đường giảm = tốt, tăng = có vấn đề
- **Histogram**: Phân bố accuracy của các episodes
- **Statistics Table**: Mean, std, min, max của accuracy/loss

#### **2. episode_results.png:**
- **So sánh 2 đường**: Augmentation vs No Augmentation
- **Box Plot**: Phân bố accuracy của 2 phương pháp
- **Statistics Table**: So sánh metrics giữa 2 phương pháp

### **🔍 Confusion Matrix - Cách Đọc:**

#### **Cấu trúc:**
```
        Predicted
Actual   A  B  C
   A    10 2  1  ← 10 đúng, 2 nhầm B, 1 nhầm C
   B     1 12 0  ← 1 nhầm A, 12 đúng, 0 nhầm C  
   C     0  1 11 ← 0 nhầm A, 1 nhầm B, 11 đúng
```

#### **Phân tích:**
- **Diagonal (đường chéo)**: True Positives (TP) - dự đoán đúng
- **Off-diagonal**: False Positives/Negatives (FP/FN) - dự đoán sai
- **Màu sắc**: Càng đậm = càng nhiều samples
- **Patterns**: 
  - Class nào có nhiều FP/FN = khó phân loại
  - Confusion giữa 2 classes = tương tự nhau

### **📊 Accuracy by Class - Cách Đọc:**

#### **Bar Chart:**
- **Bar cao**: Class có accuracy tốt
- **Bar thấp**: Class có accuracy kém
- **Đường trung bình**: So sánh với hiệu suất tổng thể
- **Số liệu trên bars**: Giá trị accuracy cụ thể

#### **Phân tích:**
- **Classes có accuracy cao**: Dễ phân loại, đặc trưng rõ ràng
- **Classes có accuracy thấp**: Khó phân loại, cần cải thiện
- **Gap với trung bình**: Classes có vấn đề đặc biệt

### **⚖️ Imbalance Analysis - Cách Đọc:**

#### **F1-Score vs Support:**
- **Trend line**: Mối quan hệ giữa số lượng samples và performance
- **Điểm đỏ**: Minority classes (ít samples)
- **Điểm xanh**: Majority classes (nhiều samples)
- **Độ dốc**: Càng dốc = imbalance càng ảnh hưởng

#### **Support Distribution:**
- **Bars đỏ**: Classes ít ảnh (minority)
- **Bars xanh**: Classes nhiều ảnh (majority)
- **Gap lớn**: Dataset mất cân bằng nghiêm trọng

#### **Minority vs Majority Box Plot:**
- **Box plot đỏ**: Performance của minority classes
- **Box plot xanh**: Performance của majority classes
- **Overlap ít**: Imbalance ảnh hưởng lớn

#### **Precision vs Recall Scatter:**
- **Điểm đỏ**: Minority classes thường có precision/recall thấp
- **Điểm xanh**: Majority classes thường có precision/recall cao
- **Góc trên phải**: Classes có performance tốt
- **Góc dưới trái**: Classes có performance kém

### **✅ Validation vs Query - So Sánh:**

#### **Overfitting Detection:**
- **Query > Validation**: Có thể bị overfitting
- **Query ≈ Validation**: Mô hình cân bằng
- **Query < Validation**: Có thể bị underfitting

#### **Class Performance:**
- **So sánh accuracy**: Classes nào ổn định/không ổn định
- **Pattern consistency**: Classes nào có pattern tương tự

## 🎯 Ứng Dụng Thực Tế

### **1. Phân Tích Dataset:**
- **Imbalanced classes**: Xác định classes thiếu dữ liệu
- **Data quality**: Phát hiện classes có vấn đề về chất lượng
- **Augmentation strategy**: Định hướng augmentation cho classes yếu

## ⚖️ Ảnh Hưởng Của Dataset Imbalance

### **🔍 Vấn Đề Chính:**

#### **1. Bias Towards Majority Classes:**
- **Mô hình thiên về classes có nhiều ảnh**
- **Classes ít ảnh bị "bỏ quên" hoặc dự đoán sai**
- **Accuracy tổng thể cao nhưng không phản ánh thực tế**

#### **2. Episodic Sampling Bias:**
- **Classes nhiều ảnh có nhiều khả năng được chọn**
- **Classes ít ảnh ít khi xuất hiện trong episodes**
- **Kết quả không đại diện cho toàn bộ dataset**

### **📊 Biểu Hiện Cụ Thể:**

#### **A. Trong Confusion Matrix:**
```
        Predicted
Actual   A  B  C
   A    50 2  1  ← Class A (nhiều ảnh) - dự đoán tốt
   B     1 8  0  ← Class B (ít ảnh) - dự đoán kém
   C     0  1 5  ← Class C (ít ảnh) - dự đoán kém
```

#### **B. Trong Metrics:**
- **Macro F1-score thấp** (do classes ít ảnh kéo xuống)
- **Weighted F1-score cao** (do thiên về classes nhiều ảnh)
- **Recall thấp cho minority classes**

#### **C. Trong Console Output:**
```
⚖️ PHÂN TÍCH ẢNH HƯỞNG IMBALANCE:
   Imbalance Ratio: 0.125
   Minority Classes (3): ['class_b', 'class_c', 'class_d']
   Minority Classes F1-Score TB: 0.4567
   Majority Classes (2): ['class_a', 'class_e']
   Majority Classes F1-Score TB: 0.8234
   Macro vs Weighted F1 Difference: 0.1567
   Balance Status: Mất cân bằng nghiêm trọng
   Impact Level: Rất cao
   ⚠️ CẢNH BÁO: Dataset mất cân bằng nghiêm trọng!
```

### **💡 Giải Pháp:**

#### **1. Data Augmentation:**
- **Tăng augmentation cho classes ít ảnh**
- **Sử dụng các kỹ thuật: rotation, flip, color jitter**
- **Tạo synthetic samples**

#### **2. Sampling Strategies:**
- **Oversampling**: Tăng số lượng samples cho minority classes
- **Undersampling**: Giảm số lượng samples cho majority classes
- **Balanced sampling**: Đảm bảo mỗi class có số samples tương đương

#### **3. Model Adjustments:**
- **Class weights**: Đặt trọng số cao hơn cho minority classes
- **Loss function**: Sử dụng focal loss hoặc balanced loss
- **Architecture**: Điều chỉnh model để xử lý imbalance

#### **4. Evaluation Metrics:**
- **Sử dụng Macro F1-score** thay vì accuracy
- **Xem xét cả precision và recall**
- **Phân tích per-class performance**

### **2. Tối Ưu Hóa Mô Hình:**
- **Hyperparameter tuning**: Dựa trên F1-score thay vì accuracy
- **Class-specific improvements**: Tập trung vào classes có recall thấp
- **Architecture changes**: Dựa trên confusion patterns

### **3. Báo Cáo Kết Quả:**
- **Academic papers**: Metrics chuyên nghiệp
- **Business reports**: Visualization dễ hiểu
- **Model comparison**: So sánh với baseline methods

## 🛠️ Troubleshooting & Best Practices

### **🔧 Các Vấn Đề Thường Gặp:**

#### **1. Output Folder Không Được Tạo:**
```bash
# Kiểm tra quyền ghi
ls -la few_shot_results_*

# Tạo thủ công nếu cần
mkdir -p few_shot_results_$(date +%Y%m%d_%H%M%S)
```

#### **2. Files Bị Thiếu:**
- **Kiểm tra config**: `SAVE_RESULTS = True`
- **Kiểm tra dependencies**: `scikit-learn`, `matplotlib`, `seaborn`
- **Kiểm tra disk space**: Đảm bảo đủ dung lượng

#### **3. Tên Class Không Hiển Thị Đúng:**
- **Vấn đề**: Hiển thị `Class_0, Class_1` thay vì tên thực tế
- **Giải pháp**: Code đã được cập nhật để hiển thị tên folder thực tế
- **Kiểm tra**: Xem console output có hiển thị tên class thực tế không

### **📈 Best Practices:**

#### **1. Phân Tích Kết Quả:**
- **Bắt đầu với dataset analysis**: Hiểu dữ liệu trước
- **So sánh Query vs Validation**: Phát hiện overfitting
- **Xem confusion matrix**: Tìm patterns và vấn đề
- **Phân tích accuracy by class**: Xác định classes khó

#### **2. Tối Ưu Hóa:**
- **Tăng N_WAY**: Sử dụng nhiều class hơn nếu có thể
- **Điều chỉnh K_SHOT**: Tăng số support images
- **Thử nghiệm augmentation**: So sánh có/không augmentation
- **Validation set**: Luôn sử dụng để tránh overfitting

#### **3. Báo Cáo:**
- **Sử dụng F1-score**: Metric tổng hợp tốt nhất
- **Include confusion matrix**: Visualization quan trọng
- **Class-specific analysis**: Chi tiết cho từng class
- **Comparison plots**: So sánh các phương pháp

### **📊 Metrics Priority:**

#### **Primary Metrics (Quan trọng nhất):**
1. **F1-Score (Macro)**: Đánh giá tổng thể
2. **Confusion Matrix**: Hiểu patterns
3. **Accuracy by Class**: Xác định vấn đề cụ thể

#### **Secondary Metrics (Bổ sung):**
1. **Precision/Recall**: Chi tiết hơn
2. **Validation vs Query**: Overfitting detection
3. **Episode progression**: Stability analysis

### **🎯 Tips cho Research:**

#### **1. Reproducibility:**
- **Save config**: Lưu tất cả parameters
- **Version control**: Track code changes
- **Environment**: Document dependencies

#### **2. Visualization:**
- **High DPI**: Sử dụng `PLOT_DPI = 300` cho papers
- **Consistent colors**: Sử dụng cùng color scheme
- **Clear labels**: Tên class rõ ràng

#### **3. Analysis:**
- **Statistical significance**: Chạy nhiều lần
- **Cross-validation**: Sử dụng different episodes
- **Ablation studies**: Test từng component

## 🔍 Giải Thích Metrics

### **Precision (Độ chính xác):**
```
Precision = TP / (TP + FP)
```
- Tỷ lệ dự đoán đúng trong số các dự đoán positive
- Cao = ít false positives

### **Recall (Độ bao phủ):**
```
Recall = TP / (TP + FN)
```
- Tỷ lệ dự đoán đúng trong số các samples thực sự positive
- Cao = ít false negatives

### **F1-Score:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Trung bình điều hòa của precision và recall
- Metric tổng hợp tốt nhất

### **Macro vs Weighted Average:**
- **Macro**: Trung bình đơn giản của tất cả classes
- **Weighted**: Trung bình có trọng số theo số lượng samples

## 📋 Console Output

### **Metrics Tổng Quan:**
```
🎯 METRICS TỔNG QUAN:
   Macro Precision: 0.8234
   Macro Recall: 0.8156
   Macro F1-Score: 0.8194
   Weighted Precision: 0.8256
   Weighted Recall: 0.8234
   Weighted F1-Score: 0.8245
```

### **Metrics Theo Từng Class:**
```
📈 METRICS THEO TỪNG CLASS:
   Class_0:
     Precision: 0.8500
     Recall: 0.8000
     F1-Score: 0.8235
     Support: 25
```

### **Xếp Hạng Hiệu Suất:**
```
🏆 XẾP HẠNG HIỆU SUẤT THEO F1-SCORE:
   1. Class_2: 0.8750
   2. Class_0: 0.8235
   3. Class_1: 0.8000
```

## ⚠️ Lưu Ý

### **Dependencies:**
- `scikit-learn>=1.3.0` - Cho các metrics chi tiết
- `matplotlib>=3.7.0` - Cho visualization
- `seaborn>=0.12.0` - Cho heatmap

### **Fallback:**
- Nếu không có scikit-learn, chỉ tính accuracy cơ bản
- Các metrics khác sẽ được set bằng accuracy

### **Performance:**
- Đánh giá chi tiết có thể làm chậm quá trình một chút
- Tuy nhiên cung cấp thông tin rất hữu ích cho phân tích

## 🎯 Ứng Dụng

### **1. Phân Tích Hiệu Suất:**
- Xác định classes khó phân loại
- Tìm patterns trong confusion matrix
- So sánh hiệu quả data augmentation

### **2. Tối Ưu Hóa Mô Hình:**
- Điều chỉnh hyperparameters dựa trên F1-score
- Cải thiện classes có recall thấp
- Cân bằng precision và recall

### **3. Báo Cáo Kết Quả:**
- Metrics chuyên nghiệp cho papers
- Visualization chất lượng cao
- So sánh với baseline methods

## 🔧 Tùy Chỉnh

### **Thêm Metrics Mới:**
```python
def custom_metric(predictions, targets):
    # Implement custom metric
    return metric_value
```

### **Thay Đổi Visualization:**
```python
def plot_custom_metric(data, save_path):
    # Custom plotting logic
    plt.savefig(save_path)
```

### **Export Results:**
```python
import json
with open('evaluation_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```
