# 🎯 Cấu Hình Phương Pháp Đo Khoảng Cách

## 📋 **Biến Cấu Hình Mới:**

### **1. `DISTANCE_METHOD`:**
```python
# config.py
DISTANCE_METHOD = "relation_network"  # "euclidean" hoặc "relation_network"
```

### **2. `USE_LEARNABLE_METRIC`:**
```python
# config.py
USE_LEARNABLE_METRIC = True  # True = Relation Network, False = Euclidean Distance
```

## 🔧 **Cách Sử Dụng:**

### **Option 1: Sử Dụng Relation Network (Có Thể Học Được)**
```python
# config.py
DISTANCE_METHOD = "relation_network"
USE_LEARNABLE_METRIC = True
```

**Kết quả:**
- ✅ **Relation Network**: Sử dụng CNN để học cách so sánh features
- ✅ **Có thể học**: Metric được tối ưu hóa trong quá trình training
- ✅ **Linh hoạt**: Có thể thích ứng với các loại dữ liệu khác nhau
- ✅ **Hiệu suất cao**: Thường cho kết quả tốt hơn Euclidean Distance

### **Option 2: Sử Dụng Euclidean Distance (Cố Định)**
```python
# config.py
DISTANCE_METHOD = "euclidean"
USE_LEARNABLE_METRIC = False
```

**Kết quả:**
- ✅ **Euclidean Distance**: Sử dụng công thức toán học cố định
- ✅ **Đơn giản**: Không cần training, chạy nhanh
- ✅ **Ổn định**: Kết quả nhất quán, không thay đổi
- ✅ **Ít tham số**: Tiết kiệm bộ nhớ

## 🧠 **Cách Hoạt Động:**

### **1. Relation Network:**
```
Input: Query Features + Support Features
↓
CNN Layers: Học cách so sánh features
↓
Fully Connected: Tạo relation scores
↓
Output: Scores từ 0-1 (cao hơn = tương tự hơn)
```

### **2. Euclidean Distance:**
```
Input: Query Features + Support Features
↓
Tính Euclidean Distance: sqrt(Σ(x1-x2)²)
↓
Chuyển thành Similarity: 1/(1+distance)
↓
Output: Scores từ 0-1 (cao hơn = tương tự hơn)
```

## 📊 **So Sánh Hai Phương Pháp:**

| Tiêu Chí | Relation Network | Euclidean Distance |
|----------|------------------|-------------------|
| **Khả năng học** | ✅ Có thể học | ❌ Cố định |
| **Tốc độ** | ⚠️ Chậm hơn | ✅ Nhanh |
| **Bộ nhớ** | ⚠️ Nhiều hơn | ✅ Ít hơn |
| **Hiệu suất** | ✅ Cao hơn | ⚠️ Thấp hơn |
| **Ổn định** | ⚠️ Có thể thay đổi | ✅ Nhất quán |
| **Linh hoạt** | ✅ Thích ứng | ❌ Cứng nhắc |

## 🚀 **Khuyến Nghị:**

### **Sử Dụng Relation Network Khi:**
- 🎯 **Mục đích**: Training và đạt hiệu suất cao
- 📊 **Dữ liệu**: Phức tạp, đa dạng
- ⏱️ **Thời gian**: Có đủ thời gian training
- 💾 **Tài nguyên**: GPU và bộ nhớ đủ

### **Sử Dụng Euclidean Distance Khi:**
- 🎯 **Mục đích**: Inference nhanh, đơn giản
- 📊 **Dữ liệu**: Đơn giản, có cấu trúc rõ ràng
- ⏱️ **Thời gian**: Cần kết quả ngay lập tức
- 💾 **Tài nguyên**: Hạn chế về GPU/bộ nhớ

## 🧪 **Test Và Demo:**

### **Chạy Với Relation Network:**
```bash
# config.py
DISTANCE_METHOD = "relation_network"
USE_LEARNABLE_METRIC = True

# Chạy
python main_new.py
```

**Kết quả mong đợi:**
```
🎯 Khởi tạo model với phương pháp: relation_network
   - Sử dụng Relation Network (có thể học được)
   - Relation dimension: 64

🧠 Kiến trúc: Vision Transformer + RELATION_NETWORK
🧠 Phương pháp: RELATION_NETWORK (Có thể học được)
📈 Relation scores: 0-1 (cao hơn = tương tự hơn)
```

### **Chạy Với Euclidean Distance:**
```bash
# config.py
DISTANCE_METHOD = "euclidean"
USE_LEARNABLE_METRIC = False

# Chạy
python main_new.py
```

**Kết quả mong đợi:**
```
🎯 Khởi tạo model với phương pháp: euclidean
   - Sử dụng Euclidean Distance (cố định)

🧠 Kiến trúc: Vision Transformer + EUCLIDEAN
🧠 Phương pháp: EUCLIDEAN (Cố định)
📈 Euclidean similarity: 0-1 (cao hơn = tương tự hơn)
```

## 🔄 **Chuyển Đổi Giữa Hai Phương Pháp:**

### **Trong Quá Trình Development:**
```python
# Thử nghiệm Relation Network
DISTANCE_METHOD = "relation_network"
USE_LEARNABLE_METRIC = True

# Chạy và đánh giá
python main_new.py

# Chuyển sang Euclidean Distance để so sánh
DISTANCE_METHOD = "euclidean"
USE_LEARNABLE_METRIC = False

# Chạy lại và so sánh kết quả
python main_new.py
```

### **Trong Production:**
```python
# Sử dụng Relation Network cho training
DISTANCE_METHOD = "relation_network"
USE_LEARNABLE_METRIC = True

# Sau khi training xong, chuyển sang Euclidean cho inference
DISTANCE_METHOD = "euclidean"
USE_LEARNABLE_METRIC = False
```

## 📝 **Lưu Ý Quan Trọng:**

### **1. Tương Thích:**
- Cả hai phương pháp đều sử dụng cùng **Transformer backbone**
- **Input/Output format** giống nhau
- Có thể **chuyển đổi** mà không cần thay đổi code khác

### **2. Hiệu Suất:**
- **Relation Network**: Cần training, nhưng hiệu suất cao hơn
- **Euclidean Distance**: Không cần training, nhưng hiệu suất thấp hơn

### **3. Bộ Nhớ:**
- **Relation Network**: Cần lưu thêm CNN layers
- **Euclidean Distance**: Chỉ cần lưu backbone

### **4. Tốc Độ:**
- **Relation Network**: Chậm hơn do có thêm CNN layers
- **Euclidean Distance**: Nhanh hơn do chỉ tính toán đơn giản

## 🎯 **Kết Luận:**

Hệ thống mới cho phép bạn **linh hoạt chọn** giữa hai phương pháp:

1. **Relation Network**: Cho hiệu suất cao, có thể học được
2. **Euclidean Distance**: Cho tốc độ nhanh, đơn giản

Bạn có thể **dễ dàng chuyển đổi** giữa hai phương pháp bằng cách thay đổi cấu hình trong `config.py`!
