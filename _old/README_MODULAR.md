# 📁 Cấu Trúc Modular cho Few-Shot Learning

## 🎯 Tổng Quan

Code đã được tách thành các module nhỏ để dễ quản lý và bảo trì hơn. Mỗi module có trách nhiệm riêng biệt và có thể được sử dụng độc lập.

## 📂 Cấu Trúc Thư Mục

```
BMDGAN-tutorial-sim/
├── 📁 config/
│   └── config.py                    # Cấu hình chính
├── 📁 utils/
│   ├── __init__.py
│   ├── config_loader.py             # Load và quản lý cấu hình
│   └── transforms.py                # Data transforms và augmentation
├── 📁 models/
│   ├── __init__.py
│   └── backbone.py                  # Model backbone và distance metrics
├── 📁 data/
│   ├── __init__.py
│   └── dataset.py                   # Dataset và episode sampling
├── 📁 analysis/
│   ├── __init__.py
│   └── dataset_analysis.py          # Phân tích dataset
├── 📁 evaluation/
│   ├── __init__.py
│   └── metrics.py                   # Evaluation metrics
├── 📁 visualization/
│   ├── __init__.py
│   └── plots.py                     # Visualization plots
├── 📁 training/
│   ├── __init__.py
│   └── episode_runner.py            # Episode training và evaluation
├── main.py                          # File cũ (để tham khảo)
├── main_new.py                      # File mới với cấu trúc modular
└── README_MODULAR.md                # File này
```

## 🔧 Chi Tiết Từng Module

### **1. 📁 config/**
- **config.py**: Chứa tất cả cấu hình cho few-shot learning
  - Tham số N_WAY, K_SHOT, Q_QUERY, Q_VALID
  - Cấu hình data augmentation
  - Tham số model và training
  - Cấu hình hiển thị và lưu kết quả

### **2. 📁 utils/**
- **config_loader.py**: 
  - `load_config()`: Load cấu hình từ file hoặc sử dụng mặc định
  - `print_config_summary()`: In tóm tắt cấu hình
- **transforms.py**:
  - `create_transforms()`: Tạo transforms cho training, validation, inference

### **3. 📁 models/**
- **backbone.py**:
  - `TransformerBackbone`: Vision Transformer backbone
  - `euclidean_distance()`: Tính khoảng cách Euclidean
  - `compute_prototypes()`: Tính prototypes cho từng class

### **4. 📁 data/**
- **dataset.py**:
  - `FewShotDataset`: Dataset cho few-shot learning
  - Episode sampling với support, query, validation sets
  - Data augmentation integration

### **5. 📁 analysis/**
- **dataset_analysis.py**:
  - `analyze_and_visualize_dataset()`: Phân tích và vẽ đồ thị dataset
  - Thống kê số lượng ảnh theo class
  - Kiểm tra balance của dataset

### **6. 📁 evaluation/**
- **metrics.py**:
  - `calculate_detailed_metrics()`: Tính precision, recall, F1-score
  - `analyze_imbalance_impact()`: Phân tích ảnh hưởng imbalance
  - `print_detailed_evaluation_metrics()`: In metrics chi tiết

### **7. 📁 visualization/**
- **plots.py**:
  - `plot_confusion_matrix()`: Vẽ confusion matrix
  - `analyze_accuracy_by_class()`: Vẽ accuracy theo class
  - `plot_imbalance_analysis()`: Vẽ phân tích imbalance
  - `plot_episode_results()`: Vẽ kết quả episodes
  - `plot_single_results()`: Vẽ kết quả đơn lẻ

### **8. 📁 training/**
- **episode_runner.py**:
  - `run_episode_with_detailed_evaluation()`: Chạy một episode
  - `run_multiple_episodes_with_detailed_evaluation()`: Chạy nhiều episodes

## 🚀 Cách Sử Dụng

### **1. Chạy với cấu trúc mới:**
```bash
python main_new.py
```

### **2. Sử dụng từng module riêng lẻ:**

#### **Load cấu hình:**
```python
from utils.config_loader import load_config
config = load_config()
```

#### **Tạo transforms:**
```python
from utils.transforms import create_transforms
transform_basic, transform_augmented, transform_inference = create_transforms(config)
```

#### **Khởi tạo model:**
```python
from models.backbone import TransformerBackbone
model = TransformerBackbone(out_dim=config['EMBED_DIM'])
```

#### **Tạo dataset:**
```python
from data.dataset import FewShotDataset
dataset = FewShotDataset(config['DATASET_PATH'], transform_train, transform_test)
```

#### **Phân tích dataset:**
```python
from analysis.dataset_analysis import analyze_and_visualize_dataset
dataset_info = analyze_and_visualize_dataset(config['DATASET_PATH'], config)
```

#### **Tính metrics:**
```python
from evaluation.metrics import calculate_detailed_metrics
metrics = calculate_detailed_metrics(predictions, targets, n_classes)
```

#### **Vẽ đồ thị:**
```python
from visualization.plots import plot_confusion_matrix
plot_confusion_matrix(cm, class_names, "confusion_matrix.png", config)
```

#### **Chạy episodes:**
```python
from training.episode_runner import run_multiple_episodes_with_detailed_evaluation
results = run_multiple_episodes_with_detailed_evaluation(model, dataset, config, num_episodes)
```

## ✅ Lợi Ích Của Cấu Trúc Modular

### **1. Dễ Bảo Trì:**
- Mỗi module có trách nhiệm rõ ràng
- Dễ sửa đổi từng phần mà không ảnh hưởng phần khác
- Code sạch và có tổ chức

### **2. Tái Sử Dụng:**
- Có thể import và sử dụng từng module riêng lẻ
- Dễ dàng tích hợp vào project khác
- Không cần chạy toàn bộ pipeline

### **3. Testing:**
- Có thể test từng module riêng biệt
- Dễ debug khi có lỗi
- Unit testing cho từng function

### **4. Mở Rộng:**
- Dễ thêm tính năng mới
- Có thể thay thế module mà không ảnh hưởng module khác
- Flexible architecture

## 🔄 Migration từ File Cũ

### **Để chuyển từ main.py sang main_new.py:**

1. **Backup file cũ:**
   ```bash
   cp main.py main_backup.py
   ```

2. **Chạy file mới:**
   ```bash
   python main_new.py
   ```

3. **So sánh kết quả:**
   - Kết quả sẽ giống hệt nhau
   - Output files được tạo trong cùng format
   - Console output tương tự

## 🛠️ Tùy Chỉnh

### **Thêm Module Mới:**
1. Tạo thư mục mới trong package tương ứng
2. Tạo `__init__.py`
3. Import trong `main_new.py`

### **Thay Đổi Cấu Hình:**
- Chỉ cần sửa `config.py`
- Không cần sửa code logic

### **Thay Đổi Model:**
- Sửa `models/backbone.py`
- Các module khác không bị ảnh hưởng

### **Thay Đổi Evaluation:**
- Sửa `evaluation/metrics.py`
- Thêm metrics mới dễ dàng

## 📊 So Sánh

| Aspect | File Cũ (main.py) | File Mới (main_new.py) |
|--------|-------------------|------------------------|
| **Kích thước** | ~1400 dòng | ~300 dòng |
| **Modularity** | Không | Có |
| **Maintainability** | Khó | Dễ |
| **Reusability** | Không | Có |
| **Testing** | Khó | Dễ |
| **Functionality** | Giống nhau | Giống nhau |

## 🎯 Kết Luận

Cấu trúc modular mới giúp:
- **Quản lý code dễ dàng hơn**
- **Bảo trì và mở rộng thuận tiện**
- **Tái sử dụng code hiệu quả**
- **Testing và debugging đơn giản**

Tất cả chức năng của file cũ đều được giữ nguyên, chỉ được tổ chức lại một cách có hệ thống hơn.
