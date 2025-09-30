# 📋 HƯỚNG DẪN SỬ DỤNG CONFIG

## 🎯 Cách thay đổi số episodes

### 1. Chỉnh sửa file `config.py`:

```python
# Thay đổi số episodes từ 10 thành 20
NUM_EPISODES = 20  # Số lượng episodes để test
```

### 2. Chạy lại chương trình:

```bash
python main.py
```

## ⚙️ Các tham số có thể tùy chỉnh

### **Few-Shot Learning:**
```python
N_WAY = 5          # Số lượng class trong mỗi episode
K_SHOT = 1         # Số lượng ảnh support mỗi class  
Q_QUERY = 5        # Số lượng ảnh query mỗi class
```

### **Mô hình:**
```python
EMBED_DIM = 512    # Kích thước embedding
IMAGE_SIZE = 224   # Kích thước ảnh đầu vào
```

### **Đánh giá:**
```python
NUM_EPISODES = 10  # Số lượng episodes để test
SAVE_RESULTS = True  # Có lưu kết quả không
```

### **Data Augmentation:**
```python
AUGMENTATION_CONFIG = {
    'random_crop_size': 32,      # Kích thước crop thêm
    'rotation_degrees': 15,      # Góc xoay tối đa
    'flip_probability': 0.5,     # Xác suất lật ảnh
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'grayscale_probability': 0.1  # Xác suất chuyển grayscale
}
```

### **Hiển thị:**
```python
DISPLAY_PROGRESS = True  # Hiển thị tiến độ
SAVE_PLOTS = True       # Lưu đồ thị
PLOT_DPI = 300         # Độ phân giải đồ thị
```

### **Device:**
```python
USE_CUDA = True        # Sử dụng GPU nếu có
```

### **Phân tích Dataset:**
```python
# Phân tích dataset được thực hiện tự động
SAVE_DETAILED_PLOTS = True  # Lưu đồ thị chi tiết
```

## 🚀 Ví dụ cấu hình

### **Test nhanh (5 episodes):**
```python
NUM_EPISODES = 5
DISPLAY_PROGRESS = True
SAVE_RESULTS = False
```

### **Test chi tiết (50 episodes):**
```python
NUM_EPISODES = 50
DISPLAY_PROGRESS = False  # Không hiển thị tiến độ để chạy nhanh hơn
SAVE_RESULTS = True
```

### **Data augmentation mạnh:**
```python
AUGMENTATION_CONFIG = {
    'random_crop_size': 64,
    'rotation_degrees': 30,
    'flip_probability': 0.7,
    'color_jitter': {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.4,
        'hue': 0.2
    },
    'grayscale_probability': 0.2
}
```

### **Data augmentation nhẹ:**
```python
AUGMENTATION_CONFIG = {
    'random_crop_size': 16,
    'rotation_degrees': 10,
    'flip_probability': 0.3,
    'color_jitter': {
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.05
    },
    'grayscale_probability': 0.05
}
```

## 📊 Kết quả

Sau khi chạy, bạn sẽ nhận được:

### **Phân tích dataset tự động:**
- `dataset_analysis/detailed_analysis.png`: Phân tích chi tiết (6 đồ thị)
- `dataset_analysis/file_formats_analysis.png`: Phân tích định dạng file

### **Kết quả chung:**
- `episode_results.png`: So sánh kết quả với/không có augmentation
- Thống kê chi tiết trong console

### **Chạy phân tích riêng:**
```bash
python run_dataset_analysis.py
```

## ⚠️ Lưu ý

- Nếu không có file `config.py`, chương trình sẽ sử dụng cấu hình mặc định
- Thay đổi `NUM_EPISODES` càng lớn thì thời gian chạy càng lâu
- Đảm bảo dataset có đủ dữ liệu cho cấu hình few-shot learning
