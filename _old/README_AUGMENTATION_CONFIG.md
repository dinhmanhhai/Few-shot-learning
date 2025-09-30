# 🎛️ Cấu Hình Data Augmentation

## 📋 Biến Cấu Hình Mới: `USE_AUGMENTATION`

### 🔧 **Cách Sử Dụng:**

Trong file `config.py`, bạn có thể bật/tắt data augmentation bằng cách thay đổi giá trị:

```python
# Bật data augmentation
USE_AUGMENTATION = True

# Tắt data augmentation  
USE_AUGMENTATION = False
```

### 📊 **Ảnh Hưởng Khi Thay Đổi:**

#### **1. Khi `USE_AUGMENTATION = True`:**
- ✅ **Support images** sẽ được augment (RandomCrop, Rotation, Flip, ColorJitter, Grayscale)
- ✅ **Query images** và **Validation images** sử dụng transform cơ bản
- ✅ Hiển thị thống kê chi tiết về augmentation
- ✅ Thông báo: "CÓ AUGMENTATION"

#### **2. Khi `USE_AUGMENTATION = False`:**
- ❌ **Tất cả images** (Support, Query, Validation) sử dụng transform cơ bản
- ❌ Không có augmentation techniques
- ❌ Hiển thị thống kê: "KHÔNG CÓ AUGMENTATION"
- ❌ Thông báo: "KHÔNG AUGMENTATION"

### 🎯 **Ví Dụ Cấu Hình:**

```python
# config.py

# Cấu hình với augmentation
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'random_crop_size': 32,
    'rotation_degrees': 15,
    'flip_probability': 0.5,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'grayscale_probability': 0.1
}

# Cấu hình không có augmentation
USE_AUGMENTATION = False
# AUGMENTATION_CONFIG vẫn giữ nguyên để tham khảo
```

### 🔄 **Cách Hoạt Động:**

1. **Trong `utils/transforms.py`:**
   - Nếu `USE_AUGMENTATION = True`: Tạo `transform_augmented` với các techniques
   - Nếu `USE_AUGMENTATION = False`: Sử dụng `transform_basic` cho tất cả

2. **Trong `data/dataset.py`:**
   - Support images: Sử dụng `use_augmentation` parameter
   - Query/Validation images: Luôn sử dụng transform cơ bản

3. **Trong `main_new.py`:**
   - Hiển thị thống kê phù hợp
   - Thông báo trạng thái augmentation
   - Truyền `use_augmentation` parameter

### 📈 **Lợi Ích:**

- **Linh hoạt**: Dễ dàng bật/tắt augmentation mà không cần sửa code
- **So sánh**: Có thể chạy cả hai chế độ để so sánh kết quả
- **Debug**: Dễ dàng kiểm tra xem augmentation có ảnh hưởng gì
- **Production**: Có thể tắt augmentation trong production để tăng tốc độ

### 🚀 **Chạy Thử Nghiệm:**

```bash
# Chạy với augmentation
python main_new.py  # USE_AUGMENTATION = True

# Chạy không có augmentation  
python main_new.py  # USE_AUGMENTATION = False
```

### 📝 **Lưu Ý:**

- Khi `USE_AUGMENTATION = False`, tất cả images đều sử dụng cùng transform
- Support images vẫn có thể được augment nếu cần thiết
- Thống kê augmentation sẽ hiển thị 0 cho các metrics liên quan
- Output folder sẽ ghi rõ trạng thái augmentation
