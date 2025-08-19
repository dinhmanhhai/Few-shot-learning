# ==== CẤU HÌNH FEW-SHOT LEARNING ====

# Đường dẫn dataset
DATASET_PATH = r'D:\AI\New-Dataset'

# Tham số Few-Shot Learning (đã tối ưu hóa)
N_WAY = 10         # Số class trong mỗi episode (sử dụng tất cả 10 class)
K_SHOT = 3         # Số ảnh support mỗi class (tăng từ 1 lên 3)
Q_QUERY = 3        # Số ảnh query mỗi class (giữ nguyên 5)
Q_VALID = 3        # Số ảnh validation mỗi class (mới thêm)

# Tham số mô hình
EMBED_DIM = 512    # Kích thước embedding
IMAGE_SIZE = 224   # Kích thước ảnh input
RELATION_DIM = 64  # Kích thước relation network hidden layer

# Cấu hình phép đo khoảng cách
DISTANCE_METHOD = "euclidean"  # "euclidean" hoặc "relation_network"
USE_LEARNABLE_METRIC = False  # True = Relation Network, False = Euclidean Distance

# Tham số training
NUM_EPISODES = 100  # Số episodes để chạy
SAVE_RESULTS = True  # Lưu kết quả
USE_AUGMENTATION = True  # Bật/tắt data augmentation (True = có, False = không)

# Tham số validation
USE_VALIDATION = True  # Sử dụng validation set

# Cấu hình data augmentation
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
    'grayscale_probability': 0.1  # Xác suất chuyển grayscale
}

# Cấu hình class-specific augmentation
CLASS_AUGMENTATION = {
    'enable_selective': True,  # Bật/tắt augment theo class cụ thể
    'augment_classes': [1, 2, 9],  # Chỉ augment các class này (0-based index)
    'skip_classes': [0, 3, 4, 5, 6, 7, 8],     # Bỏ qua augment cho các class này
    'augment_ratio': 1.5,                 # Tỷ lệ augment (1.5 = tăng 50% số ảnh)
    'min_images_per_class': 5             # Số ảnh tối thiểu mỗi class sau khi augment
}

# Cấu hình hiển thị
DISPLAY_PROGRESS = True  # Hiển thị tiến độ
SAVE_PLOTS = True        # Lưu đồ thị
PLOT_DPI = 300          # Độ phân giải đồ thị
SHOW_PLOTS = False      # Không hiển thị ảnh pop-up (chỉ lưu file)

# Cấu hình phân tích
DETAILED_ANALYSIS = True  # Phân tích chi tiết dataset (cần để có class_distribution)

# Tham số device
USE_CUDA = True        # Sử dụng GPU nếu có
