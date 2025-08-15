# ==== CẤU HÌNH FEW-SHOT LEARNING ====

# Đường dẫn dataset
DATASET_PATH = r'D:\AI\Dataset'

# Tham số Few-Shot Learning (đã tối ưu hóa)
N_WAY = 10         # Số class trong mỗi episode (sử dụng tất cả 10 class)
K_SHOT = 3         # Số ảnh support mỗi class (tăng từ 1 lên 3)
Q_QUERY = 20        # Số ảnh query mỗi class (giữ nguyên 5)
Q_VALID = 10        # Số ảnh validation mỗi class (mới thêm)

# Tham số mô hình
EMBED_DIM = 512    # Kích thước embedding
IMAGE_SIZE = 224   # Kích thước ảnh input
RELATION_DIM = 64  # Kích thước relation network hidden layer

# Tham số training
NUM_EPISODES = 100  # Số episodes để chạy
SAVE_RESULTS = True  # Lưu kết quả
COMPARE_WITHOUT_AUG = False  # So sánh với/không augmentation

# Tham số validation
USE_VALIDATION = True  # Sử dụng validation set
VALIDATION_EPISODES = 10  # Số episodes cho validation
SAVE_VALIDATION_RESULTS = True  # Lưu kết quả validation

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

# Cấu hình hiển thị
DISPLAY_PROGRESS = True  # Hiển thị tiến độ
SAVE_PLOTS = True        # Lưu đồ thị
PLOT_DPI = 300          # Độ phân giải đồ thị
SHOW_PLOTS = False      # Không hiển thị ảnh pop-up (chỉ lưu file)

# Cấu hình phân tích
DETAILED_ANALYSIS = False  # Phân tích chi tiết dataset
SAVE_DETAILED_PLOTS = True  # Lưu đồ thị phân tích chi tiết

# Tham số device
USE_CUDA = True        # Sử dụng GPU nếu có
