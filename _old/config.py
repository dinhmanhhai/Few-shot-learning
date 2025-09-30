"""
Cấu hình Few-Shot Learning với hỗ trợ đa dạng Transformer models
"""

# =============================================================================
# ĐƯỜNG DẪN VÀ THAM SỐ CƠ BẢN
# =============================================================================

# Đường dẫn dataset chính để tìm kiếm
DATASET_PATH = r'D:\AI\Tea_Leaf_Disease'

# =============================================================================
# THAM SỐ FEW-SHOT LEARNING
# =============================================================================

# Cấu hình episode
N_WAY = 6          # Số class trong mỗi episode (phù hợp với dataset có 6 class)
K_SHOT = 5         # Số ảnh support mỗi class
Q_QUERY = 8        # Số ảnh query mỗi class
Q_VALID = 2        # Số ảnh validation mỗi class

# =============================================================================
# THAM SỐ TRAINING VÀ EVALUATION
# =============================================================================

NUM_EPISODES = 200      # Số episodes để chạy 
SAVE_RESULTS = True    # Lưu kết quả
USE_VALIDATION = True  # Sử dụng validation set

# =============================================================================
# THAM SỐ MÔ HÌNH
# =============================================================================

# Kích thước và embedding
EMBED_DIM = 512    # Kích thước embedding vector
IMAGE_SIZE = 224   # Kích thước ảnh input (pixels)
RELATION_DIM = 64  # Kích thước relation network hidden layer

# =============================================================================
# CẤU HÌNH TRANSFORMER BACKBONE
# =============================================================================

# Model được hỗ trợ:
# - Swin: "swin_base_patch4_window7_224", "swin_large_patch4_window12_384"
# - ConvNeXt: "convnext_base", "convnext_large"
# - ViT: "vit_base_patch16_224", "vit_large_patch16_224", "vit_huge_patch14_224"
# - DeiT: "deit_base_patch16_224", "deit_large_patch16_224"
TRANSFORMER_MODEL = "vit_base_patch16_224"

# =============================================================================
# CẤU HÌNH PHƯƠNG PHÁP ĐO KHOẢNG CÁCH
# =============================================================================

DISTANCE_METHOD = "euclidean"      # "euclidean" hoặc "relation_network"
USE_LEARNABLE_METRIC = False       # True = Relation Network, False = Euclidean

# Pre-trained model cho Relation Network
# Các lựa chọn: "mobilenet_v2", "efficientnet_b0", "resnet18"
RELATION_PRETRAINED_MODEL = "mobilenet_v2"

# =============================================================================
# CẤU HÌNH DATA AUGMENTATION
# =============================================================================

USE_AUGMENTATION = True  # Bật/tắt data augmentation

# Cấu hình augmentation cơ bản
AUGMENTATION_CONFIG = {
    'random_crop_size': 224,        # Tăng từ 32 lên 40
    'rotation_degrees': 20,        # Tăng từ 15 lên 20
    'flip_probability': 0.6,       # Tăng từ 0.5 lên 0.6
    'color_jitter': {
        'brightness': 0.25,        # Tăng từ 0.2 lên 0.25
        'contrast': 0.25,          # Tăng từ 0.2 lên 0.25
        'saturation': 0.25,        # Tăng từ 0.2 lên 0.25
        'hue': 0.15                # Tăng từ 0.1 lên 0.15
    },
    'grayscale_probability': 0.15  # Tăng từ 0.1 lên 0.15
}

# Cấu hình augmentation theo class cụ thể
CLASS_AUGMENTATION = {
    'enable_selective': True,           # Bật/tắt augment theo class
    'augment_classes': [0, 1, 2, 3, 4, 5],       # Class sẽ được augment (0-based index)
    'skip_classes': [],  # Class bỏ qua augment
    'augment_ratio': 3.3,               # Tỷ lệ augment (1.5 = tăng 50%)
    'min_images_per_class': 10           # Số ảnh tối thiểu mỗi class
}

# =============================================================================
# CẤU HÌNH HIỂN THỊ VÀ LƯU TRỮ
# =============================================================================

DISPLAY_PROGRESS = True   # Hiển thị tiến độ
SAVE_PLOTS = True         # Lưu đồ thị
PLOT_DPI = 300           # Độ phân giải đồ thị
SHOW_PLOTS = False       # Hiển thị ảnh pop-up (False = chỉ lưu file)

# =============================================================================
# CẤU HÌNH HARDWARE
# =============================================================================

USE_CUDA = True  # Sử dụng GPU nếu có
