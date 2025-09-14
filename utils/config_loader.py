"""
Module để load và quản lý cấu hình
"""
import os
import datetime
import sys
from pathlib import Path

def load_config():
    """
    Load cấu hình từ config.py hoặc sử dụng giá trị mặc định
    """
    # Cấu hình mặc định
    default_config = {
        'DATASET_PATH': r'D:\AI\Dataset',
        'N_WAY': 5,
        'K_SHOT': 1,
        'Q_QUERY': 5,
        'EMBED_DIM': 512,
        'IMAGE_SIZE': 224,
        'TRANSFORMER_MODEL': 'swin_base_patch4_window7_224',
        'RELATION_DIM': 64,  # Thêm tham số cho Relation Network
        'DISTANCE_METHOD': 'relation_network',  # Phương pháp đo khoảng cách
        'USE_LEARNABLE_METRIC': True,  # Sử dụng metric có thể học được
        'NUM_EPISODES': 10,
        'SAVE_RESULTS': True,
        'USE_AUGMENTATION': False,  # Bật/tắt data augmentation
        'AUGMENTATION_CONFIG': {
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
        },
        'CLASS_AUGMENTATION': {
            'enable_selective': True,
            'augment_classes': [1, 2, 9],
            'skip_classes': [0, 3, 4, 5, 6, 7, 8],
            'augment_ratio': 1.5,
            'min_images_per_class': 5
        },
        'SHOW_PLOTS': False,  # Thêm tham số cho hiển thị plots
        'DISPLAY_PROGRESS': True,  # Hiển thị tiến độ
        'SAVE_PLOTS': True,        # Lưu đồ thị
        'PLOT_DPI': 300,          # Độ phân giải đồ thị
        'USE_CUDA': True,
        'USE_VALIDATION': True,
        'Q_VALID': 3,
        'DETAILED_ANALYSIS': False
    }
    
    # Thử load từ config.py
    config_path = Path('config.py')
    if config_path.exists():
        try:
            # Đọc nội dung file config.py
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Tạo namespace để execute config
            config_namespace = {}
            exec(config_content, config_namespace)
            
            # Cập nhật default_config với các giá trị từ config.py
            for key, value in config_namespace.items():
                if key.isupper():
                    # Thêm key mới nếu chưa có trong default_config
                    if key not in default_config:
                        default_config[key] = value
                    else:
                        default_config[key] = value
            
            print("✅ Đã tải cấu hình từ config.py")
        except Exception as e:
            print(f"⚠️ Lỗi khi tải config.py: {e}")
            print("⚠️ Sử dụng cấu hình mặc định")
    else:
        print("⚠️ Không tìm thấy config.py, sử dụng cấu hình mặc định")

    # Giới hạn chỉ hỗ trợ Swin/ConvNeXt và đồng bộ IMAGE_SIZE
    allowed_models = {
        'swin_base_patch4_window7_224',
        'swin_large_patch4_window12_384',
        'convnext_base',
        'convnext_large'
    }
    image_size_map = {
        'swin_base_patch4_window7_224': 224,
        'swin_large_patch4_window12_384': 384,
        'convnext_base': 224,
        'convnext_large': 224
    }

    cfg_model = default_config.get('TRANSFORMER_MODEL', 'swin_base_patch4_window7_224')
    if cfg_model not in allowed_models:
        print(f"⚠️ Model '{cfg_model}' không được hỗ trợ. Chỉ hỗ trợ Swin/ConvNeXt.")
        cfg_model = 'swin_base_patch4_window7_224'
        default_config['TRANSFORMER_MODEL'] = cfg_model
        print(f"➡️ Tự động chuyển sang: {cfg_model}")

    target_img_size = image_size_map.get(cfg_model, 224)
    if default_config.get('IMAGE_SIZE', 224) != target_img_size:
        print(f"ℹ️ IMAGE_SIZE ({default_config.get('IMAGE_SIZE')}) không khớp với model '{cfg_model}'.")
        default_config['IMAGE_SIZE'] = target_img_size
        print(f"➡️ Tự động đặt IMAGE_SIZE = {target_img_size}")
    
    # Thiết lập output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_config['OUTPUT_FOLDER'] = f"few_shot_results_{timestamp}"
    os.makedirs(default_config['OUTPUT_FOLDER'], exist_ok=True)
    print(f"📁 Tạo output folder: {default_config['OUTPUT_FOLDER']}")
    
    return default_config

def print_config_summary(config):
    """
    In tóm tắt cấu hình
    """
    print("📋 THÔNG TIN CẤU HÌNH:")
    print(f"   Dataset: {config['DATASET_PATH']}")
    print(f"   Few-Shot: {config['N_WAY']}-way, {config['K_SHOT']}-shot, {config['Q_QUERY']}-query, {config['Q_VALID']}-valid")
    print(f"   Episodes: {config['NUM_EPISODES']}")
    print(f"   Use validation: {config['USE_VALIDATION']}")
    print(f"   Use augmentation: {config.get('USE_AUGMENTATION', False)}")
    if config.get('USE_AUGMENTATION', False):
        class_aug = config.get('CLASS_AUGMENTATION', {})
        if class_aug.get('enable_selective', False):
            print(f"   Selective augmentation: BẬT")
            print(f"   Augment classes: {class_aug.get('augment_classes', [])}")
            print(f"   Skip classes: {class_aug.get('skip_classes', [])}")
            print(f"   Augment ratio: {class_aug.get('augment_ratio', 1.0)}x")
        else:
            print(f"   Selective augmentation: TẮT (augment tất cả class)")
    print(f"   Device: {'cuda' if config['USE_CUDA'] else 'cpu'}")
    print(f"   Image size: {config['IMAGE_SIZE']}x{config['IMAGE_SIZE']}")
    print(f"   Embedding dim: {config['EMBED_DIM']}")
    print(f"   Relation Network dim: {config['RELATION_DIM']}")
    print(f"   Transformer: {config.get('TRANSFORMER_MODEL', 'swin_base_patch4_window7_224')}")
    print(f"   Distance method: {config.get('DISTANCE_METHOD', 'relation_network')}")
    print(f"   Learnable metric: {config.get('USE_LEARNABLE_METRIC', True)}")
    print(f"   Show plots: {config.get('SHOW_PLOTS', False)}")
    print(f"   Display progress: {config.get('DISPLAY_PROGRESS', True)}")
    print(f"   Save plots: {config.get('SAVE_PLOTS', True)}")
    print(f"   Plot DPI: {config.get('PLOT_DPI', 300)}")
    print("=" * 60)
