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
        'RELATION_DIM': 64,  # Thêm tham số cho Relation Network
        'NUM_EPISODES': 10,
        'SAVE_RESULTS': True,
        'COMPARE_WITHOUT_AUG': False,
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
        'DISPLAY_PROGRESS': True,
        'SAVE_PLOTS': True,
        'PLOT_DPI': 300,
        'SHOW_PLOTS': False,  # Thêm tham số cho hiển thị plots
        'USE_CUDA': True,
        'USE_VALIDATION': True,
        'Q_VALID': 3,
        'DETAILED_ANALYSIS': False,
        'SAVE_DETAILED_PLOTS': True
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
    print(f"   Compare without augmentation: {config['COMPARE_WITHOUT_AUG']}")
    print(f"   Device: {'cuda' if config['USE_CUDA'] else 'cpu'}")
    print(f"   Image size: {config['IMAGE_SIZE']}x{config['IMAGE_SIZE']}")
    print(f"   Embedding dim: {config['EMBED_DIM']}")
    print(f"   Relation Network dim: {config['RELATION_DIM']}")
    print(f"   Show plots: {config.get('SHOW_PLOTS', False)}")
    print("=" * 60)
