"""
Main script cho Few-Shot Learning với cấu trúc modular
"""
import torch
import json
import os
from datetime import datetime

# Import các module
from utils.config_loader import load_config, print_config_summary
from utils.transforms import create_transforms
from models.backbone import RelationNetworkModel
from data.dataset import FewShotDataset
from analysis.dataset_analysis import analyze_and_visualize_dataset
from evaluation.metrics import calculate_detailed_metrics, print_detailed_evaluation_metrics
from visualization.plots import (
    plot_confusion_matrix, 
    analyze_accuracy_by_class, 
    plot_single_results
)
from training.episode_runner import run_multiple_episodes_with_detailed_evaluation

def save_config_to_output_folder(config, output_folder, aug_stats=None):
    """
    Lưu cấu hình vào file JSON trong folder kết quả
    """
    config_file = os.path.join(output_folder, "config.json")
    
    # Tạo bản sao của config để loại bỏ các object không thể serialize
    config_to_save = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            config_to_save[key] = value
        else:
            config_to_save[key] = str(value)  # Chuyển thành string nếu không thể serialize
    
    # Thêm timestamp
    config_to_save['timestamp'] = datetime.now().isoformat()
    config_to_save['output_folder'] = output_folder
    
    # Thêm thống kê augmentation nếu có
    if aug_stats:
        config_to_save['augmentation_stats'] = aug_stats
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Cấu hình đã được lưu vào: {config_file}")

def calculate_augmentation_stats(config, dataset_info):
    """
    Tính toán thống kê về data augmentation
    """
    total_original_images = dataset_info['total_images']
    n_way = config['N_WAY']
    k_shot = config['K_SHOT']
    q_query = config['Q_QUERY']
    q_valid = config['Q_VALID'] if config['USE_VALIDATION'] else 0
    num_episodes = config['NUM_EPISODES']
    
    # Số ảnh gốc được sử dụng trong mỗi episode
    images_per_episode = n_way * (k_shot + q_query + q_valid)
    
    # Tổng số ảnh gốc được sử dụng
    total_original_used = images_per_episode * num_episodes
    
    # Tính toán hiệu quả augmentation
    # Mỗi ảnh support (k_shot) sẽ được augment
    augmented_images_per_episode = n_way * k_shot
    total_augmented_images = augmented_images_per_episode * num_episodes
    
    # Tổng số ảnh sau augmentation (gốc + augmented)
    total_images_after_aug = total_original_used + total_augmented_images
    
    # Tỷ lệ augmentation
    augmentation_ratio = total_augmented_images / total_original_used if total_original_used > 0 else 0
    
    stats = {
        'total_original_images': total_original_images,
        'images_per_episode': images_per_episode,
        'total_original_used': total_original_used,
        'augmented_images_per_episode': augmented_images_per_episode,
        'total_augmented_images': total_augmented_images,
        'total_images_after_aug': total_images_after_aug,
        'augmentation_ratio': augmentation_ratio,
        'effective_dataset_size': total_images_after_aug
    }
    
    return stats

def print_augmentation_stats(aug_stats, config):
    """
    In thống kê augmentation
    """
    print("\n📊 THỐNG KÊ DATA AUGMENTATION:")
    print("=" * 50)
    print(f"📈 Tổng ảnh gốc trong dataset: {aug_stats['total_original_images']:,}")
    print(f"🎯 Ảnh sử dụng mỗi episode: {aug_stats['images_per_episode']}")
    print(f"📦 Tổng ảnh gốc được sử dụng: {aug_stats['total_original_used']:,}")
    print(f"🔄 Ảnh được augment mỗi episode: {aug_stats['augmented_images_per_episode']}")
    print(f"✨ Tổng ảnh được augment: {aug_stats['total_augmented_images']:,}")
    print(f"📊 Tổng ảnh sau augmentation: {aug_stats['total_images_after_aug']:,}")
    print(f"📈 Tỷ lệ augmentation: {aug_stats['augmentation_ratio']:.2%}")
    print(f"🚀 Kích thước dataset hiệu quả: {aug_stats['effective_dataset_size']:,}")
    
    # Thông tin chi tiết về augmentation
    aug_config = config['AUGMENTATION_CONFIG']
    print(f"\n🔧 Cấu hình augmentation:")
    print(f"   - Random Crop: +{aug_config['random_crop_size']}px")
    print(f"   - Rotation: ±{aug_config['rotation_degrees']}°")
    print(f"   - Horizontal Flip: {aug_config['flip_probability']:.1%}")
    print(f"   - Color Jitter: Brightness={aug_config['color_jitter']['brightness']}, "
          f"Contrast={aug_config['color_jitter']['contrast']}, "
          f"Saturation={aug_config['color_jitter']['saturation']}, "
          f"Hue={aug_config['color_jitter']['hue']}")
    print(f"   - Grayscale: {aug_config['grayscale_probability']:.1%}")
    
    print("=" * 50)

def main():
    """
    Main function
    """
    print("🚀 Khởi tạo Few-Shot Learning với Data Augmentation")
    print("=" * 60)
    
    # Load cấu hình
    config = load_config()
    print_config_summary(config)
    
    # Lưu cấu hình vào folder kết quả (sẽ cập nhật sau khi có aug_stats)
    save_config_to_output_folder(config, config['OUTPUT_FOLDER'])
    
    # Thiết lập device
    DEVICE = 'cuda' if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu'
    config['DEVICE'] = DEVICE
    
    # Phân tích và vẽ đồ thị dataset
    print("📈 PHÂN TÍCH DATASET:")
    if config['DETAILED_ANALYSIS']:
        print("🔍 Chạy phân tích chi tiết...")
        dataset_info = analyze_and_visualize_dataset(config['DATASET_PATH'], config)
    else:
        print("🔍 Chạy phân tích cơ bản...")
        dataset_info = analyze_and_visualize_dataset(config['DATASET_PATH'], config)
    
    if dataset_info is None:
        print("❌ Không thể phân tích dataset. Thoát chương trình.")
        exit()
    
    # Kiểm tra xem dataset có đủ dữ liệu cho few-shot learning không
    if dataset_info['total_images'] < config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY']):
        print(f"⚠️ Cảnh báo: Dataset có thể không đủ dữ liệu cho {config['N_WAY']}-way {config['K_SHOT']}-shot learning")
        print(f"   Cần ít nhất {config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY'])} ảnh, hiện có {dataset_info['total_images']} ảnh")
    
    # Tính toán và hiển thị thống kê augmentation
    aug_stats = calculate_augmentation_stats(config, dataset_info)
    print_augmentation_stats(aug_stats, config)
    
    # Cập nhật config với thống kê augmentation
    save_config_to_output_folder(config, config['OUTPUT_FOLDER'], aug_stats)
    
    print("\n" + "=" * 60)
    
    # Tạo transforms
    transform_basic, transform_augmented, transform_inference = create_transforms(config)
    
    # Khởi tạo dataset với data augmentation
    fewshot_data = FewShotDataset(
        config['DATASET_PATH'], 
        transform_train=transform_augmented,
        transform_test=transform_basic
    )
    
    # Khởi tạo model
    model = RelationNetworkModel(embed_dim=config['EMBED_DIM'], relation_dim=config['RELATION_DIM']).to(DEVICE)
    print(f"✅ Relation Network Model đã được tải lên {DEVICE}")
    print(f"📊 Cấu hình: {config['N_WAY']}-way, {config['K_SHOT']}-shot, {config['Q_QUERY']}-query")
    print(f"🧠 Kiến trúc: Vision Transformer + Relation Network (CNN)")
    print("=" * 60)

    # Chạy episodes với Relation Network
    print(f"\n🎯 CHẠY {config['NUM_EPISODES']} EPISODES VỚI RELATION NETWORK:")
    results_with_aug = run_multiple_episodes_with_detailed_evaluation(
        model, fewshot_data, config, config['NUM_EPISODES'], 
        use_augmentation=True, include_validation=config['USE_VALIDATION']
    )
    
    print(f"\n📊 KẾT QUẢ VỚI RELATION NETWORK:")
    print(f"   Query Accuracy trung bình: {results_with_aug['avg_query_acc']:.4f} ± {results_with_aug['std_query_acc']:.4f}")
    print(f"   Query Loss trung bình: {results_with_aug['avg_query_loss']:.4f} ± {results_with_aug['std_query_loss']:.4f}")
    print(f"   Query Accuracy min/max: {results_with_aug['min_query_acc']:.4f} / {results_with_aug['max_query_acc']:.4f}")
    
    if 'avg_valid_acc' in results_with_aug:
        print(f"   Validation Accuracy trung bình: {results_with_aug['avg_valid_acc']:.4f} ± {results_with_aug['std_valid_acc']:.4f}")
        print(f"   Validation Loss trung bình: {results_with_aug['avg_valid_loss']:.4f} ± {results_with_aug['std_valid_loss']:.4f}")
        print(f"   Validation Accuracy min/max: {results_with_aug['min_valid_acc']:.4f} / {results_with_aug['max_valid_acc']:.4f}")
        
        # So sánh Query vs Validation
        acc_diff = results_with_aug['avg_query_acc'] - results_with_aug['avg_valid_acc']
        print(f"   Chênh lệch Query-Validation Accuracy: {acc_diff:.4f}")
        if acc_diff > 0.05:
            print(f"   ⚠️ Có thể bị overfitting (Query > Validation)")
        elif acc_diff < -0.05:
            print(f"   ⚠️ Có thể bị underfitting (Query < Validation)")
        else:
            print(f"   ✅ Mô hình cân bằng tốt")
    
    # ==== ĐÁNH GIÁ CHI TIẾT QUERY SET ====
    print(f"\n🔍 ĐÁNH GIÁ CHI TIẾT QUERY SET:")
    query_predictions = results_with_aug['all_query_predictions']
    query_targets = results_with_aug['all_query_targets']
    
    # Lấy tên class thực tế từ dataset
    dataset_classes = fewshot_data.dataset.classes
    print(f"📋 Các class có sẵn trong dataset: {dataset_classes}")
    print(f"🎯 Sử dụng {config['N_WAY']} class trong mỗi episode (chọn ngẫu nhiên)")
    
    # Hiển thị các class được sử dụng trong các episodes
    print(f"📊 Các class được sử dụng trong {config['NUM_EPISODES']} episodes:")
    for i, episode_classes in enumerate(results_with_aug['all_episode_class_names']):
        print(f"   Episode {i+1}: {episode_classes}")
    
    # Sử dụng tên class thực tế từ episode cuối cùng cho đánh giá
    class_names = results_with_aug['all_episode_class_names'][-1]  # Lấy tên class từ episode cuối
    
    # Tính metrics chi tiết
    query_metrics = calculate_detailed_metrics(query_predictions, query_targets, config['N_WAY'])
    
    # In metrics
    print_detailed_evaluation_metrics(query_metrics, class_names, "Query Set")
    
    # Vẽ confusion matrix
    if config['SAVE_RESULTS']:
        plot_confusion_matrix(query_metrics['confusion_matrix'], class_names, "query_confusion_matrix.png", config)
        analyze_accuracy_by_class(query_predictions, query_targets, class_names, "query_accuracy_by_class.png", config)
        # plot_imbalance_analysis(query_metrics, class_names, "query_imbalance_analysis.png", config)
    
    # ==== ĐÁNH GIÁ CHI TIẾT VALIDATION SET (nếu có) ====
    if 'all_valid_predictions' in results_with_aug:
        print(f"\n🔍 ĐÁNH GIÁ CHI TIẾT VALIDATION SET:")
        valid_predictions = results_with_aug['all_valid_predictions']
        valid_targets = results_with_aug['all_valid_targets']
        
        # Tính metrics chi tiết
        valid_metrics = calculate_detailed_metrics(valid_predictions, valid_targets, config['N_WAY'])
        
        # In metrics (sử dụng cùng tên class thực tế)
        print_detailed_evaluation_metrics(valid_metrics, class_names, "Validation Set")
        
        # Vẽ confusion matrix
        if config['SAVE_RESULTS']:
            plot_confusion_matrix(valid_metrics['confusion_matrix'], class_names, "valid_confusion_matrix.png", config)
            analyze_accuracy_by_class(valid_predictions, valid_targets, class_names, "valid_accuracy_by_class.png", config)
            # plot_imbalance_analysis(valid_metrics, class_names, "valid_imbalance_analysis.png", config)
    
    # Vẽ đồ thị kết quả với augmentation
    if config['SAVE_RESULTS']:
        plot_single_results(results_with_aug, "episode_results_single.png", config)
    
    print("\n✅ Hoàn thành!")
    print(f"📁 Tất cả kết quả đã được lưu trong folder: {config['OUTPUT_FOLDER']}")
    print(f"💾 Cấu hình đã được lưu vào: {config['OUTPUT_FOLDER']}/config.json")
    if not config.get('SHOW_PLOTS', False):
        print("🔇 Chế độ không hiển thị ảnh pop-up đã được bật (SHOW_PLOTS = False)")
    print("=" * 60)
    
    if config['DETAILED_ANALYSIS']:
        print(f"📊 Đồ thị phân tích chi tiết: {config['OUTPUT_FOLDER']}/detailed_analysis.png")
        print(f"📊 Đồ thị định dạng file: {config['OUTPUT_FOLDER']}/file_formats_analysis.png")
    else:
        print(f"📊 Đồ thị phân tích cơ bản: {config['OUTPUT_FOLDER']}/dataset_analysis.png")
    
    if config['SAVE_RESULTS']:
            print(f"📊 Đồ thị kết quả episodes: {config['OUTPUT_FOLDER']}/episode_results_single.png")
    print(f"📊 Confusion matrix: {config['OUTPUT_FOLDER']}/query_confusion_matrix.png")
    print(f"📊 Accuracy theo class: {config['OUTPUT_FOLDER']}/query_accuracy_by_class.png")
    # print(f"📊 Imbalance analysis: {config['OUTPUT_FOLDER']}/query_imbalance_analysis.png")
    
    if config['USE_VALIDATION']:
        print(f"📊 Validation confusion matrix: {config['OUTPUT_FOLDER']}/valid_confusion_matrix.png")
        print(f"📊 Validation accuracy theo class: {config['OUTPUT_FOLDER']}/valid_accuracy_by_class.png")
        # print(f"📊 Validation imbalance analysis: {config['OUTPUT_FOLDER']}/valid_imbalance_analysis.png")
    
    print(f"🧠 Phương pháp: Relation Network (CNN) thay vì Euclidean Distance")
    print(f"📈 Relation scores: 0-1 (cao hơn = tương tự hơn)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
