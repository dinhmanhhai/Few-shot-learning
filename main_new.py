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
from models.backbone import FlexibleDistanceModel
from data.dataset import FewShotDataset
from analysis.dataset_analysis import analyze_and_visualize_dataset
from evaluation.metrics import calculate_detailed_metrics, print_detailed_evaluation_metrics
from visualization.plots import (
    plot_confusion_matrix, 
    analyze_accuracy_by_class, 
    plot_single_results
)
from training.episode_runner import run_multiple_episodes_with_detailed_evaluation
from utils.class_augmentation import ClassSpecificAugmentation

def save_config_to_output_folder(config, output_folder, aug_stats=None, model_info=None):
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
    
    # Thêm thông tin model nếu có
    if model_info:
        config_to_save['model_info'] = model_info
    
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
    # print(f"📦 Tổng ảnh gốc được sử dụng: {aug_stats['total_original_used']:,}")
    # print(f"🔄 Ảnh được augment mỗi episode: {aug_stats['augmented_images_per_episode']}")
    # print(f"✨ Tổng ảnh được augment: {aug_stats['total_augmented_images']:,}")
    # print(f"📊 Tổng ảnh sau augmentation: {aug_stats['total_images_after_aug']:,}")
    # print(f"📈 Tỷ lệ augmentation: {aug_stats['augmentation_ratio']:.2%}")
    # print(f"🚀 Kích thước dataset hiệu quả: {aug_stats['effective_dataset_size']:,}")
    
    # Tính toán tổng số lượng ảnh của dataset sau augmentation
    total_dataset_after_aug = aug_stats['total_original_images'] + aug_stats['total_augmented_images']
    print(f"📊 Tổng số lượng ảnh dataset sau augmentation: {total_dataset_after_aug:,}")
    print(f"📈 Tỷ lệ tăng dataset: {(aug_stats['total_augmented_images']/aug_stats['total_original_images'])*100:.1f}%")
    
    # Thông tin về config được sử dụng
    print(f"\n⚙️ CẤU HÌNH FEW-SHOT LEARNING:")
    print(f"   • {config['N_WAY']}-way, {config['K_SHOT']}-shot, {config['Q_QUERY']}-query, {config['Q_VALID']}-valid")
    print(f"   • Số episodes: {config['NUM_EPISODES']}")
    print(f"   • Tổng ảnh được sử dụng: {aug_stats['total_original_used']:,} (gốc) + {aug_stats['total_augmented_images']:,} (augment) = {aug_stats['total_images_after_aug']:,}")
    print(f"   • Tỷ lệ sử dụng dataset: {(aug_stats['total_images_after_aug']/total_dataset_after_aug)*100:.1f}%")
    
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
    
    # Lưu cấu hình vào folder kết quả (sẽ cập nhật sau khi có model_info và aug_stats)
    save_config_to_output_folder(config, config['OUTPUT_FOLDER'])
    
    # Thiết lập device
    DEVICE = 'cuda' if torch.cuda.is_available() and config['USE_CUDA'] else 'cpu'
    config['DEVICE'] = DEVICE
    
    # Phân tích dataset
    print("📈 PHÂN TÍCH DATASET:")
    dataset_info = analyze_and_visualize_dataset(config['DATASET_PATH'], config)
    
    # Tạo đồ thị phân bố class riêng biệt
    print("📊 TẠO ĐỒ THỊ PHÂN BỐ CLASS:")
    from analysis.dataset_analysis import create_class_distribution_chart
    class_dist_info = create_class_distribution_chart(config['DATASET_PATH'], config)
    
    if dataset_info is None:
        print("❌ Không thể phân tích dataset. Thoát chương trình.")
        exit()
    
    # Debug: Kiểm tra class distribution
    print(f"🔍 Debug - Dataset info keys: {list(dataset_info.keys())}")
    if 'class_distribution' in dataset_info:
        print(f"📊 Class distribution có sẵn: {len(dataset_info['class_distribution'])} classes")
        print(f"   Sample: {dict(list(dataset_info['class_distribution'].items())[:3])}")
    else:
        print("⚠️ Không có class_distribution trong dataset_info")
        print("   Có thể cần chạy DETAILED_ANALYSIS = True")
    
    # Kiểm tra xem dataset có đủ dữ liệu cho few-shot learning không
    if dataset_info['total_images'] < config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY']):
        print(f"⚠️ Cảnh báo: Dataset có thể không đủ dữ liệu cho {config['N_WAY']}-way {config['K_SHOT']}-shot learning")
        print(f"   Cần ít nhất {config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY'])} ảnh, hiện có {dataset_info['total_images']} ảnh")
    
    # Tính toán và hiển thị thống kê augmentation
    if config.get('USE_AUGMENTATION', False):
        aug_stats = calculate_augmentation_stats(config, dataset_info)
        print_augmentation_stats(aug_stats, config)
        
        # Cập nhật config với thống kê augmentation (model_info sẽ được thêm sau)
        save_config_to_output_folder(config, config['OUTPUT_FOLDER'], aug_stats)
        
        # Xử lý class-specific augmentation
        if config.get('CLASS_AUGMENTATION', {}).get('enable_selective', False):
            print("\n🎯 PHÂN TÍCH CLASS-SPECIFIC AUGMENTATION:")
            print("=" * 60)
            
            # Khởi tạo class-specific augmentation
            class_aug = ClassSpecificAugmentation(config)
            
            # Lấy class distribution từ dataset_info
            class_distribution = dataset_info.get('class_distribution', {})
            if class_distribution:
                # Tính toán kế hoạch augmentation
                augmentation_plan = class_aug.calculate_augmentation_needs(class_distribution)
                
                # In kế hoạch augmentation
                class_aug.print_augmentation_plan(class_distribution)
                

                
                # Cập nhật aug_stats với thông tin class-specific
                aug_stats['class_specific_info'] = {
                    'augmentation_plan': augmentation_plan,
                    'classes_to_augment': [name for name, plan in augmentation_plan.items() 
                                         if plan['should_augment']],
                    'classes_to_skip': [name for name, plan in augmentation_plan.items() 
                                       if not plan['should_augment']]
                }
                
                # Cập nhật config với thống kê mới (model_info sẽ được thêm sau)
                save_config_to_output_folder(config, config['OUTPUT_FOLDER'], aug_stats)
                
                # Tạo đồ thị so sánh augmentation
                print("📊 TẠO ĐỒ THỊ SO SÁNH AUGMENTATION:")
                from analysis.dataset_analysis import create_augmentation_comparison_chart
                aug_comparison_info = create_augmentation_comparison_chart(config['DATASET_PATH'], config, aug_stats)
            else:
                print("⚠️ Không thể lấy thông tin class distribution từ dataset")
                print("   Dataset info keys:", list(dataset_info.keys()))
        else:
            print("ℹ️ Class-specific augmentation không được bật")
            
            # Tạo đồ thị so sánh augmentation (cho trường hợp không có class-specific)
            print("📊 TẠO ĐỒ THỊ SO SÁNH AUGMENTATION:")
            from analysis.dataset_analysis import create_augmentation_comparison_chart
            aug_comparison_info = create_augmentation_comparison_chart(config['DATASET_PATH'], config, aug_stats)
    else:
        # Tạo thống kê giả khi không có augmentation
        aug_stats = {
            'total_original_images': dataset_info['total_images'],
            'images_per_episode': config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY'] + config['Q_VALID']),
            'total_original_used': config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY'] + config['Q_VALID']) * config['NUM_EPISODES'],
            'augmented_images_per_episode': 0,  # Không có augmentation
            'total_augmented_images': 0,  # Không có augmentation
            'total_images_after_aug': config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY'] + config['Q_VALID']) * config['NUM_EPISODES'],
            'augmentation_ratio': 0.0,  # Không có augmentation
            'effective_dataset_size': config['N_WAY'] * (config['K_SHOT'] + config['Q_QUERY'] + config['Q_VALID']) * config['NUM_EPISODES']
        }
        
        print("\n📊 THỐNG KÊ (KHÔNG CÓ AUGMENTATION):")
        print("=" * 50)
        print(f"📈 Tổng ảnh gốc trong dataset: {aug_stats['total_original_images']:,}")
        print(f"🎯 Ảnh sử dụng mỗi episode: {aug_stats['images_per_episode']}")
        print(f"📦 Tổng ảnh được sử dụng: {aug_stats['total_original_used']:,}")
        print(f"🔄 Ảnh được augment: 0 (USE_AUGMENTATION = False)")
        print(f"📊 Tổng ảnh sau augmentation: {aug_stats['total_images_after_aug']:,}")
        print(f"📈 Tỷ lệ augmentation: 0.00% (USE_AUGMENTATION = False)")
        print(f"🚀 Kích thước dataset hiệu quả: {aug_stats['effective_dataset_size']:,}")
        print("=" * 50)
        
        # Cập nhật config với thống kê (model_info sẽ được thêm sau)
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
    
    # Khởi tạo model với phương pháp được chọn
    distance_method = config.get('DISTANCE_METHOD', 'relation_network')
    use_learnable = config.get('USE_LEARNABLE_METRIC', True)
    transformer_model = config.get('TRANSFORMER_MODEL', 'swin_base_patch4_window7_224')
    
    if use_learnable:
        model = FlexibleDistanceModel(
            embed_dim=config['EMBED_DIM'], 
            relation_dim=config['RELATION_DIM'],
            distance_method=distance_method,
            transformer_model=transformer_model
        ).to(DEVICE)
    else:
        model = FlexibleDistanceModel(
            embed_dim=config['EMBED_DIM'], 
            relation_dim=config['RELATION_DIM'],
            distance_method="euclidean",
            transformer_model=transformer_model
        ).to(DEVICE)
    
    # Tạo thông tin model để lưu vào JSON
    transformer_names = {
        'swin_base_patch4_window7_224': 'Swin-Base',
        'swin_large_patch4_window12_384': 'Swin-Large',
        'convnext_base': 'ConvNeXt-Base',
        'convnext_large': 'ConvNeXt-Large'
    }
    
    model_info = {
        'model_name': 'FlexibleDistanceModel',
        'backbone': transformer_names.get(model.transformer_model, model.transformer_model),
        'backbone_architecture': model.transformer_model,
        'embed_dim': config['EMBED_DIM'],
        'relation_dim': config['RELATION_DIM'],
        'distance_method': distance_method,
        'use_learnable_metric': use_learnable,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': DEVICE
    }
    
    print(f"✅ Flexible Distance Model đã được tải lên {DEVICE}")
    print(f"📊 Cấu hình: {config['N_WAY']}-way, {config['K_SHOT']}-shot, {config['Q_QUERY']}-query")
    print(f"🧠 Kiến trúc: {model_info['backbone']} + {distance_method.upper()}")
    print(f"📈 Tổng tham số: {model_info['total_parameters']:,}")
    print(f"🔧 Tham số có thể train: {model_info['trainable_parameters']:,}")
    print("=" * 60)
    
    # Lưu thông tin model vào config JSON
    save_config_to_output_folder(config, config['OUTPUT_FOLDER'], aug_stats, model_info)

    # Chạy episodes với Relation Network
    use_aug = config.get('USE_AUGMENTATION', False)
    aug_status = "CÓ AUGMENTATION" if use_aug else "KHÔNG AUGMENTATION"
    
    print(f"\n🎯 CHẠY {config['NUM_EPISODES']} EPISODES VỚI RELATION NETWORK ({aug_status}):")
    results_with_aug = run_multiple_episodes_with_detailed_evaluation(
        model, fewshot_data, config, config['NUM_EPISODES'], 
        use_augmentation=use_aug, include_validation=config['USE_VALIDATION']
    )
    
    print(f"\n📊 KẾT QUẢ VỚI RELATION NETWORK ({aug_status}):")
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
    
    # Vẽ đồ thị kết quả
    if config['SAVE_RESULTS']:
        plot_single_results(results_with_aug, "episode_results_single.png", config)
    
    print("\n✅ Hoàn thành!")
    print(f"📁 Tất cả kết quả đã được lưu trong folder: {config['OUTPUT_FOLDER']}")
    print(f"💾 Cấu hình đã được lưu vào: {config['OUTPUT_FOLDER']}/config.json")
    if not config.get('SHOW_PLOTS', False):
        print("🔇 Chế độ không hiển thị ảnh pop-up đã được bật (SHOW_PLOTS = False)")
    print("=" * 60)
    
    print(f"📊 Đồ thị phân bố class: {config['OUTPUT_FOLDER']}/class_distribution.png")
    
    if config.get('USE_AUGMENTATION', False):
        print(f"📊 Đồ thị so sánh augmentation: {config['OUTPUT_FOLDER']}/augmentation_comparison.png")
    
    if config['SAVE_RESULTS']:
            print(f"📊 Đồ thị kết quả episodes: {config['OUTPUT_FOLDER']}/episode_results_single.png")
    print(f"📊 Confusion matrix: {config['OUTPUT_FOLDER']}/query_confusion_matrix.png")
    print(f"📊 Accuracy theo class: {config['OUTPUT_FOLDER']}/query_accuracy_by_class.png")
    

    
    # print(f"📊 Imbalance analysis: {config['OUTPUT_FOLDER']}/query_imbalance_analysis.png")
    
    if config['USE_VALIDATION']:
        print(f"📊 Validation confusion matrix: {config['OUTPUT_FOLDER']}/valid_confusion_matrix.png")
        print(f"📊 Validation accuracy theo class: {config['OUTPUT_FOLDER']}/valid_accuracy_by_class.png")
        # print(f"📊 Validation imbalance analysis: {config['OUTPUT_FOLDER']}/valid_imbalance_analysis.png")
    
    print(f"🧠 Model: {model_info['model_name']} ({model_info['backbone']})")
    print(f"🔧 Phương pháp: {distance_method.upper()} ({'Có thể học được' if use_learnable else 'Cố định'})")
    if use_learnable:
        print(f"📈 Relation scores: 0-1 (cao hơn = tương tự hơn)")
    else:
        print(f"📈 Euclidean similarity: 0-1 (cao hơn = tương tự hơn)")
    print(f"🔄 Data Augmentation: {'BẬT' if config.get('USE_AUGMENTATION', False) else 'TẮT'} (USE_AUGMENTATION = {config.get('USE_AUGMENTATION', False)})")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
