"""
Module cho phân tích dataset
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def create_class_distribution_chart(dataset_path, config):
    """
    Tạo đồ thị bar chart riêng cho số lượng ảnh theo từng class
    """
    PLOT_DPI = config['PLOT_DPI']
    
    if not os.path.exists(dataset_path):
        print(f"❌ Thư mục {dataset_path} không tồn tại!")
        return None
    
    print("🔍 Đang tạo đồ thị phân bố class...")
    
    # Lấy thông tin các class
    class_names = []
    class_counts = []
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Đếm số file ảnh trong class
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
            count = 0
            for file in os.listdir(class_path):
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    count += 1
            
            if count > 0:
                class_names.append(class_name)
                class_counts.append(count)
    
    if not class_names:
        print("❌ Không tìm thấy ảnh nào!")
        return None
    
    # Tạo đồ thị
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(class_names)), class_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Số lượng ảnh theo từng class', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Số lượng ảnh', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Thêm đường trung bình
    avg_count = np.mean(class_counts)
    plt.axhline(y=avg_count, color='red', linestyle='--', alpha=0.7, 
                label=f'Trung bình: {avg_count:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Đảm bảo output folder tồn tại
    os.makedirs(config['OUTPUT_FOLDER'], exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['OUTPUT_FOLDER'], 'class_distribution.png'), dpi=PLOT_DPI, bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print(f"📊 Đồ thị phân bố class đã được lưu vào: {config['OUTPUT_FOLDER']}/class_distribution.png")
    
    return {
        'class_names': class_names,
        'class_counts': class_counts,
        'total_images': sum(class_counts),
        'avg_count': avg_count
    }

def create_augmentation_comparison_chart(dataset_path, config, aug_stats):
    """
    Tạo đồ thị so sánh số lượng ảnh trước và sau augmentation
    """
    PLOT_DPI = config['PLOT_DPI']
    
    if not os.path.exists(dataset_path):
        print(f"❌ Thư mục {dataset_path} không tồn tại!")
        return None
    
    print("🔍 Đang tạo đồ thị so sánh augmentation...")
    
    # Lấy thông tin các class
    class_names = []
    original_counts = []
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Đếm số file ảnh trong class
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
            count = 0
            for file in os.listdir(class_path):
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    count += 1
            
            if count > 0:
                class_names.append(class_name)
                original_counts.append(count)
    
    if not class_names:
        print("❌ Không tìm thấy ảnh nào!")
        return None
    
    # Tính toán số lượng ảnh sau augmentation
    if config.get('USE_AUGMENTATION', False):
        # Tính augmentation cho từng class
        augmented_counts = []
        for i, class_name in enumerate(class_names):
            original_count = original_counts[i]
            
            # Kiểm tra xem class này có được augment không
            should_augment = True
            if config.get('CLASS_AUGMENTATION', {}).get('enable_selective', False):
                class_aug_config = config['CLASS_AUGMENTATION']
                # Chuyển tên class thành index (nếu cần)
                class_index = i  # Giả sử thứ tự class trong dataset
                
                if class_index in class_aug_config.get('skip_classes', []):
                    should_augment = False
                elif class_aug_config.get('augment_classes') and class_index not in class_aug_config['augment_classes']:
                    should_augment = False
            
            if should_augment:
                # Tính số ảnh được augment cho class này
                augment_ratio = config.get('CLASS_AUGMENTATION', {}).get('augment_ratio', 1.5)
                augmented_count = int(original_count * augment_ratio)
            else:
                augmented_count = original_count
            
            augmented_counts.append(augmented_count)
    else:
        # Không có augmentation
        augmented_counts = original_counts.copy()
    
    # Tạo đồ thị so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Bar chart so sánh trước và sau augmentation
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_counts, width, label='Trước Augmentation', 
                    color='lightblue', edgecolor='navy', alpha=0.7)
    bars2 = ax1.bar(x + width/2, augmented_counts, width, label='Sau Augmentation', 
                    color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    ax1.set_title('So sánh số lượng ảnh trước và sau Augmentation', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Số lượng ảnh', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Thêm số liệu trên bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(augmented_counts)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(augmented_counts)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 2. Pie chart tỷ lệ augmentation
    total_original = sum(original_counts)
    total_augmented = sum(augmented_counts)
    total_increase = total_augmented - total_original
    
    labels = ['Ảnh gốc', 'Ảnh được augment']
    sizes = [total_original, total_increase]
    colors = ['lightblue', 'lightcoral']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Tỷ lệ ảnh gốc vs ảnh được augment', fontsize=16, fontweight='bold')
    
    # Thêm thông tin tổng quan
    fig.suptitle('PHÂN TÍCH DATA AUGMENTATION', fontsize=18, fontweight='bold', y=0.95)
    

    
    # Đảm bảo output folder tồn tại
    os.makedirs(config['OUTPUT_FOLDER'], exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['OUTPUT_FOLDER'], 'augmentation_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print(f"📊 Đồ thị so sánh augmentation đã được lưu vào: {config['OUTPUT_FOLDER']}/augmentation_comparison.png")
    
    return {
        'class_names': class_names,
        'original_counts': original_counts,
        'augmented_counts': augmented_counts,
        'total_original': total_original,
        'total_augmented': total_augmented,
        'total_increase': total_increase
    }

def analyze_and_visualize_dataset(dataset_path, config):
    """
    Phân tích dataset và trả về thông tin (không tạo đồ thị)
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Thư mục {dataset_path} không tồn tại!")
        return None
    
    print("🔍 Đang phân tích dataset...")
    
    # Lấy thông tin các class
    class_names = []
    class_counts = []
    class_distribution = {}
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Đếm số file ảnh trong class
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
            count = 0
            for file in os.listdir(class_path):
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    count += 1
            
            if count > 0:
                class_names.append(class_name)
                class_counts.append(count)
                class_distribution[class_name] = count
    
    if not class_names:
        print("❌ Không tìm thấy ảnh nào!")
        return None
    
    total_images = sum(class_counts)
    avg_count = np.mean(class_counts)
    min_count = min(class_counts)
    max_count = max(class_counts)
    
    print(f"✅ Phân tích dataset hoàn thành!")
    print(f"📊 Tổng số class: {len(class_names)}")
    print(f"📊 Tổng số ảnh: {total_images:,}")
    print(f"📊 Trung bình ảnh/class: {avg_count:.1f}")
    print(f"📊 Ít nhất: {min_count} ảnh")
    print(f"📊 Nhiều nhất: {max_count} ảnh")
    
    return {
        'class_names': class_names,
        'class_counts': class_counts,
        'class_distribution': class_distribution,
        'total_images': total_images,
        'avg_count': avg_count,
        'min_count': min_count,
        'max_count': max_count
    }
