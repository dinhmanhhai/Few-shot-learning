"""
Module for dataset analysis
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def create_class_distribution_chart(dataset_path, config):
    """
    Create bar chart for number of images by class
    """
    PLOT_DPI = config['PLOT_DPI']
    
    if not os.path.exists(dataset_path):
        print(f"❌ Directory {dataset_path} does not exist!")
        return None
    
    print("🔍 Creating class distribution chart...")
    
    # Get class information
    class_names = []
    class_counts = []
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Count image files in class
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
        print("❌ No images found!")
        return None
    
    # Create chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(class_names)), class_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Number of Images by Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Add values on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Add average line
    avg_count = np.mean(class_counts)
    plt.axhline(y=avg_count, color='red', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_count:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure output folder exists
    os.makedirs(config['OUTPUT_FOLDER'], exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['OUTPUT_FOLDER'], 'class_distribution.png'), dpi=PLOT_DPI, bbox_inches='tight')
    
    # Only show if configured
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Close figure to save memory
    
    print(f"📊 Class distribution chart saved to: {config['OUTPUT_FOLDER']}/class_distribution.png")
    
    return {
        'class_names': class_names,
        'class_counts': class_counts,
        'total_images': sum(class_counts),
        'avg_count': avg_count
    }

def create_augmentation_comparison_chart(dataset_path, config, aug_stats):
    """
    Create comparison chart for number of images before and after augmentation
    """
    PLOT_DPI = config['PLOT_DPI']
    
    if not os.path.exists(dataset_path):
        print(f"❌ Directory {dataset_path} does not exist!")
        return None
    
    print("🔍 Creating augmentation comparison chart...")
    
    # Get class information
    class_names = []
    original_counts = []
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Count image files in class
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
        print("❌ No images found!")
        return None
    
    # Calculate number of images after augmentation
    if config.get('USE_AUGMENTATION', False):
        # Calculate augmentation for each class
        augmented_counts = []
        for i, class_name in enumerate(class_names):
            original_count = original_counts[i]
            
            # Check if this class should be augmented
            should_augment = True
            if config.get('CLASS_AUGMENTATION', {}).get('enable_selective', False):
                class_aug_config = config['CLASS_AUGMENTATION']
                # Convert class name to index (if needed)
                class_index = i  # Assume class order in dataset
                
                if class_index in class_aug_config.get('skip_classes', []):
                    should_augment = False
                elif class_aug_config.get('augment_classes') and class_index not in class_aug_config['augment_classes']:
                    should_augment = False
            
            if should_augment:
                # Calculate number of augmented images for this class
                augment_ratio = config.get('CLASS_AUGMENTATION', {}).get('augment_ratio', 1.5)
                augmented_count = int(original_count * augment_ratio)
            else:
                augmented_count = original_count
            
            augmented_counts.append(augmented_count)
    else:
        # No augmentation
        augmented_counts = original_counts.copy()
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Bar chart comparing before and after augmentation
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_counts, width, label='Before Augmentation', 
                    color='lightblue', edgecolor='navy', alpha=0.7)
    bars2 = ax1.bar(x + width/2, augmented_counts, width, label='After Augmentation', 
                    color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    ax1.set_title('Comparison of Number of Images Before and After Augmentation', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(augmented_counts)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(augmented_counts)*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 2. Pie chart for augmentation ratio
    total_original = sum(original_counts)
    total_augmented = sum(augmented_counts)
    total_increase = total_augmented - total_original
    
    labels = ['Original Images', 'Augmented Images']
    sizes = [total_original, total_increase]
    colors = ['lightblue', 'lightcoral']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Ratio of Original vs Augmented Images', fontsize=16, fontweight='bold')
    
    # Add overview information
    fig.suptitle('DATA AUGMENTATION ANALYSIS', fontsize=18, fontweight='bold', y=0.95)
    

    
    # Ensure output folder exists
    os.makedirs(config['OUTPUT_FOLDER'], exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['OUTPUT_FOLDER'], 'augmentation_comparison.png'), dpi=PLOT_DPI, bbox_inches='tight')
    
    # Only show if configured
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Close figure to save memory
    
    print(f"📊 Augmentation comparison chart saved to: {config['OUTPUT_FOLDER']}/augmentation_comparison.png")
    
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
