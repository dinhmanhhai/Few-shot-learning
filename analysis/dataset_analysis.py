"""
Module cho phân tích dataset
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_visualize_dataset(dataset_path, config):
    """
    Phân tích và vẽ đồ thị số lượng ảnh trong từng class (phiên bản cơ bản)
    """
    PLOT_DPI = config['PLOT_DPI']
    
    if not os.path.exists(dataset_path):
        print(f"❌ Thư mục {dataset_path} không tồn tại!")
        return None
    
    print("🔍 Đang phân tích dataset...")
    
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
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    count += 1
            
            class_names.append(class_name)
            class_counts.append(count)
    
    if not class_names:
        print("❌ Không tìm thấy class nào!")
        return None
    
    # Tạo figure với subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PHÂN TÍCH DATASET - SỐ LƯỢNG ẢNH THEO CLASS', fontsize=16, fontweight='bold')
    
    # 1. Bar chart
    bars = ax1.bar(range(len(class_names)), class_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Số lượng ảnh theo từng class', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Số lượng ảnh', fontsize=12)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(class_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    wedges, texts, autotexts = ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Tỷ lệ phân bố các class', fontsize=14, fontweight='bold')
    
    # 3. Horizontal bar chart (top 10 classes)
    top_n = min(10, len(class_names))
    top_indices = np.argsort(class_counts)[-top_n:]
    top_names = [class_names[i] for i in top_indices]
    top_counts = [class_counts[i] for i in top_indices]
    
    bars_h = ax3.barh(range(len(top_names)), top_counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax3.set_title(f'Top {top_n} classes có nhiều ảnh nhất', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Số lượng ảnh', fontsize=12)
    ax3.set_yticks(range(len(top_names)))
    ax3.set_yticklabels(top_names)
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars_h, top_counts):
        width = bar.get_width()
        ax3.text(width + max(top_counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontweight='bold')
    
    # 4. Statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Tính toán thống kê
    total_images = sum(class_counts)
    avg_images = total_images / len(class_names)
    min_images = min(class_counts)
    max_images = max(class_counts)
    std_images = np.std(class_counts)
    
    stats_data = [
        ['Tổng số class', f'{len(class_names)}'],
        ['Tổng số ảnh', f'{total_images:,}'],
        ['Trung bình ảnh/class', f'{avg_images:.1f}'],
        ['Ít nhất', f'{min_images}'],
        ['Nhiều nhất', f'{max_images}'],
        ['Độ lệch chuẩn', f'{std_images:.1f}'],
        ['Class ít ảnh nhất', f'{class_names[np.argmin(class_counts)]} ({min_images})'],
        ['Class nhiều ảnh nhất', f'{class_names[np.argmax(class_counts)]} ({max_images})']
    ]
    
    table = ax4.table(cellText=stats_data, colLabels=['Thống kê', 'Giá trị'], 
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Tô màu header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Thống kê tổng quan', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['OUTPUT_FOLDER'], 'dataset_analysis.png'), dpi=PLOT_DPI, bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    # In thống kê ra console
    print("\n📊 THỐNG KÊ DATASET:")
    print("=" * 50)
    print(f"Tổng số class: {len(class_names)}")
    print(f"Tổng số ảnh: {total_images:,}")
    print(f"Trung bình ảnh/class: {avg_images:.1f}")
    print(f"Ít nhất: {min_images} ảnh")
    print(f"Nhiều nhất: {max_images} ảnh")
    print(f"Độ lệch chuẩn: {std_images:.1f}")
    print(f"Class ít ảnh nhất: {class_names[np.argmin(class_counts)]} ({min_images} ảnh)")
    print(f"Class nhiều ảnh nhất: {class_names[np.argmax(class_counts)]} ({max_images} ảnh)")
    
    # Kiểm tra balance
    balance_ratio = min_images / max_images
    if balance_ratio > 0.8:
        balance_status = "Cân bằng tốt"
    elif balance_ratio > 0.5:
        balance_status = "Cân bằng trung bình"
    else:
        balance_status = "Mất cân bằng"
    
    print(f"Tỷ lệ cân bằng: {balance_ratio:.2f} ({balance_status})")
    print("=" * 50)
    
    return {
        'class_names': class_names,
        'class_counts': class_counts,
        'total_images': total_images,
        'balance_ratio': balance_ratio
    }
