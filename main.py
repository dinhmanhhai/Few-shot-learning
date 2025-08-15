import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from timm import create_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import cho đánh giá chi tiết
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# ==== IMPORT CẤU HÌNH ====
try:
    from config import *
    print("✅ Đã tải cấu hình từ config.py")
except ImportError:
    print("⚠️ Không tìm thấy config.py, sử dụng cấu hình mặc định")
    # Cấu hình mặc định nếu không có config.py
    DATASET_PATH = r'D:\AI\Dataset'
    N_WAY = 5
    K_SHOT = 1
    Q_QUERY = 5
    EMBED_DIM = 512
    IMAGE_SIZE = 224
    NUM_EPISODES = 10
    SAVE_RESULTS = True
    COMPARE_WITHOUT_AUG = False
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
        'grayscale_probability': 0.1
    }
    DISPLAY_PROGRESS = True
    SAVE_PLOTS = True
    PLOT_DPI = 300
    USE_CUDA = True

# ==== THIẾT LẬP OUTPUT FOLDER ====
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = f"few_shot_results_{timestamp}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"📁 Tạo output folder: {OUTPUT_FOLDER}")

# ==== THIẾT LẬP DEVICE ====
DEVICE = 'cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu'
root_dir = DATASET_PATH  # Sử dụng đường dẫn từ config

# ==== HÀM VẼ ĐỒ THỊ PHÂN TÍCH DATASET ====
def analyze_and_visualize_dataset(dataset_path):
    """
    Phân tích và vẽ đồ thị số lượng ảnh trong từng class (phiên bản cơ bản)
    """
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
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'dataset_analysis.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()
    
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

# ==== PHÂN TÍCH CHI TIẾT DATASET ====
def analyze_dataset_detailed(dataset_path, save_dir=None):
    """
    Phân tích chi tiết dataset với nhiều loại đồ thị (từ data_analyzer.py)
    """
    if save_dir is None:
        save_dir = OUTPUT_FOLDER
    if not os.path.exists(dataset_path):
        print(f"❌ Thư mục {dataset_path} không tồn tại!")
        return None
    
    # Tạo thư mục lưu kết quả
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔍 Đang phân tích dataset chi tiết...")
    
    # Thu thập dữ liệu
    class_data = {}
    total_images = 0
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Đếm số file ảnh và phân loại theo định dạng
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
            count = 0
            format_counts = {}
            
            for file in os.listdir(class_path):
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    count += 1
                    # Xác định định dạng file
                    for ext in image_extensions:
                        if file_lower.endswith(ext):
                            format_counts[ext[1:]] = format_counts.get(ext[1:], 0) + 1
                            break
            
            if count > 0:
                class_data[class_name] = {
                    'count': count,
                    'formats': format_counts
                }
                total_images += count
    
    if not class_data:
        print("❌ Không tìm thấy ảnh nào!")
        return None
    
    # Tạo DataFrame để dễ xử lý
    df_data = []
    for class_name, data in class_data.items():
        df_data.append({
            'class': class_name,
            'count': data['count'],
            'formats': data['formats']
        })
    
    # Sắp xếp theo số lượng ảnh
    df_data.sort(key=lambda x: x['count'], reverse=True)
    class_names = [item['class'] for item in df_data]
    class_counts = [item['count'] for item in df_data]
    
    # Tạo figure với nhiều subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Bar chart chính
    ax1 = fig.add_subplot(gs[0, :2])
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
    ax2 = fig.add_subplot(gs[0, 2])
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    wedges, texts, autotexts = ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Tỷ lệ phân bố các class', fontsize=14, fontweight='bold')
    
    # 3. Top 10 classes
    ax3 = fig.add_subplot(gs[1, :])
    top_10 = df_data[:10] if len(df_data) > 10 else df_data
    top_names = [item['class'] for item in top_10]
    top_counts = [item['count'] for item in top_10]
    
    bars_h = ax3.barh(range(len(top_names)), top_counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax3.set_title(f'Top {len(top_names)} classes có nhiều ảnh nhất', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Số lượng ảnh', fontsize=12)
    ax3.set_yticks(range(len(top_names)))
    ax3.set_yticklabels(top_names)
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars_h, top_counts):
        width = bar.get_width()
        ax3.text(width + max(top_counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontweight='bold')
    
    # 4. Distribution histogram
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(class_counts, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax4.set_title('Phân bố số lượng ảnh', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Số lượng ảnh', fontsize=12)
    ax4.set_ylabel('Số class', fontsize=12)
    
    # 5. Box plot
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.boxplot(class_counts, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax5.set_title('Box plot số lượng ảnh', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Số lượng ảnh', fontsize=12)
    
    # 6. Statistics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Tính toán thống kê
    avg_images = np.mean(class_counts)
    median_images = np.median(class_counts)
    min_images = min(class_counts)
    max_images = max(class_counts)
    std_images = np.std(class_counts)
    
    stats_data = [
        ['Tổng số class', f'{len(class_names)}'],
        ['Tổng số ảnh', f'{total_images:,}'],
        ['Trung bình ảnh/class', f'{avg_images:.1f}'],
        ['Median ảnh/class', f'{median_images:.1f}'],
        ['Ít nhất', f'{min_images}'],
        ['Nhiều nhất', f'{max_images}'],
        ['Độ lệch chuẩn', f'{std_images:.1f}'],
        ['Class ít ảnh nhất', f'{class_names[-1]} ({min_images})'],
        ['Class nhiều ảnh nhất', f'{class_names[0]} ({max_images})']
    ]
    
    table = ax6.table(cellText=stats_data, colLabels=['Thống kê', 'Giá trị'], 
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Tô màu header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Thống kê tổng quan', fontsize=14, fontweight='bold', pad=20)
    
    # Tổng tiêu đề
    fig.suptitle('PHÂN TÍCH CHI TIẾT DATASET', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'detailed_analysis.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()
    
    # Tạo đồ thị phân tích định dạng file
    analyze_file_formats(df_data, OUTPUT_FOLDER)
    
    # In thống kê ra console
    print_detailed_statistics(class_names, class_counts, total_images)
    
    return {
        'class_names': class_names,
        'class_counts': class_counts,
        'total_images': total_images,
        'class_data': class_data
    }

def analyze_file_formats(df_data, save_dir):
    """
    Phân tích định dạng file ảnh
    """
    # Thu thập thông tin định dạng
    all_formats = {}
    for item in df_data:
        for fmt, count in item['formats'].items():
            all_formats[fmt] = all_formats.get(fmt, 0) + count
    
    if not all_formats:
        return
    
    # Vẽ đồ thị định dạng file
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart định dạng
    formats = list(all_formats.keys())
    counts = list(all_formats.values())
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(formats)))
    
    wedges, texts, autotexts = ax1.pie(counts, labels=formats, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax1.set_title('Phân bố định dạng file ảnh', fontsize=14, fontweight='bold')
    
    # Bar chart định dạng
    bars = ax2.bar(formats, counts, color=colors, alpha=0.7)
    ax2.set_title('Số lượng file theo định dạng', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Định dạng', fontsize=12)
    ax2.set_ylabel('Số lượng file', fontsize=12)
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'file_formats_analysis.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()

def print_detailed_statistics(class_names, class_counts, total_images):
    """
    In thống kê chi tiết ra console
    """
    print("\n📊 THỐNG KÊ CHI TIẾT DATASET:")
    print("=" * 60)
    print(f"Tổng số class: {len(class_names)}")
    print(f"Tổng số ảnh: {total_images:,}")
    print(f"Trung bình ảnh/class: {np.mean(class_counts):.1f}")
    print(f"Median ảnh/class: {np.median(class_counts):.1f}")
    print(f"Ít nhất: {min(class_counts)} ảnh")
    print(f"Nhiều nhất: {max(class_counts)} ảnh")
    print(f"Độ lệch chuẩn: {np.std(class_counts):.1f}")
    print(f"Class ít ảnh nhất: {class_names[-1]} ({min(class_counts)} ảnh)")
    print(f"Class nhiều ảnh nhất: {class_names[0]} ({max(class_counts)} ảnh)")
    
    # Kiểm tra balance
    balance_ratio = min(class_counts) / max(class_counts)
    if balance_ratio > 0.8:
        balance_status = "Cân bằng tốt"
    elif balance_ratio > 0.5:
        balance_status = "Cân bằng trung bình"
    else:
        balance_status = "Mất cân bằng"
    
    print(f"Tỷ lệ cân bằng: {balance_ratio:.2f} ({balance_status})")
    
    # Phân tích quartiles
    q25, q50, q75 = np.percentile(class_counts, [25, 50, 75])
    print(f"Quartile 25%: {q25:.1f} ảnh")
    print(f"Quartile 50% (median): {q50:.1f} ảnh")
    print(f"Quartile 75%: {q75:.1f} ảnh")
    
    print("=" * 60)

# ==== HÀM ĐÁNH GIÁ ĐỘ CHÍNH XÁC CHI TIẾT ====
def calculate_detailed_metrics(predictions, targets, n_classes):
    """
    Tính toán các metrics chi tiết: precision, recall, F1-score, confusion matrix
    """

    # Tính precision, recall, F1-score cho từng class
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    
    # Tính macro và weighted averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(n_classes))
    
    # Classification report
    class_report = classification_report(targets, predictions, output_dict=True, zero_division=0)
    
    return {
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Vẽ confusion matrix
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(OUTPUT_FOLDER):
        save_path = os.path.join(OUTPUT_FOLDER, save_path)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()
    print(f"📊 Confusion matrix đã được lưu vào: {save_path}")

def analyze_accuracy_by_class(predictions, targets, class_names, save_path="accuracy_by_class.png"):
    """
    Phân tích độ chính xác theo từng class
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(OUTPUT_FOLDER):
        save_path = os.path.join(OUTPUT_FOLDER, save_path)
    n_classes = len(class_names)
    class_accuracies = []
    
    for i in range(n_classes):
        class_mask = targets == i
        if class_mask.sum() > 0:
            class_acc = (predictions[class_mask] == targets[class_mask]).mean().item()
        else:
            class_acc = 0.0
        class_accuracies.append(class_acc)
    
    # Vẽ đồ thị accuracy theo class
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(n_classes), class_accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Độ chính xác theo từng Class', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(n_classes), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    # Thêm số liệu trên bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Thêm đường trung bình
    avg_acc = np.mean(class_accuracies)
    plt.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'Trung bình: {avg_acc:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()
    print(f"📊 Đồ thị accuracy theo class đã được lưu vào: {save_path}")
    
    return class_accuracies

def analyze_imbalance_impact(metrics, class_names):
    """
    Phân tích ảnh hưởng của dataset imbalance
    """
    print("\n⚖️ PHÂN TÍCH ẢNH HƯỞNG IMBALANCE:")
    print("-" * 40)
    
    # Lấy support (số lượng samples) cho từng class
    support_per_class = metrics['support_per_class']
    f1_per_class = metrics['f1_per_class']
    
    # Tính toán imbalance ratio
    max_support = max(support_per_class)
    min_support = min(support_per_class)
    imbalance_ratio = min_support / max_support
    
    print(f"   Imbalance Ratio: {imbalance_ratio:.3f}")
    
    # Phân loại classes theo số lượng samples
    minority_classes = []
    majority_classes = []
    
    for i, support in enumerate(support_per_class):
        if support <= max_support * 0.3:  # Classes có ít hơn 30% samples so với class lớn nhất
            minority_classes.append((class_names[i], support, f1_per_class[i]))
        else:
            majority_classes.append((class_names[i], support, f1_per_class[i]))
    
    # Phân tích performance
    if minority_classes:
        minority_f1_avg = np.mean([f1 for _, _, f1 in minority_classes])
        print(f"   Minority Classes ({len(minority_classes)}): {[name for name, _, _ in minority_classes]}")
        print(f"   Minority Classes F1-Score TB: {minority_f1_avg:.4f}")
    
    if majority_classes:
        majority_f1_avg = np.mean([f1 for _, _, f1 in majority_classes])
        print(f"   Majority Classes ({len(majority_classes)}): {[name for name, _, _ in majority_classes]}")
        print(f"   Majority Classes F1-Score TB: {majority_f1_avg:.4f}")
    
    # So sánh Macro vs Weighted
    macro_f1 = metrics['macro_f1']
    weighted_f1 = metrics['weighted_f1']
    f1_difference = weighted_f1 - macro_f1
    
    print(f"   Macro vs Weighted F1 Difference: {f1_difference:.4f}")
    
    # Đánh giá mức độ imbalance
    if imbalance_ratio > 0.8:
        balance_status = "Cân bằng tốt"
        impact_level = "Thấp"
    elif imbalance_ratio > 0.5:
        balance_status = "Cân bằng trung bình"
        impact_level = "Trung bình"
    elif imbalance_ratio > 0.2:
        balance_status = "Mất cân bằng"
        impact_level = "Cao"
    else:
        balance_status = "Mất cân bằng nghiêm trọng"
        impact_level = "Rất cao"
    
    print(f"   Balance Status: {balance_status}")
    print(f"   Impact Level: {impact_level}")
    
    # Cảnh báo nếu có vấn đề
    if imbalance_ratio < 0.5:
        print(f"   ⚠️ CẢNH BÁO: Dataset mất cân bằng nghiêm trọng!")
        print(f"      - Classes ít ảnh có thể bị dự đoán sai")
        print(f"      - Kết quả có thể không đại diện cho toàn bộ dataset")
        print(f"      - Cần xem xét data augmentation hoặc resampling")
    
    if f1_difference > 0.1:
        print(f"   ⚠️ CẢNH BÁO: Chênh lệch Macro-Weighted F1 lớn!")
        print(f"      - Mô hình thiên về classes nhiều ảnh")
        print(f"      - Cần cải thiện performance cho minority classes")
    
    # Gợi ý cải thiện
    if imbalance_ratio < 0.5:
        print(f"   💡 GỢI Ý CẢI THIỆN:")
        print(f"      - Tăng data augmentation cho classes ít ảnh")
        print(f"      - Sử dụng class weights trong loss function")
        print(f"      - Thử nghiệm oversampling/undersampling")
        print(f"      - Điều chỉnh N_WAY để đảm bảo đại diện đủ classes")

def plot_imbalance_analysis(metrics, class_names, save_path="imbalance_analysis.png"):
    """
    Vẽ đồ thị phân tích imbalance
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(OUTPUT_FOLDER):
        save_path = os.path.join(OUTPUT_FOLDER, save_path)
    
    support_per_class = metrics['support_per_class']
    f1_per_class = metrics['f1_per_class']
    precision_per_class = metrics['precision_per_class']
    recall_per_class = metrics['recall_per_class']
    
    # Tạo figure với subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PHÂN TÍCH ẢNH HƯỞNG DATASET IMBALANCE', fontsize=16, fontweight='bold')
    
    # 1. Support vs F1-Score
    colors = ['red' if sup <= max(support_per_class) * 0.3 else 'blue' for sup in support_per_class]
    scatter = ax1.scatter(support_per_class, f1_per_class, c=colors, s=100, alpha=0.7)
    ax1.set_xlabel('Số lượng Samples (Support)', fontsize=12)
    ax1.set_ylabel('F1-Score', fontsize=12)
    ax1.set_title('F1-Score vs Số lượng Samples', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Thêm trend line
    z = np.polyfit(support_per_class, f1_per_class, 1)
    p = np.poly1d(z)
    ax1.plot(support_per_class, p(support_per_class), "r--", alpha=0.8, label=f'Trend line')
    ax1.legend()
    
    # 2. Support distribution
    bars = ax2.bar(range(len(class_names)), support_per_class, color=colors, alpha=0.7)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Số lượng Samples', fontsize=12)
    ax2.set_title('Phân bố Số lượng Samples theo Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars, support_per_class):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(support_per_class)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Performance comparison (Minority vs Majority)
    minority_indices = [i for i, sup in enumerate(support_per_class) if sup <= max(support_per_class) * 0.3]
    majority_indices = [i for i, sup in enumerate(support_per_class) if sup > max(support_per_class) * 0.3]
    
    if minority_indices and majority_indices:
        minority_f1 = [f1_per_class[i] for i in minority_indices]
        majority_f1 = [f1_per_class[i] for i in majority_indices]
        
        data = [minority_f1, majority_f1]
        labels = ['Minority Classes', 'Majority Classes']
        colors_box = ['lightcoral', 'lightblue']
        
        box_plot = ax3.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors_box):
            patch.set_facecolor(color)
        
        ax3.set_title('So sánh F1-Score: Minority vs Majority', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1-Score', fontsize=12)
        ax3.grid(True, alpha=0.3)
    
    # 4. Precision vs Recall scatter
    scatter2 = ax4.scatter(precision_per_class, recall_per_class, c=colors, s=100, alpha=0.7)
    ax4.set_xlabel('Precision', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Precision vs Recall theo Class', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Thêm tên class cho một số điểm
    for i, (name, prec, rec) in enumerate(zip(class_names, precision_per_class, recall_per_class)):
        if support_per_class[i] <= max(support_per_class) * 0.3:  # Chỉ label minority classes
            ax4.annotate(name, (prec, rec), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    # Thêm legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Minority Classes'),
                      Patch(facecolor='blue', alpha=0.7, label='Majority Classes')]
    ax4.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()
    print(f"📊 Đồ thị phân tích imbalance đã được lưu vào: {save_path}")

def print_detailed_evaluation_metrics(metrics, class_names, dataset_name="Dataset"):
    """
    In ra các metrics đánh giá chi tiết
    """
    print(f"\n📊 ĐÁNH GIÁ CHI TIẾT - {dataset_name.upper()}:")
    print("=" * 60)
    
    # Overall metrics
    print("🎯 METRICS TỔNG QUAN:")
    print(f"   Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"   Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"   Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"   Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"   Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"   Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    
    # Imbalance analysis
    analyze_imbalance_impact(metrics, class_names)
    
    print("\n📈 METRICS THEO TỪNG CLASS:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        print(f"   {class_name}:")
        print(f"     Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"     Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"     F1-Score: {metrics['f1_per_class'][i]:.4f}")
        print(f"     Support: {metrics['support_per_class'][i]}")
    
    # Confusion matrix summary
    cm = metrics['confusion_matrix']
    print(f"\n🔍 CONFUSION MATRIX SUMMARY:")
    print(f"   True Positives (TP): {np.sum(np.diag(cm))}")
    print(f"   False Positives (FP): {np.sum(cm) - np.sum(np.diag(cm))}")
    print(f"   Total Predictions: {np.sum(cm)}")
    
    # Class performance ranking
    f1_scores = metrics['f1_per_class']
    class_ranking = sorted(zip(class_names, f1_scores), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 XẾP HẠNG HIỆU SUẤT THEO F1-SCORE:")
    for rank, (class_name, f1) in enumerate(class_ranking, 1):
        print(f"   {rank}. {class_name}: {f1:.4f}")

# ==== TIỀN XỬ LÝ ẢNH ====
def create_transforms():
    """
    Tạo transforms sử dụng cấu hình từ config
    """
    # Transform cơ bản cho validation/test
    transform_basic = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform với data augmentation cho training
    transform_augmented = transforms.Compose([
                transforms.Resize((IMAGE_SIZE + AUGMENTATION_CONFIG['random_crop_size'],
                                IMAGE_SIZE + AUGMENTATION_CONFIG['random_crop_size'])),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG['flip_probability']),
                transforms.RandomRotation(degrees=AUGMENTATION_CONFIG['rotation_degrees']),
                transforms.ColorJitter(
                    brightness=AUGMENTATION_CONFIG['color_jitter']['brightness'],
                    contrast=AUGMENTATION_CONFIG['color_jitter']['contrast'],
                    saturation=AUGMENTATION_CONFIG['color_jitter']['saturation'],
                    hue=AUGMENTATION_CONFIG['color_jitter']['hue']
                ),
                transforms.RandomGrayscale(p=AUGMENTATION_CONFIG['grayscale_probability']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform cho inference (không có augmentation)
    transform_inference = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform_basic, transform_augmented, transform_inference

# Tạo transforms
transform_basic, transform_augmented, transform_inference = create_transforms()

# ==== MÔ HÌNH BACKBONE ====
class TransformerBackbone(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.encoder = create_model('vit_base_patch16_224', pretrained=True)
        self.encoder.head = nn.Identity()
        self.project = nn.Linear(768, out_dim)

    def forward(self, x):
        return self.project(self.encoder(x))

# ==== KHOẢNG CÁCH EUCLIDEAN ====
def euclidean_distance(a, b):
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return torch.pow(a - b, 2).sum(2)

# ==== TÍNH PROTOTYPE ====
def compute_prototypes(support_embeddings, support_labels, n_classes):
    prototypes = []
    for c in range(n_classes):
        class_embeddings = support_embeddings[support_labels == c]
        prototype = class_embeddings.mean(0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

# ==== TẠO EPISODE TỪ DATASET ====
class FewShotDataset:
    def __init__(self, root_dir, transform_train=None, transform_test=None):
        self.root_dir = root_dir
        self.transform_train = transform_train
        self.transform_test = transform_test
        
        # Tạo dataset với transform cơ bản để lấy thông tin
        self.dataset = ImageFolder(root_dir, transform=transforms.ToTensor())
        self.class_to_indices = self._group_by_class()

    def _group_by_class(self):
        class_to_idx = {}
        for idx, (img_path, label) in enumerate(self.dataset.samples):
            class_to_idx.setdefault(label, []).append(idx)
        return class_to_idx

    def load_image_with_transform(self, img_path, use_augmentation=True):
        """Load ảnh với transform tương ứng"""
        img = Image.open(img_path).convert('RGB')
        if use_augmentation and self.transform_train:
            return self.transform_train(img)
        elif self.transform_test:
            return self.transform_test(img)
        else:
            return transforms.ToTensor()(img)

    def sample_episode(self, n_way, k_shot, q_query, q_valid=0, use_augmentation=True):
        selected_classes = random.sample(list(self.class_to_indices.keys()), n_way)
        support_idx, query_idx, valid_idx = [], [], []
        label_map = {}
        
        # Lưu thông tin class được chọn
        self.selected_class_names = [self.dataset.classes[class_id] for class_id in selected_classes]

        for new_label, class_id in enumerate(selected_classes):
            indices = self.class_to_indices[class_id]
            total_needed = k_shot + q_query + q_valid
            sampled = random.sample(indices, total_needed)
            support = sampled[:k_shot]
            query = sampled[k_shot:k_shot + q_query]
            valid = sampled[k_shot + q_query:] if q_valid > 0 else []

            support_idx += support
            query_idx += query
            valid_idx += valid
            label_map[class_id] = new_label

        # Tạo support, query và validation sets với augmentation
        support_set = []
        query_set = []
        valid_set = []
        
        for i in support_idx:
            img_path, label = self.dataset.samples[i]
            img_tensor = self.load_image_with_transform(img_path, use_augmentation=True)
            support_set.append((img_tensor, label_map[label]))
            
        for i in query_idx:
            img_path, label = self.dataset.samples[i]
            img_tensor = self.load_image_with_transform(img_path, use_augmentation=False)
            query_set.append((img_tensor, label_map[label]))
            
        for i in valid_idx:
            img_path, label = self.dataset.samples[i]
            img_tensor = self.load_image_with_transform(img_path, use_augmentation=False)
            valid_set.append((img_tensor, label_map[label]))
            
        return support_set, query_set, valid_set

# ==== CHẠY MỘT EPISODE VỚI ĐÁNH GIÁ CHI TIẾT ====
def run_episode_with_detailed_evaluation(model, dataset, use_augmentation=True, include_validation=False):
    """
    Chạy một episode với đánh giá chi tiết
    """
    model.eval()
    
    if include_validation and USE_VALIDATION:
        support_set, query_set, valid_set = dataset.sample_episode(N_WAY, K_SHOT, Q_QUERY, Q_VALID, use_augmentation)
        support_imgs, support_labels = zip(*support_set)
        query_imgs, query_labels = zip(*query_set)
        valid_imgs, valid_labels = zip(*valid_set)
        
        # Lấy tên class thực tế được sử dụng trong episode này
        episode_class_names = dataset.selected_class_names

        support_imgs = torch.stack(support_imgs).to(DEVICE)
        support_labels = torch.tensor(support_labels).to(DEVICE)
        query_imgs = torch.stack(query_imgs).to(DEVICE)
        query_labels = torch.tensor(query_labels).to(DEVICE)
        valid_imgs = torch.stack(valid_imgs).to(DEVICE)
        valid_labels = torch.tensor(valid_labels).to(DEVICE)

        with torch.no_grad():
            support_emb = model(support_imgs)
            query_emb = model(query_imgs)
            valid_emb = model(valid_imgs)
            prototypes = compute_prototypes(support_emb, support_labels, N_WAY)
            
            # Query metrics
            query_dists = euclidean_distance(query_emb, prototypes)
            query_log_p_y = F.log_softmax(-query_dists, dim=1)
            query_loss = F.nll_loss(query_log_p_y, query_labels)
            query_acc = (query_log_p_y.argmax(1) == query_labels).float().mean()
            query_predictions = query_log_p_y.argmax(1).cpu().numpy()
            query_targets = query_labels.cpu().numpy()
            
            # Validation metrics
            valid_dists = euclidean_distance(valid_emb, prototypes)
            valid_log_p_y = F.log_softmax(-valid_dists, dim=1)
            valid_loss = F.nll_loss(valid_log_p_y, valid_labels)
            valid_acc = (valid_log_p_y.argmax(1) == valid_labels).float().mean()
            valid_predictions = valid_log_p_y.argmax(1).cpu().numpy()
            valid_targets = valid_labels.cpu().numpy()

        return {
            'query_loss': query_loss.item(),
            'query_acc': query_acc.item(),
            'query_predictions': query_predictions,
            'query_targets': query_targets,
            'valid_loss': valid_loss.item(),
            'valid_acc': valid_acc.item(),
            'valid_predictions': valid_predictions,
            'valid_targets': valid_targets,
            'episode_class_names': episode_class_names
        }
    else:
        support_set, query_set, _ = dataset.sample_episode(N_WAY, K_SHOT, Q_QUERY, 0, use_augmentation)
        support_imgs, support_labels = zip(*support_set)
        query_imgs, query_labels = zip(*query_set)
        
        # Lấy tên class thực tế được sử dụng trong episode này
        episode_class_names = dataset.selected_class_names

    support_imgs = torch.stack(support_imgs).to(DEVICE)
    support_labels = torch.tensor(support_labels).to(DEVICE)
    query_imgs = torch.stack(query_imgs).to(DEVICE)
    query_labels = torch.tensor(query_labels).to(DEVICE)

    with torch.no_grad():
        support_emb = model(support_imgs)
        query_emb = model(query_imgs)
        prototypes = compute_prototypes(support_emb, support_labels, N_WAY)
        dists = euclidean_distance(query_emb, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_p_y, query_labels)
        acc = (log_p_y.argmax(1) == query_labels).float().mean()
        predictions = log_p_y.argmax(1).cpu().numpy()
        targets = query_labels.cpu().numpy()

        return {
            'query_loss': loss.item(),
            'query_acc': acc.item(),
            'query_predictions': predictions,
            'query_targets': targets,
            'valid_loss': None,
            'valid_acc': None,
            'valid_predictions': None,
            'valid_targets': None,
            'episode_class_names': episode_class_names
        }

# ==== CHẠY NHIỀU EPISODES VỚI ĐÁNH GIÁ CHI TIẾT ====
def run_multiple_episodes_with_detailed_evaluation(model, dataset, num_episodes, use_augmentation=True, include_validation=False):
    """
    Chạy nhiều episodes với đánh giá chi tiết
    """
    query_losses = []
    query_accuracies = []
    valid_losses = []
    valid_accuracies = []
    
    # Thu thập predictions và targets cho đánh giá tổng hợp
    all_query_predictions = []
    all_query_targets = []
    all_valid_predictions = []
    all_valid_targets = []
    
    # Thu thập tên class được sử dụng trong các episodes
    all_episode_class_names = []
    
    print(f"🔄 Đang chạy {num_episodes} episodes với đánh giá chi tiết...")
    
    for episode in range(num_episodes):
        results = run_episode_with_detailed_evaluation(model, dataset, use_augmentation, include_validation)
        query_losses.append(results['query_loss'])
        query_accuracies.append(results['query_acc'])
        
        # Thu thập predictions và targets
        all_query_predictions.extend(results['query_predictions'])
        all_query_targets.extend(results['query_targets'])
        
        # Thu thập tên class
        all_episode_class_names.append(results['episode_class_names'])
        
        if results['valid_loss'] is not None:
            valid_losses.append(results['valid_loss'])
            valid_accuracies.append(results['valid_acc'])
            all_valid_predictions.extend(results['valid_predictions'])
            all_valid_targets.extend(results['valid_targets'])
        
        # In tiến độ theo cấu hình
        if DISPLAY_PROGRESS and ((episode + 1) % 5 == 0 or episode == 0):
            if results['valid_loss'] is not None:
                print(f"   Episode {episode+1}/{num_episodes}: Q_Loss={results['query_loss']:.4f}, Q_Acc={results['query_acc']:.4f}, V_Loss={results['valid_loss']:.4f}, V_Acc={results['valid_acc']:.4f}")
                print(f"      Classes: {results['episode_class_names']}")
            else:
                print(f"   Episode {episode+1}/{num_episodes}: Loss={results['query_loss']:.4f}, Acc={results['query_acc']:.4f}")
                print(f"      Classes: {results['episode_class_names']}")
    
    # Tính thống kê query
    avg_query_loss = np.mean(query_losses)
    avg_query_acc = np.mean(query_accuracies)
    std_query_loss = np.std(query_losses)
    std_query_acc = np.std(query_accuracies)
    min_query_acc = np.min(query_accuracies)
    max_query_acc = np.max(query_accuracies)
    
    result = {
        'query_losses': query_losses,
        'query_accuracies': query_accuracies,
        'avg_query_loss': avg_query_loss,
        'avg_query_acc': avg_query_acc,
        'std_query_loss': std_query_loss,
        'std_query_acc': std_query_acc,
        'min_query_acc': min_query_acc,
        'max_query_acc': max_query_acc,
        'all_query_predictions': np.array(all_query_predictions),
        'all_query_targets': np.array(all_query_targets),
        'all_episode_class_names': all_episode_class_names
    }
    
    # Tính thống kê validation nếu có
    if valid_losses:
        avg_valid_loss = np.mean(valid_losses)
        avg_valid_acc = np.mean(valid_accuracies)
        std_valid_loss = np.std(valid_losses)
        std_valid_acc = np.std(valid_accuracies)
        min_valid_acc = np.min(valid_accuracies)
        max_valid_acc = np.max(valid_accuracies)
        
        result.update({
            'valid_losses': valid_losses,
            'valid_accuracies': valid_accuracies,
            'avg_valid_loss': avg_valid_loss,
            'avg_valid_acc': avg_valid_acc,
            'std_valid_loss': std_valid_loss,
            'std_valid_acc': std_valid_acc,
            'min_valid_acc': min_valid_acc,
            'max_valid_acc': max_valid_acc,
            'all_valid_predictions': np.array(all_valid_predictions),
            'all_valid_targets': np.array(all_valid_targets)
        })
    
    return result

# ==== VẼ ĐỒ THỊ KẾT QUẢ ====
def plot_episode_results(results_with_aug, results_without_aug, save_path="episode_results.png"):
    """
    Vẽ đồ thị so sánh kết quả với và không có data augmentation
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(OUTPUT_FOLDER):
        save_path = os.path.join(OUTPUT_FOLDER, save_path)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SO SÁNH KẾT QUẢ VỚI VÀ KHÔNG CÓ DATA AUGMENTATION', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    episodes = range(1, len(results_with_aug['accuracies']) + 1)
    ax1.plot(episodes, results_with_aug['accuracies'], 'b-o', label='Với Augmentation', alpha=0.7)
    ax1.plot(episodes, results_without_aug['accuracies'], 'r-s', label='Không Augmentation', alpha=0.7)
    ax1.set_title('Accuracy theo từng Episode', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss comparison
    ax2.plot(episodes, results_with_aug['losses'], 'b-o', label='Với Augmentation', alpha=0.7)
    ax2.plot(episodes, results_without_aug['losses'], 'r-s', label='Không Augmentation', alpha=0.7)
    ax2.set_title('Loss theo từng Episode', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot accuracy
    acc_data = [results_with_aug['accuracies'], results_without_aug['accuracies']]
    labels = ['Với Augmentation', 'Không Augmentation']
    box_plot = ax3.boxplot(acc_data, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    ax3.set_title('Phân bố Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    stats_data = [
        ['Metric', 'Với Augmentation', 'Không Augmentation'],
        ['Avg Accuracy', f"{results_with_aug['avg_acc']:.4f}", f"{results_without_aug['avg_acc']:.4f}"],
        ['Std Accuracy', f"{results_with_aug['std_acc']:.4f}", f"{results_without_aug['std_acc']:.4f}"],
        ['Min Accuracy', f"{results_with_aug['min_acc']:.4f}", f"{results_without_aug['min_acc']:.4f}"],
        ['Max Accuracy', f"{results_with_aug['max_acc']:.4f}", f"{results_without_aug['max_acc']:.4f}"],
        ['Avg Loss', f"{results_with_aug['avg_loss']:.4f}", f"{results_without_aug['avg_loss']:.4f}"],
        ['Std Loss', f"{results_with_aug['std_loss']:.4f}", f"{results_without_aug['std_loss']:.4f}"]
    ]
    
    table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Tô màu header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Thống kê so sánh', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Đồ thị kết quả đã được lưu vào: {save_path}")

# ==== VẼ ĐỒ THỊ KẾT QUẢ ĐƠN LẺ ====
def plot_single_results(results_with_aug, save_path="episode_results_single.png"):
    """
    Vẽ đồ thị chỉ cho kết quả với data augmentation
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(OUTPUT_FOLDER):
        save_path = os.path.join(OUTPUT_FOLDER, save_path)
    has_validation = 'valid_accuracies' in results_with_aug
    
    if has_validation:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('KẾT QUẢ FEW-SHOT LEARNING VỚI DATA AUGMENTATION (CÓ VALIDATION)', fontsize=16, fontweight='bold')
        
        # 1. Accuracy theo episodes (Query vs Validation)
        episodes = range(1, len(results_with_aug['query_accuracies']) + 1)
        ax1.plot(episodes, results_with_aug['query_accuracies'], 'b-o', alpha=0.7, linewidth=2, markersize=6, label='Query')
        ax1.plot(episodes, results_with_aug['valid_accuracies'], 'g-s', alpha=0.7, linewidth=2, markersize=6, label='Validation')
        ax1.set_title('Accuracy theo từng Episode', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Thêm đường trung bình
        avg_query_acc = results_with_aug['avg_query_acc']
        avg_valid_acc = results_with_aug['avg_valid_acc']
        ax1.axhline(y=avg_query_acc, color='blue', linestyle='--', alpha=0.7, label=f'Query TB: {avg_query_acc:.3f}')
        ax1.axhline(y=avg_valid_acc, color='green', linestyle='--', alpha=0.7, label=f'Valid TB: {avg_valid_acc:.3f}')
        ax1.legend()
        
        # 2. Loss theo episodes (Query vs Validation)
        ax2.plot(episodes, results_with_aug['query_losses'], 'r-o', alpha=0.7, linewidth=2, markersize=6, label='Query')
        ax2.plot(episodes, results_with_aug['valid_losses'], 'm-s', alpha=0.7, linewidth=2, markersize=6, label='Validation')
        ax2.set_title('Loss theo từng Episode', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Thêm đường trung bình
        avg_query_loss = results_with_aug['avg_query_loss']
        avg_valid_loss = results_with_aug['avg_valid_loss']
        ax2.axhline(y=avg_query_loss, color='red', linestyle='--', alpha=0.7, label=f'Query TB: {avg_query_loss:.3f}')
        ax2.axhline(y=avg_valid_loss, color='magenta', linestyle='--', alpha=0.7, label=f'Valid TB: {avg_valid_loss:.3f}')
        ax2.legend()
        
        # 3. Histogram accuracy (Query vs Validation)
        ax3.hist(results_with_aug['query_accuracies'], bins=15, color='lightblue', edgecolor='navy', alpha=0.7, label='Query')
        ax3.hist(results_with_aug['valid_accuracies'], bins=15, color='lightgreen', edgecolor='darkgreen', alpha=0.7, label='Validation')
        ax3.set_title('Phân bố Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Accuracy', fontsize=12)
        ax3.set_ylabel('Số Episode', fontsize=12)
        ax3.axvline(x=avg_query_acc, color='blue', linestyle='--', alpha=0.7, label=f'Query TB: {avg_query_acc:.3f}')
        ax3.axvline(x=avg_valid_acc, color='green', linestyle='--', alpha=0.7, label=f'Valid TB: {avg_valid_acc:.3f}')
        ax3.legend()
        
        # 4. Statistics table
        ax4.axis('tight')
        ax4.axis('off')
        
        stats_data = [
            ['Metric', 'Query', 'Validation'],
            ['Avg Accuracy', f"{avg_query_acc:.4f}", f"{avg_valid_acc:.4f}"],
            ['Std Accuracy', f"{results_with_aug['std_query_acc']:.4f}", f"{results_with_aug['std_valid_acc']:.4f}"],
            ['Min Accuracy', f"{results_with_aug['min_query_acc']:.4f}", f"{results_with_aug['min_valid_acc']:.4f}"],
            ['Max Accuracy', f"{results_with_aug['max_query_acc']:.4f}", f"{results_with_aug['max_valid_acc']:.4f}"],
            ['Avg Loss', f"{avg_query_loss:.4f}", f"{avg_valid_loss:.4f}"],
            ['Std Loss', f"{results_with_aug['std_query_loss']:.4f}", f"{results_with_aug['std_valid_loss']:.4f}"],
            ['Tổng Episodes', f"{len(results_with_aug['query_accuracies'])}", f"{len(results_with_aug['valid_accuracies'])}"]
        ]
        
        table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Tô màu header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Thống kê kết quả', fontsize=14, fontweight='bold', pad=20)
        
    else:
        # Fallback cho trường hợp không có validation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('KẾT QUẢ FEW-SHOT LEARNING VỚI DATA AUGMENTATION', fontsize=16, fontweight='bold')
        
        # 1. Accuracy theo episodes
        episodes = range(1, len(results_with_aug['query_accuracies']) + 1)
        ax1.plot(episodes, results_with_aug['query_accuracies'], 'b-o', alpha=0.7, linewidth=2, markersize=6)
        ax1.set_title('Accuracy theo từng Episode', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Thêm đường trung bình
        avg_acc = results_with_aug['avg_query_acc']
        ax1.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.7, label=f'Trung bình: {avg_acc:.3f}')
        ax1.legend()
        
        # 2. Loss theo episodes
        ax2.plot(episodes, results_with_aug['query_losses'], 'r-s', alpha=0.7, linewidth=2, markersize=6)
        ax2.set_title('Loss theo từng Episode', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Thêm đường trung bình
        avg_loss = results_with_aug['avg_query_loss']
        ax2.axhline(y=avg_loss, color='blue', linestyle='--', alpha=0.7, label=f'Trung bình: {avg_loss:.3f}')
        ax2.legend()
        
        # 3. Histogram accuracy
        ax3.hist(results_with_aug['query_accuracies'], bins=15, color='lightblue', edgecolor='navy', alpha=0.7)
        ax3.set_title('Phân bố Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Accuracy', fontsize=12)
        ax3.set_ylabel('Số Episode', fontsize=12)
        ax3.axvline(x=avg_acc, color='red', linestyle='--', alpha=0.7, label=f'Trung bình: {avg_acc:.3f}')
        ax3.legend()
        
        # 4. Statistics table
        ax4.axis('tight')
        ax4.axis('off')
        
        stats_data = [
            ['Metric', 'Giá trị'],
            ['Avg Accuracy', f"{avg_acc:.4f}"],
            ['Std Accuracy', f"{results_with_aug['std_query_acc']:.4f}"],
            ['Min Accuracy', f"{results_with_aug['min_query_acc']:.4f}"],
            ['Max Accuracy', f"{results_with_aug['max_query_acc']:.4f}"],
            ['Avg Loss', f"{avg_loss:.4f}"],
            ['Std Loss', f"{results_with_aug['std_query_loss']:.4f}"],
            ['Tổng Episodes', f"{len(results_with_aug['query_accuracies'])}"]
        ]
        
        table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Tô màu header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Thống kê kết quả', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Đồ thị kết quả đã được lưu vào: {save_path}")

# ==== MAIN ====
if __name__ == "__main__":
    print("🚀 Khởi tạo Few-Shot Learning với Data Augmentation")
    print("=" * 60)
    
    # Hiển thị thông tin cấu hình
    print("📋 THÔNG TIN CẤU HÌNH:")
    print(f"   Dataset: {DATASET_PATH}")
    print(f"   Few-Shot: {N_WAY}-way, {K_SHOT}-shot, {Q_QUERY}-query, {Q_VALID}-valid")
    print(f"   Episodes: {NUM_EPISODES}")
    print(f"   Use validation: {USE_VALIDATION}")
    print(f"   Compare without augmentation: {COMPARE_WITHOUT_AUG}")
    print(f"   Device: {DEVICE}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Embedding dim: {EMBED_DIM}")
    print("=" * 60)
    
    # Phân tích và vẽ đồ thị dataset
    print("📈 PHÂN TÍCH DATASET:")
    if DETAILED_ANALYSIS:
        print("🔍 Chạy phân tích chi tiết...")
        dataset_info = analyze_dataset_detailed(root_dir, OUTPUT_FOLDER)
    else:
        print("🔍 Chạy phân tích cơ bản...")
        dataset_info = analyze_and_visualize_dataset(root_dir)
    
    if dataset_info is None:
        print("❌ Không thể phân tích dataset. Thoát chương trình.")
        exit()
    
    # Kiểm tra xem dataset có đủ dữ liệu cho few-shot learning không
    if dataset_info['total_images'] < N_WAY * (K_SHOT + Q_QUERY):
        print(f"⚠️ Cảnh báo: Dataset có thể không đủ dữ liệu cho {N_WAY}-way {K_SHOT}-shot learning")
        print(f"   Cần ít nhất {N_WAY * (K_SHOT + Q_QUERY)} ảnh, hiện có {dataset_info['total_images']} ảnh")
    
    print("\n" + "=" * 60)
    
    # Khởi tạo dataset với data augmentation
    fewshot_data = FewShotDataset(
        root_dir, 
        transform_train=transform_augmented,
        transform_test=transform_basic
    )
    
    model = TransformerBackbone().to(DEVICE)
    print(f"✅ Mô hình đã được tải lên {DEVICE}")
    print(f"📊 Cấu hình: {N_WAY}-way, {K_SHOT}-shot, {Q_QUERY}-query")
    print("=" * 60)

    # Chạy episodes với data augmentation
    print(f"\n🎯 CHẠY {NUM_EPISODES} EPISODES VỚI DATA AUGMENTATION:")
    results_with_aug = run_multiple_episodes_with_detailed_evaluation(model, fewshot_data, NUM_EPISODES, use_augmentation=True, include_validation=USE_VALIDATION)
    
    print(f"\n📊 KẾT QUẢ VỚI DATA AUGMENTATION:")
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
    print(f"🎯 Sử dụng {N_WAY} class trong mỗi episode (chọn ngẫu nhiên)")
    
    # Hiển thị các class được sử dụng trong các episodes
    print(f"📊 Các class được sử dụng trong {NUM_EPISODES} episodes:")
    for i, episode_classes in enumerate(results_with_aug['all_episode_class_names']):
        print(f"   Episode {i+1}: {episode_classes}")
    
    # Sử dụng tên class thực tế từ episode cuối cùng cho đánh giá
    # Điều này sẽ hiển thị tên folder thực tế trong confusion matrix và accuracy plots
    class_names = results_with_aug['all_episode_class_names'][-1]  # Lấy tên class từ episode cuối
    
    # Tính metrics chi tiết
    query_metrics = calculate_detailed_metrics(query_predictions, query_targets, N_WAY)
    
    # In metrics
    print_detailed_evaluation_metrics(query_metrics, class_names, "Query Set")
    
    # Vẽ confusion matrix
    if SAVE_RESULTS:
        plot_confusion_matrix(query_metrics['confusion_matrix'], class_names, "query_confusion_matrix.png")
        analyze_accuracy_by_class(query_predictions, query_targets, class_names, "query_accuracy_by_class.png")
        plot_imbalance_analysis(query_metrics, class_names, "query_imbalance_analysis.png")
    
    # ==== ĐÁNH GIÁ CHI TIẾT VALIDATION SET (nếu có) ====
    if 'all_valid_predictions' in results_with_aug:
        print(f"\n🔍 ĐÁNH GIÁ CHI TIẾT VALIDATION SET:")
        valid_predictions = results_with_aug['all_valid_predictions']
        valid_targets = results_with_aug['all_valid_targets']
        
        # Tính metrics chi tiết
        valid_metrics = calculate_detailed_metrics(valid_predictions, valid_targets, N_WAY)
        
        # In metrics (sử dụng cùng tên class thực tế)
        print_detailed_evaluation_metrics(valid_metrics, class_names, "Validation Set")
        
        # Vẽ confusion matrix
        if SAVE_RESULTS:
            plot_confusion_matrix(valid_metrics['confusion_matrix'], class_names, "valid_confusion_matrix.png")
            analyze_accuracy_by_class(valid_predictions, valid_targets, class_names, "valid_accuracy_by_class.png")
            plot_imbalance_analysis(valid_metrics, class_names, "valid_imbalance_analysis.png")
    
    # Kiểm tra có cần so sánh với episodes không có data augmentation không
    if COMPARE_WITHOUT_AUG:
        # Chạy episodes không có data augmentation để so sánh
        print(f"\n🔍 CHẠY {NUM_EPISODES} EPISODES KHÔNG CÓ DATA AUGMENTATION:")
        results_without_aug = run_multiple_episodes_with_detailed_evaluation(model, fewshot_data, NUM_EPISODES, use_augmentation=False, include_validation=USE_VALIDATION)
        
        print(f"\n📊 KẾT QUẢ KHÔNG CÓ DATA AUGMENTATION:")
        print(f"   Query Accuracy trung bình: {results_without_aug['avg_query_acc']:.4f} ± {results_without_aug['std_query_acc']:.4f}")
        print(f"   Query Loss trung bình: {results_without_aug['avg_query_loss']:.4f} ± {results_without_aug['std_query_loss']:.4f}")
        print(f"   Query Accuracy min/max: {results_without_aug['min_query_acc']:.4f} / {results_without_aug['max_query_acc']:.4f}")
        
        if 'avg_valid_acc' in results_without_aug:
            print(f"   Validation Accuracy trung bình: {results_without_aug['avg_valid_acc']:.4f} ± {results_without_aug['std_valid_acc']:.4f}")
            print(f"   Validation Loss trung bình: {results_without_aug['avg_valid_loss']:.4f} ± {results_without_aug['std_valid_loss']:.4f}")
            print(f"   Validation Accuracy min/max: {results_without_aug['min_valid_acc']:.4f} / {results_without_aug['max_valid_acc']:.4f}")
        
        # ==== ĐÁNH GIÁ CHI TIẾT QUERY SET (KHÔNG AUGMENTATION) ====
        print(f"\n🔍 ĐÁNH GIÁ CHI TIẾT QUERY SET (KHÔNG AUGMENTATION):")
        query_predictions_no_aug = results_without_aug['all_query_predictions']
        query_targets_no_aug = results_without_aug['all_query_targets']
        
        # Lấy tên class thực tế từ episode cuối của kết quả không augmentation
        class_names_no_aug = results_without_aug['all_episode_class_names'][-1]
        
        # Tính metrics chi tiết
        query_metrics_no_aug = calculate_detailed_metrics(query_predictions_no_aug, query_targets_no_aug, N_WAY)
        
        # In metrics (sử dụng tên class thực tế)
        print_detailed_evaluation_metrics(query_metrics_no_aug, class_names_no_aug, "Query Set (No Augmentation)")
        
        # Vẽ confusion matrix
        if SAVE_RESULTS:
            plot_confusion_matrix(query_metrics_no_aug['confusion_matrix'], class_names_no_aug, "query_confusion_matrix_no_aug.png")
            analyze_accuracy_by_class(query_predictions_no_aug, query_targets_no_aug, class_names_no_aug, "query_accuracy_by_class_no_aug.png")
            plot_imbalance_analysis(query_metrics_no_aug, class_names_no_aug, "query_imbalance_analysis_no_aug.png")
        
        # ==== ĐÁNH GIÁ CHI TIẾT VALIDATION SET (KHÔNG AUGMENTATION) ====
        if 'all_valid_predictions' in results_without_aug:
            print(f"\n🔍 ĐÁNH GIÁ CHI TIẾT VALIDATION SET (KHÔNG AUGMENTATION):")
            valid_predictions_no_aug = results_without_aug['all_valid_predictions']
            valid_targets_no_aug = results_without_aug['all_valid_targets']
            
            # Tính metrics chi tiết
            valid_metrics_no_aug = calculate_detailed_metrics(valid_predictions_no_aug, valid_targets_no_aug, N_WAY)
            
            # In metrics (sử dụng tên class thực tế)
            print_detailed_evaluation_metrics(valid_metrics_no_aug, class_names_no_aug, "Validation Set (No Augmentation)")
            
            # Vẽ confusion matrix
            if SAVE_RESULTS:
                plot_confusion_matrix(valid_metrics_no_aug['confusion_matrix'], class_names_no_aug, "valid_confusion_matrix_no_aug.png")
                analyze_accuracy_by_class(valid_predictions_no_aug, valid_targets_no_aug, class_names_no_aug, "valid_accuracy_by_class_no_aug.png")
                plot_imbalance_analysis(valid_metrics_no_aug, class_names_no_aug, "valid_imbalance_analysis_no_aug.png")
        
        # So sánh hiệu quả chi tiết
        print(f"\n📈 SO SÁNH HIỆU QUẢ CHI TIẾT:")
        acc_improvement = results_with_aug['avg_query_acc'] - results_without_aug['avg_query_acc']
        f1_improvement = query_metrics['macro_f1'] - query_metrics_no_aug['macro_f1']
        precision_improvement = query_metrics['macro_precision'] - query_metrics_no_aug['macro_precision']
        recall_improvement = query_metrics['macro_recall'] - query_metrics_no_aug['macro_recall']
        
        print(f"   Cải thiện Query Accuracy: {acc_improvement:.4f}")
        print(f"   Cải thiện Macro F1-Score: {f1_improvement:.4f}")
        print(f"   Cải thiện Macro Precision: {precision_improvement:.4f}")
        print(f"   Cải thiện Macro Recall: {recall_improvement:.4f}")
        
        if acc_improvement > 0:
            print(f"   ✅ Data augmentation có hiệu quả tích cực!")
        else:
            print(f"   ⚠️ Data augmentation chưa cải thiện hiệu suất")
        
        # Vẽ đồ thị so sánh
        if SAVE_RESULTS:
            plot_episode_results(results_with_aug, results_without_aug)
    else:
        print(f"\n⏭️ Bỏ qua so sánh với episodes không có data augmentation (COMPARE_WITHOUT_AUG = False)")
        
        # Vẽ đồ thị chỉ cho kết quả với augmentation
        if SAVE_RESULTS:
            plot_single_results(results_with_aug)
    
    print("\n✅ Hoàn thành!")
    print(f"📁 Tất cả kết quả đã được lưu trong folder: {OUTPUT_FOLDER}")
    print("=" * 60)
    
    if DETAILED_ANALYSIS:
        print(f"📊 Đồ thị phân tích chi tiết: {OUTPUT_FOLDER}/detailed_analysis.png")
        print(f"📊 Đồ thị định dạng file: {OUTPUT_FOLDER}/file_formats_analysis.png")
    else:
        print(f"📊 Đồ thị phân tích cơ bản: {OUTPUT_FOLDER}/dataset_analysis.png")
    
    if SAVE_RESULTS:
        if COMPARE_WITHOUT_AUG:
            print(f"📊 Đồ thị kết quả episodes: {OUTPUT_FOLDER}/episode_results.png")
            print(f"📊 Confusion matrix (với augmentation): {OUTPUT_FOLDER}/query_confusion_matrix.png")
            print(f"📊 Confusion matrix (không augmentation): {OUTPUT_FOLDER}/query_confusion_matrix_no_aug.png")
            print(f"📊 Accuracy theo class (với augmentation): {OUTPUT_FOLDER}/query_accuracy_by_class.png")
            print(f"📊 Accuracy theo class (không augmentation): {OUTPUT_FOLDER}/query_accuracy_by_class_no_aug.png")
            print(f"📊 Imbalance analysis (với augmentation): {OUTPUT_FOLDER}/query_imbalance_analysis.png")
            print(f"📊 Imbalance analysis (không augmentation): {OUTPUT_FOLDER}/query_imbalance_analysis_no_aug.png")
        else:
            print(f"📊 Đồ thị kết quả episodes: {OUTPUT_FOLDER}/episode_results_single.png")
            print(f"📊 Confusion matrix: {OUTPUT_FOLDER}/query_confusion_matrix.png")
            print(f"📊 Accuracy theo class: {OUTPUT_FOLDER}/query_accuracy_by_class.png")
            print(f"📊 Imbalance analysis: {OUTPUT_FOLDER}/query_imbalance_analysis.png")
        
        if USE_VALIDATION:
            if COMPARE_WITHOUT_AUG:
                print(f"📊 Validation confusion matrix (với augmentation): {OUTPUT_FOLDER}/valid_confusion_matrix.png")
                print(f"📊 Validation confusion matrix (không augmentation): {OUTPUT_FOLDER}/valid_confusion_matrix_no_aug.png")
                print(f"📊 Validation accuracy theo class (với augmentation): {OUTPUT_FOLDER}/valid_accuracy_by_class.png")
                print(f"📊 Validation accuracy theo class (không augmentation): {OUTPUT_FOLDER}/valid_accuracy_by_class_no_aug.png")
                print(f"📊 Validation imbalance analysis (với augmentation): {OUTPUT_FOLDER}/valid_imbalance_analysis.png")
                print(f"📊 Validation imbalance analysis (không augmentation): {OUTPUT_FOLDER}/valid_imbalance_analysis_no_aug.png")
            else:
                print(f"📊 Validation confusion matrix: {OUTPUT_FOLDER}/valid_confusion_matrix.png")
                print(f"📊 Validation accuracy theo class: {OUTPUT_FOLDER}/valid_accuracy_by_class.png")
                print(f"📊 Validation imbalance analysis: {OUTPUT_FOLDER}/valid_imbalance_analysis.png")
    
    print("=" * 60)
