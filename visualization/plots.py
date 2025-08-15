"""
Module cho visualization plots
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def plot_confusion_matrix(cm, class_names, save_path, config):
    """
    Vẽ confusion matrix
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(config['OUTPUT_FOLDER']):
        save_path = os.path.join(config['OUTPUT_FOLDER'], save_path)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=config['PLOT_DPI'], bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print(f"📊 Confusion matrix đã được lưu vào: {save_path}")

def analyze_accuracy_by_class(predictions, targets, class_names, save_path, config):
    """
    Phân tích độ chính xác theo từng class
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(config['OUTPUT_FOLDER']):
        save_path = os.path.join(config['OUTPUT_FOLDER'], save_path)
    
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
    plt.savefig(save_path, dpi=config['PLOT_DPI'], bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print(f"📊 Đồ thị accuracy theo class đã được lưu vào: {save_path}")
    
    return class_accuracies

def plot_imbalance_analysis(metrics, class_names, save_path, config):
    """
    Vẽ đồ thị phân tích imbalance
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(config['OUTPUT_FOLDER']):
        save_path = os.path.join(config['OUTPUT_FOLDER'], save_path)
    
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
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Minority Classes'),
                      Patch(facecolor='blue', alpha=0.7, label='Majority Classes')]
    ax4.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config['PLOT_DPI'], bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print(f"📊 Đồ thị phân tích imbalance đã được lưu vào: {save_path}")

def plot_episode_results(results_with_aug, results_without_aug, save_path, config):
    """
    Vẽ đồ thị so sánh kết quả với và không có data augmentation
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(config['OUTPUT_FOLDER']):
        save_path = os.path.join(config['OUTPUT_FOLDER'], save_path)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SO SÁNH KẾT QUẢ VỚI VÀ KHÔNG CÓ DATA AUGMENTATION', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    episodes = range(1, len(results_with_aug['query_accuracies']) + 1)
    ax1.plot(episodes, results_with_aug['query_accuracies'], 'b-o', label='Với Augmentation', alpha=0.7)
    ax1.plot(episodes, results_without_aug['query_accuracies'], 'r-s', label='Không Augmentation', alpha=0.7)
    ax1.set_title('Accuracy theo từng Episode', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss comparison
    ax2.plot(episodes, results_with_aug['query_losses'], 'b-o', label='Với Augmentation', alpha=0.7)
    ax2.plot(episodes, results_without_aug['query_losses'], 'r-s', label='Không Augmentation', alpha=0.7)
    ax2.set_title('Loss theo từng Episode', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot accuracy
    acc_data = [results_with_aug['query_accuracies'], results_without_aug['query_accuracies']]
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
        ['Avg Accuracy', f"{results_with_aug['avg_query_acc']:.4f}", f"{results_without_aug['avg_query_acc']:.4f}"],
        ['Std Accuracy', f"{results_with_aug['std_query_acc']:.4f}", f"{results_without_aug['std_query_acc']:.4f}"],
        ['Min Accuracy', f"{results_with_aug['min_query_acc']:.4f}", f"{results_without_aug['min_query_acc']:.4f}"],
        ['Max Accuracy', f"{results_with_aug['max_query_acc']:.4f}", f"{results_without_aug['max_query_acc']:.4f}"],
        ['Avg Loss', f"{results_with_aug['avg_query_loss']:.4f}", f"{results_without_aug['avg_query_loss']:.4f}"],
        ['Std Loss', f"{results_with_aug['std_query_loss']:.4f}", f"{results_without_aug['std_query_loss']:.4f}"]
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
    plt.savefig(save_path, dpi=config['PLOT_DPI'], bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print(f"📊 Đồ thị kết quả đã được lưu vào: {save_path}")

def plot_single_results(results_with_aug, save_path, config):
    """
    Vẽ đồ thị chỉ cho kết quả với data augmentation
    """
    # Đảm bảo save_path nằm trong output folder
    if not save_path.startswith(config['OUTPUT_FOLDER']):
        save_path = os.path.join(config['OUTPUT_FOLDER'], save_path)
    
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
    plt.savefig(save_path, dpi=config['PLOT_DPI'], bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình
    if config.get('SHOW_PLOTS', False):
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    print(f"📊 Đồ thị kết quả đã được lưu vào: {save_path}")
