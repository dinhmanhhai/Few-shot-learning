import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import pandas as pd

def analyze_dataset_detailed(dataset_path, save_dir="dataset_analysis"):
    """
    Phân tích chi tiết dataset với nhiều loại đồ thị
    """
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
            format_counts = defaultdict(int)
            
            for file in os.listdir(class_path):
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in image_extensions):
                    count += 1
                    # Xác định định dạng file
                    for ext in image_extensions:
                        if file_lower.endswith(ext):
                            format_counts[ext[1:]] += 1  # Bỏ dấu chấm
                            break
            
            if count > 0:
                class_data[class_name] = {
                    'count': count,
                    'formats': dict(format_counts)
                }
                total_images += count
    
    if not class_data:
        print("❌ Không tìm thấy ảnh nào!")
        return None
    
    # Tạo DataFrame để dễ xử lý
    df = pd.DataFrame([
        {
            'class': class_name,
            'count': data['count'],
            'formats': data['formats']
        }
        for class_name, data in class_data.items()
    ])
    
    # Sắp xếp theo số lượng ảnh
    df = df.sort_values('count', ascending=False)
    
    # Tạo figure với nhiều subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Bar chart chính
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(range(len(df)), df['count'], color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Số lượng ảnh theo từng class', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Số lượng ảnh', fontsize=12)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['class'], rotation=45, ha='right')
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars, df['count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(df['count'])*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    wedges, texts, autotexts = ax2.pie(df['count'], labels=df['class'], autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Tỷ lệ phân bố các class', fontsize=14, fontweight='bold')
    
    # 3. Top 10 classes
    ax3 = fig.add_subplot(gs[1, :])
    top_10 = df.head(10)
    bars_h = ax3.barh(range(len(top_10)), top_10['count'], color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax3.set_title('Top 10 classes có nhiều ảnh nhất', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Số lượng ảnh', fontsize=12)
    ax3.set_yticks(range(len(top_10)))
    ax3.set_yticklabels(top_10['class'])
    
    # Thêm số liệu trên bars
    for bar, count in zip(bars_h, top_10['count']):
        width = bar.get_width()
        ax3.text(width + max(top_10['count'])*0.01, bar.get_y() + bar.get_height()/2,
                f'{count}', ha='left', va='center', fontweight='bold')
    
    # 4. Distribution histogram
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(df['count'], bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax4.set_title('Phân bố số lượng ảnh', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Số lượng ảnh', fontsize=12)
    ax4.set_ylabel('Số class', fontsize=12)
    
    # 5. Box plot
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.boxplot(df['count'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax5.set_title('Box plot số lượng ảnh', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Số lượng ảnh', fontsize=12)
    
    # 6. Statistics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Tính toán thống kê
    stats_data = [
        ['Tổng số class', f'{len(df)}'],
        ['Tổng số ảnh', f'{total_images:,}'],
        ['Trung bình ảnh/class', f'{df["count"].mean():.1f}'],
        ['Median ảnh/class', f'{df["count"].median():.1f}'],
        ['Ít nhất', f'{df["count"].min()}'],
        ['Nhiều nhất', f'{df["count"].max()}'],
        ['Độ lệch chuẩn', f'{df["count"].std():.1f}'],
        ['Class ít ảnh nhất', f'{df.iloc[-1]["class"]} ({df["count"].min()})'],
        ['Class nhiều ảnh nhất', f'{df.iloc[0]["class"]} ({df["count"].max()})']
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
    plt.savefig(os.path.join(save_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình (mặc định là False)
    show_plots = False  # Có thể thay đổi thành True nếu muốn hiển thị
    if show_plots:
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory
    
    # Tạo đồ thị phân tích định dạng file
    analyze_file_formats(df, save_dir)
    
    # In thống kê ra console
    print_statistics(df, total_images)
    
    return {
        'dataframe': df,
        'total_images': total_images,
        'class_data': class_data
    }

def analyze_file_formats(df, save_dir):
    """
    Phân tích định dạng file ảnh
    """
    # Thu thập thông tin định dạng
    all_formats = defaultdict(int)
    for formats in df['formats']:
        for fmt, count in formats.items():
            all_formats[fmt] += count
    
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
    plt.savefig(os.path.join(save_dir, 'file_formats_analysis.png'), dpi=300, bbox_inches='tight')
    
    # Chỉ hiển thị nếu được cấu hình (mặc định là False)
    show_plots = False  # Có thể thay đổi thành True nếu muốn hiển thị
    if show_plots:
        plt.show()
    else:
        plt.close()  # Đóng figure để tiết kiệm memory

def print_statistics(df, total_images):
    """
    In thống kê chi tiết ra console
    """
    print("\n📊 THỐNG KÊ CHI TIẾT DATASET:")
    print("=" * 60)
    print(f"Tổng số class: {len(df)}")
    print(f"Tổng số ảnh: {total_images:,}")
    print(f"Trung bình ảnh/class: {df['count'].mean():.1f}")
    print(f"Median ảnh/class: {df['count'].median():.1f}")
    print(f"Ít nhất: {df['count'].min()} ảnh")
    print(f"Nhiều nhất: {df['count'].max()} ảnh")
    print(f"Độ lệch chuẩn: {df['count'].std():.1f}")
    print(f"Class ít ảnh nhất: {df.iloc[-1]['class']} ({df['count'].min()} ảnh)")
    print(f"Class nhiều ảnh nhất: {df.iloc[0]['class']} ({df['count'].max()} ảnh)")
    
    # Kiểm tra balance
    balance_ratio = df['count'].min() / df['count'].max()
    if balance_ratio > 0.8:
        balance_status = "Cân bằng tốt"
    elif balance_ratio > 0.5:
        balance_status = "Cân bằng trung bình"
    else:
        balance_status = "Mất cân bằng"
    
    print(f"Tỷ lệ cân bằng: {balance_ratio:.2f} ({balance_status})")
    
    # Phân tích quartiles
    q25, q50, q75 = df['count'].quantile([0.25, 0.5, 0.75])
    print(f"Quartile 25%: {q25:.1f} ảnh")
    print(f"Quartile 50% (median): {q50:.1f} ảnh")
    print(f"Quartile 75%: {q75:.1f} ảnh")
    
    print("=" * 60)

if __name__ == "__main__":
    # Thay đổi đường dẫn này theo dataset của bạn
    dataset_path = r"D:\AI\Dataset"
    
    print("🚀 PHÂN TÍCH DATASET CHI TIẾT")
    print("=" * 50)
    
    result = analyze_dataset_detailed(dataset_path)
    
    if result:
        print(f"\n✅ Phân tích hoàn thành!")
        print(f"📊 Các đồ thị đã được lưu vào thư mục: dataset_analysis/")
    else:
        print("❌ Không thể phân tích dataset!")
