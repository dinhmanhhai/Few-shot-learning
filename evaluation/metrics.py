"""
Module cho evaluation metrics
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

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
