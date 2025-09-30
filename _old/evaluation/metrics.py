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
