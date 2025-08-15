# Relation Network cho Few-Shot Learning

## 🧠 **Tổng quan**

Dự án này đã được cập nhật để sử dụng **Relation Network** thay vì phép đo Euclidean distance truyền thống. Relation Network là một kiến trúc có thể học được để so sánh giữa query samples và support samples.

## 🔄 **Thay đổi từ Euclidean Distance sang Relation Network**

### **Trước đây (Euclidean Distance):**
```python
# Tính prototypes cho từng class
prototypes = compute_prototypes(support_embeddings, support_labels, n_classes)

# Tính khoảng cách Euclidean
distances = euclidean_distance(query_embeddings, prototypes)

# Phân loại dựa trên khoảng cách ngắn nhất
predictions = torch.argmin(distances, dim=1)
```

### **Bây giờ (Relation Network):**
```python
# Tính relation scores giữa query và support samples
relation_scores = model.compute_class_scores(support_imgs, support_labels, query_imgs, n_classes)

# Phân loại dựa trên relation scores cao nhất
predictions = torch.argmax(relation_scores, dim=1)
```

## 🏗️ **Kiến trúc Relation Network**

### **1. Transformer Backbone**
- **Model**: Vision Transformer (vit_base_patch16_224)
- **Chức năng**: Trích xuất đặc trưng từ ảnh
- **Output**: Embedding vectors (512 chiều)

### **2. Relation Network (CNN)**
- **Input**: Concatenated features từ query và support samples
- **Kiến trúc CNN**:
  ```
  Conv2d(1, 64) → BatchNorm → ReLU → MaxPool2d(2)
  Conv2d(64, 64) → BatchNorm → ReLU → MaxPool2d(2)
  Conv2d(64, 64) → BatchNorm → ReLU → MaxPool2d(2)
  Conv2d(64, 64) → BatchNorm → ReLU → AdaptiveAvgPool2d(1)
  ```
- **Fully Connected Layers**:
  ```
  Linear(64, relation_dim) → ReLU → Linear(relation_dim, 1) → Sigmoid
  ```
- **Output**: Relation scores từ 0-1

## 📊 **Quy trình hoạt động**

### **1. Feature Extraction**
```python
# Trích xuất đặc trưng từ support và query images
support_features = backbone(support_imgs)  # (n_support, embed_dim)
query_features = backbone(query_imgs)      # (n_query, embed_dim)
```

### **2. Feature Concatenation**
```python
# Concatenate features của từng cặp query-support
combined_features = torch.cat([query_features, support_features], dim=2)
# Shape: (n_query, n_support, embed_dim*2)
```

### **3. Relation Score Computation**
```python
# Reshape cho CNN
combined_features = combined_features.view(n_query * n_support, 1, feature_size, feature_size)

# Pass qua CNN + FC layers
relation_scores = relation_net(combined_features)
# Shape: (n_query, n_support) với scores từ 0-1
```

### **4. Class Score Aggregation**
```python
# Tính average relation score cho từng class
for c in range(n_classes):
    class_mask = (support_labels == c)
    class_relations = relation_scores[:, class_mask]
    class_scores[:, c] = class_relations.mean(dim=1)
```

### **5. Classification**
```python
# Phân loại dựa trên class scores cao nhất
predictions = torch.argmax(class_scores, dim=1)
```

## ⚙️ **Cấu hình mới**

### **config.py**
```python
# Tham số mô hình
EMBED_DIM = 512        # Kích thước embedding từ Transformer
RELATION_DIM = 64      # Kích thước hidden layer của Relation Network
IMAGE_SIZE = 224       # Kích thước ảnh input

# Tham số Few-Shot Learning
N_WAY = 10             # Số class trong mỗi episode
K_SHOT = 3             # Số ảnh support mỗi class
Q_QUERY = 20           # Số ảnh query mỗi class
Q_VALID = 10           # Số ảnh validation mỗi class
```

## 🎯 **Ưu điểm của Relation Network**

### **1. Khả năng học được**
- **Euclidean Distance**: Cố định, không thể học
- **Relation Network**: Có thể học cách so sánh tối ưu

### **2. Linh hoạt hơn**
- Có thể học các mối quan hệ phức tạp
- Không bị giới hạn bởi metric distance

### **3. Hiệu suất tốt hơn**
- Thường cho kết quả tốt hơn trên nhiều dataset
- Có thể xử lý các trường hợp đặc biệt

### **4. Interpretability**
- Relation scores có thể giải thích được
- Có thể phân tích mối quan hệ giữa samples

## 🔧 **Cách sử dụng**

### **1. Chạy với Relation Network**
```bash
python main_new.py
```

### **2. Kết quả output**
- **Model**: RelationNetworkModel thay vì TransformerBackbone
- **Method**: Relation Network thay vì Euclidean Distance
- **Scores**: Relation scores (0-1) thay vì distances

### **3. Thông báo console**
```
✅ Relation Network Model đã được tải lên cuda
📊 Cấu hình: 10-way, 3-shot, 20-query
🧠 Kiến trúc: Vision Transformer + Relation Network (CNN)
🔄 Đang chạy 100 episodes với Relation Network...
```

## 📈 **So sánh hiệu suất**

### **Expected Improvements:**
- **Accuracy**: Tăng 2-5% so với Euclidean distance
- **Robustness**: Ít bị ảnh hưởng bởi noise
- **Generalization**: Tốt hơn trên unseen classes

### **Trade-offs:**
- **Computational Cost**: Cao hơn do CNN layers
- **Memory Usage**: Cần nhiều memory hơn
- **Training Time**: Lâu hơn do tham số nhiều hơn

## 🚀 **Tương lai**

### **Có thể cải tiến:**
1. **Attention Mechanism**: Thêm attention vào Relation Network
2. **Multi-scale Features**: Sử dụng features từ nhiều layers
3. **Meta-learning**: Train Relation Network với meta-learning
4. **Ensemble**: Kết hợp nhiều Relation Networks

### **Experiments:**
1. **Ablation Study**: So sánh các components
2. **Hyperparameter Tuning**: Tối ưu relation_dim, CNN architecture
3. **Cross-dataset**: Test trên nhiều dataset khác nhau

## 📚 **References**

1. **Relation Networks for Object Detection** - Hu et al., 2018
2. **Learning to Compare: Relation Network for Few-Shot Learning** - Sung et al., 2018
3. **Prototypical Networks for Few-shot Learning** - Snell et al., 2017

---

**Lưu ý**: Relation Network đã được implement và test. Có thể cần fine-tuning hyperparameters để đạt hiệu suất tối ưu trên dataset cụ thể của bạn.
