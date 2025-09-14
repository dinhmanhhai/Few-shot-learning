"""
Module cho model backbone và Relation Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class TransformerBackbone(nn.Module):
    """
    Vision Transformer backbone cho few-shot learning
    """
    def __init__(self, out_dim=512, model_name='swin_base_patch4_window7_224'):
        super().__init__()
        self.model_name = model_name
        # Chỉ cho phép Swin và ConvNeXt
        allowed_models = {
            'swin_base_patch4_window7_224', 'swin_large_patch4_window12_384',
            'convnext_base', 'convnext_large'
        }
        if self.model_name not in allowed_models:
            print(f"⚠️ Model '{self.model_name}' không được hỗ trợ (chỉ Swin/ConvNeXt). Fallback sang 'swin_base_patch4_window7_224'.")
            self.model_name = 'swin_base_patch4_window7_224'
        try:
            self.encoder = create_model(self.model_name, pretrained=True)
        except Exception as e:
            print(f"⚠️ Không thể tải model '{self.model_name}' từ timm ({e}). Dùng fallback 'swin_base_patch4_window7_224'.")
            self.model_name = 'swin_base_patch4_window7_224'
            self.encoder = create_model(self.model_name, pretrained=True)

        # Lấy số features TRƯỚC khi thay head/classifier bằng Identity
        in_features = getattr(self.encoder, 'num_features', None)
        if in_features is None:
            if hasattr(self.encoder, 'head') and hasattr(self.encoder.head, 'in_features'):
                in_features = self.encoder.head.in_features
            elif hasattr(self.encoder, 'classifier') and hasattr(self.encoder.classifier, 'in_features'):
                in_features = self.encoder.classifier.in_features
            else:
                # Fallback dựa trên tên model
                if 'swin' in model_name:
                    in_features = 1024 if 'large' in model_name else 768
                elif 'convnext' in model_name:
                    in_features = 1024 if 'large' in model_name else 768
                else:
                    in_features = 768

        # Loại bỏ classifier để lấy feature vector
        if hasattr(self.encoder, 'reset_classifier'):
            # timm models thường có API này
            self.encoder.reset_classifier(0)
        else:
            if hasattr(self.encoder, 'head'):
                self.encoder.head = nn.Identity()
            if hasattr(self.encoder, 'classifier'):
                self.encoder.classifier = nn.Identity()

        self.project = nn.Linear(in_features, out_dim)

    def forward(self, x):
        features = self.encoder(x)
        return self.project(features)

class RelationNetwork(nn.Module):
    """
    Relation Network để học cách so sánh giữa query và support samples
    """
    def __init__(self, feature_dim=512, relation_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.relation_dim = relation_dim
        
        # Mạng CNN để học relation score
        self.relation_net = nn.Sequential(
            # Input: concatenated features (feature_dim * 2)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fully connected layers để output relation score
        self.fc = nn.Sequential(
            nn.Linear(64, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()  # Output score từ 0-1
        )
    
    def forward(self, query_features, support_features):
        """
        Tính relation score giữa query và support features
        
        Args:
            query_features: (n_query, feature_dim)
            support_features: (n_support, feature_dim)
        
        Returns:
            relation_scores: (n_query, n_support) - scores từ 0-1
        """
        n_query = query_features.size(0)
        n_support = support_features.size(0)
        
        # Reshape features để concatenate
        query_features = query_features.unsqueeze(1).expand(n_query, n_support, -1)  # (n_query, n_support, feature_dim)
        support_features = support_features.unsqueeze(0).expand(n_query, n_support, -1)  # (n_query, n_support, feature_dim)
        
        # Concatenate features
        combined_features = torch.cat([query_features, support_features], dim=2)  # (n_query, n_support, feature_dim*2)
        
        # Reshape cho CNN: (n_query * n_support, 1, sqrt(feature_dim*2), sqrt(feature_dim*2))
        feature_size = int((self.feature_dim * 2) ** 0.5)
        if feature_size * feature_size != self.feature_dim * 2:
            # Nếu không phải số chính phương, pad hoặc resize
            target_size = int((self.feature_dim * 2) ** 0.5) + 1
            target_size = target_size * target_size
            if target_size > self.feature_dim * 2:
                # Pad với zeros
                padding_size = target_size - self.feature_dim * 2
                combined_features = F.pad(combined_features, (0, padding_size))
            feature_size = int(target_size ** 0.5)
        
        combined_features = combined_features.view(n_query * n_support, 1, feature_size, feature_size)
        
        # Pass qua CNN
        conv_out = self.relation_net(combined_features)  # (n_query * n_support, 64, 1, 1)
        conv_out = conv_out.view(n_query * n_support, -1)  # (n_query * n_support, 64)
        
        # Pass qua FC layers
        relation_scores = self.fc(conv_out)  # (n_query * n_support, 1)
        relation_scores = relation_scores.view(n_query, n_support)  # (n_query, n_support)
        
        return relation_scores

class RelationNetworkModel(nn.Module):
    """
    Model hoàn chỉnh kết hợp Transformer backbone và Relation Network
    """
    def __init__(self, embed_dim=512, relation_dim=64):
        super().__init__()
        self.backbone = TransformerBackbone(out_dim=embed_dim)
        self.relation_net = RelationNetwork(feature_dim=embed_dim, relation_dim=relation_dim)
        self.embed_dim = embed_dim
    
    def forward(self, support_imgs, query_imgs):
        """
        Forward pass cho Relation Network
        
        Args:
            support_imgs: (n_support, 3, H, W)
            query_imgs: (n_query, 3, H, W)
        
        Returns:
            relation_scores: (n_query, n_support) - scores từ 0-1
        """
        # Extract features
        support_features = self.backbone(support_imgs)  # (n_support, embed_dim)
        query_features = self.backbone(query_imgs)      # (n_query, embed_dim)
        
        # Compute relation scores
        relation_scores = self.relation_net(query_features, support_features)
        
        return relation_scores
    
    def compute_class_scores(self, support_imgs, support_labels, query_imgs, n_classes):
        """
        Tính scores cho từng class dựa trên relation với support samples
        
        Args:
            support_imgs: (n_support, 3, H, W)
            support_labels: (n_support,)
            query_imgs: (n_query, 3, H, W)
            n_classes: số class
        
        Returns:
            class_scores: (n_query, n_classes) - scores cho từng class
        """
        relation_scores = self.forward(support_imgs, query_imgs)  # (n_query, n_support)
        
        # Tính average relation score cho từng class
        class_scores = torch.zeros(query_imgs.size(0), n_classes, device=query_imgs.device)
        
        for c in range(n_classes):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_relations = relation_scores[:, class_mask]  # (n_query, n_class_samples)
                class_scores[:, c] = class_relations.mean(dim=1)  # Average over class samples
        
        return class_scores

def compute_relation_scores(query_features, support_features, relation_net):
    """
    Tính relation scores giữa query và support features
    """
    return relation_net(query_features, support_features)

def compute_class_scores_relation(query_features, support_features, support_labels, n_classes, relation_net):
    """
    Tính class scores sử dụng Relation Network
    """
    relation_scores = relation_net(query_features, support_features)
    
    # Tính average relation score cho từng class
    class_scores = torch.zeros(query_features.size(0), n_classes, device=query_features.device)
    
    for c in range(n_classes):
        class_mask = (support_labels == c)
        if class_mask.sum() > 0:
            class_relations = relation_scores[:, class_mask]
            class_scores[:, c] = class_relations.mean(dim=1)
    
    return class_scores

# Giữ lại các hàm cũ để backward compatibility (có thể xóa sau)
def euclidean_distance(a, b):
    """
    Tính khoảng cách Euclidean giữa 2 tensor (legacy)
    """
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return torch.pow(a - b, 2).sum(2)

def compute_prototypes(support_embeddings, support_labels, n_classes):
    """
    Tính prototypes cho từng class (legacy)
    """
    prototypes = []
    for c in range(n_classes):
        class_embeddings = support_embeddings[support_labels == c]
        prototype = class_embeddings.mean(0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

class FlexibleDistanceModel(nn.Module):
    """
    Model linh hoạt có thể chọn giữa Euclidean Distance và Relation Network
    """
    def __init__(self, embed_dim=512, relation_dim=64, distance_method="relation_network", transformer_model="swin_base_patch4_window7_224"):
        super().__init__()
        self.backbone = TransformerBackbone(embed_dim, transformer_model)
        self.relation_net = RelationNetwork(embed_dim, relation_dim)
        self.embed_dim = embed_dim
        self.relation_dim = relation_dim
        self.distance_method = distance_method
        # Cập nhật tên model thực tế (sau fallback nếu có)
        self.transformer_model = getattr(self.backbone, 'model_name', transformer_model)
        
        # Tên hiển thị cho transformer
        transformer_names = {
            'swin_base_patch4_window7_224': 'Swin-Base',
            'swin_large_patch4_window12_384': 'Swin-Large',
            'convnext_base': 'ConvNeXt-Base',
            'convnext_large': 'ConvNeXt-Large'
        }
        
        display_name = transformer_names.get(self.transformer_model, self.transformer_model)
        
        print(f"🎯 Khởi tạo model với transformer: {display_name}")
        print(f"   - Architecture: {self.transformer_model}")
        print(f"   - Phương pháp: {distance_method}")
        if distance_method == "relation_network":
            print(f"   - Sử dụng Relation Network (có thể học được)")
            print(f"   - Relation dimension: {relation_dim}")
        else:
            print(f"   - Sử dụng Euclidean Distance (cố định)")
    
    def forward(self, support_imgs, query_imgs):
        """
        Forward pass với phương pháp được chọn
        """
        if self.distance_method == "relation_network":
            return self._forward_relation_network(support_imgs, query_imgs)
        else:
            return self._forward_euclidean(support_imgs, query_imgs)
    
    def _forward_relation_network(self, support_imgs, query_imgs):
        """
        Forward pass cho Relation Network
        """
        # Extract features
        support_features = self.backbone(support_imgs)  # (n_support, embed_dim)
        query_features = self.backbone(query_imgs)      # (n_query, embed_dim)
        
        # Compute relation scores
        relation_scores = self.relation_net(query_features, support_features)
        
        return relation_scores
    
    def _forward_euclidean(self, support_imgs, query_imgs):
        """
        Forward pass cho Euclidean Distance
        """
        # Extract features
        support_features = self.backbone(support_imgs)  # (n_support, embed_dim)
        query_features = self.backbone(query_imgs)      # (n_query, embed_dim)
        
        # Compute Euclidean distances (chuyển thành similarity scores)
        distances = euclidean_distance(query_features, support_features)
        # Chuyển distance thành similarity (càng gần càng cao)
        similarity_scores = 1.0 / (1.0 + distances)
        
        return similarity_scores
    
    def compute_class_scores(self, support_imgs, support_labels, query_imgs, n_classes):
        """
        Tính scores cho từng class với phương pháp được chọn
        """
        if self.distance_method == "relation_network":
            return self._compute_class_scores_relation(support_imgs, support_labels, query_imgs, n_classes)
        else:
            return self._compute_class_scores_euclidean(support_imgs, support_labels, query_imgs, n_classes)
    
    def _compute_class_scores_relation(self, support_imgs, support_labels, query_imgs, n_classes):
        """
        Tính class scores sử dụng Relation Network
        """
        relation_scores = self._forward_relation_network(support_imgs, query_imgs)
        
        # Tính average relation score cho từng class
        class_scores = torch.zeros(query_imgs.size(0), n_classes, device=query_imgs.device)
        
        for c in range(n_classes):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_relations = relation_scores[:, class_mask]
                class_scores[:, c] = class_relations.mean(dim=1)
        
        return class_scores
    
    def _compute_class_scores_euclidean(self, support_imgs, support_labels, query_imgs, n_classes):
        """
        Tính class scores sử dụng Euclidean Distance
        """
        # Extract features
        support_features = self.backbone(support_imgs)
        query_features = self.backbone(query_imgs)
        
        # Tính prototypes cho từng class
        prototypes = compute_prototypes(support_features, support_labels, n_classes)
        
        # Tính Euclidean distances với prototypes
        distances = euclidean_distance(query_features, prototypes)
        
        # Chuyển distance thành similarity scores
        similarity_scores = 1.0 / (1.0 + distances)
        
        return similarity_scores
