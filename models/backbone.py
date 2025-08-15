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
    def __init__(self, out_dim=512):
        super().__init__()
        self.encoder = create_model('vit_base_patch16_224', pretrained=True)
        self.encoder.head = nn.Identity()
        self.project = nn.Linear(768, out_dim)

    def forward(self, x):
        return self.project(self.encoder(x))

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
