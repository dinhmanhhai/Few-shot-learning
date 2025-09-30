"""
Module cho model backbone và Relation Network

Hỗ trợ đa dạng Transformer architectures:
- Swin Transformer
- ConvNeXt
- Vision Transformer (ViT)
- Data-efficient Image Transformers (DeiT)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from typing import Dict, List, Optional, Tuple

class TransformerBackbone(nn.Module):
    """
    Vision Transformer backbone cho few-shot learning
    
    Hỗ trợ nhiều kiến trúc Transformer từ timm library:
    - Swin Transformer: swin_base_patch4_window7_224, swin_large_patch4_window12_384
    - ConvNeXt: convnext_base, convnext_large
    - ViT: vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224
    - DeiT: deit_base_patch16_224, deit_large_patch16_224
    """
    
    def __init__(self, out_dim: int = 512, model_name: str = 'swin_base_patch4_window7_224'):
        super().__init__()
        self.model_name = model_name
        self.out_dim = out_dim
        
        # Danh sách các model được hỗ trợ
        self.allowed_models = {
            # Swin Transformer
            'swin_base_patch4_window7_224', 'swin_large_patch4_window12_384',
            # ConvNeXt
            'convnext_base', 'convnext_large',
            # Vision Transformer (ViT)
            'vit_base_patch16_224', 'vit_large_patch16_224', 'vit_huge_patch14_224',
            # Data-efficient Image Transformers (DeiT)
            'deit_base_patch16_224', 'deit_large_patch16_224'
        }
        
        # Validate và load model
        self._validate_and_load_model()
        
        # Tạo projection layer
        self._create_projection_layer()
    
    def _validate_and_load_model(self) -> None:
        """Validate model name và load model từ timm"""
        if self.model_name not in self.allowed_models:
            print(f"⚠️ Model '{self.model_name}' không được hỗ trợ. "
                  f"Fallback sang 'swin_base_patch4_window7_224'.")
            self.model_name = 'swin_base_patch4_window7_224'
        
        try:
            self.encoder = create_model(self.model_name, pretrained=True)
            print(f"✅ Đã tải thành công model: {self.model_name}")
        except Exception as e:
            print(f"⚠️ Không thể tải model '{self.model_name}' từ timm ({e}). "
                  f"Dùng fallback 'swin_base_patch4_window7_224'.")
            self.model_name = 'swin_base_patch4_window7_224'
            self.encoder = create_model(self.model_name, pretrained=True)
    
    def _create_projection_layer(self) -> None:
        """Tạo projection layer từ encoder features sang out_dim"""
        # Lấy số features từ encoder
        in_features = self._get_encoder_features()
        
        # Loại bỏ classifier để lấy feature vector
        self._remove_classifier()
        
        # Tạo projection layer
        self.project = nn.Linear(in_features, self.out_dim)
    
    def _get_encoder_features(self) -> int:
        """Lấy số features từ encoder"""
        # Thử lấy từ num_features attribute
        in_features = getattr(self.encoder, 'num_features', None)
        
        if in_features is not None:
            return in_features
        
        # Thử lấy từ head/classifier
        if hasattr(self.encoder, 'head') and hasattr(self.encoder.head, 'in_features'):
            return self.encoder.head.in_features
        elif hasattr(self.encoder, 'classifier') and hasattr(self.encoder.classifier, 'in_features'):
            return self.encoder.classifier.in_features
        
        # Fallback dựa trên tên model
        return self._get_features_by_model_name()
    
    def _get_features_by_model_name(self) -> int:
        """Lấy số features dựa trên tên model"""
        if 'swin' in self.model_name:
            return 1024 if 'large' in self.model_name else 768
        elif 'convnext' in self.model_name:
            return 1024 if 'large' in self.model_name else 768
        elif 'vit' in self.model_name:
            if 'huge' in self.model_name:
                return 1280
            elif 'large' in self.model_name:
                return 1024
            else:  # base
                return 768
        elif 'deit' in self.model_name:
            return 1024 if 'large' in self.model_name else 768
        else:
            return 768
    
    def _remove_classifier(self) -> None:
        """Loại bỏ classifier để lấy feature vector"""
        if hasattr(self.encoder, 'reset_classifier'):
            # timm models thường có API này
            self.encoder.reset_classifier(0)
        else:
            # Fallback: thay thế bằng Identity
            if hasattr(self.encoder, 'head'):
                self.encoder.head = nn.Identity()
            if hasattr(self.encoder, 'classifier'):
                self.encoder.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của Transformer backbone
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            features: Feature tensor (batch_size, out_dim)
        """
        features = self.encoder(x)
        return self.project(features)


class RelationNetwork(nn.Module):
    """
    Relation Network để học cách so sánh giữa query và support samples
    
    Sử dụng pre-trained model nhỏ gọn để học relation scores từ concatenated features
    """
    
    def __init__(self, feature_dim: int = 512, relation_dim: int = 64, pretrained_model: str = "mobilenet_v2"):
        super().__init__()
        self.feature_dim = feature_dim
        self.relation_dim = relation_dim
        self.pretrained_model = pretrained_model
        
        # Pre-trained backbone cho relation network
        self.relation_backbone = self._build_pretrained_backbone()
        
        # Fully connected layers để output relation score
        self.fc = self._build_fc_layers()
    
    def _build_pretrained_backbone(self):
        """Xây dựng pre-trained backbone cho relation network"""
        try:
            # Sử dụng MobileNetV2 - nhỏ gọn và hiệu quả
            if self.pretrained_model == "mobilenet_v2":
                from torchvision.models import mobilenet_v2
                backbone = mobilenet_v2(pretrained=True)
                # Loại bỏ classifier cuối
                backbone.classifier = nn.Identity()
                # Lấy số features từ backbone
                self.backbone_features = 1280  # MobileNetV2 features
                print(f"✅ Sử dụng MobileNetV2 pre-trained cho Relation Network")
                
            elif self.pretrained_model == "efficientnet_b0":
                from torchvision.models import efficientnet_b0
                backbone = efficientnet_b0(pretrained=True)
                backbone.classifier = nn.Identity()
                self.backbone_features = 1280  # EfficientNet-B0 features
                print(f"✅ Sử dụng EfficientNet-B0 pre-trained cho Relation Network")
                
            elif self.pretrained_model == "resnet18":
                from torchvision.models import resnet18
                backbone = resnet18(pretrained=True)
                backbone.fc = nn.Identity()
                self.backbone_features = 512  # ResNet18 features
                print(f"✅ Sử dụng ResNet18 pre-trained cho Relation Network")
                
            else:
                # Fallback về MobileNetV2
                from torchvision.models import mobilenet_v2
                backbone = mobilenet_v2(pretrained=True)
                backbone.classifier = nn.Identity()
                self.backbone_features = 1280
                print(f"✅ Fallback về MobileNetV2 pre-trained cho Relation Network")
                
            return backbone
            
        except Exception as e:
            print(f"⚠️ Không thể tải pre-trained model: {e}")
            print("⚠️ Fallback về CNN tự dựng")
            return self._build_fallback_cnn()
    
    def _build_fallback_cnn(self):
        """Fallback CNN nếu không tải được pre-trained model"""
        self.backbone_features = 64
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def _build_fc_layers(self) -> nn.Sequential:
        """Xây dựng fully connected layers"""
        return nn.Sequential(
            nn.Linear(self.backbone_features, self.relation_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.relation_dim, 1),
            nn.Sigmoid()  # Output score từ 0-1
        )
    
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
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
        query_expanded = query_features.unsqueeze(1).expand(n_query, n_support, -1)
        support_expanded = support_features.unsqueeze(0).expand(n_query, n_support, -1)
        
        # Concatenate features
        combined_features = torch.cat([query_expanded, support_expanded], dim=2)
        
        # Reshape cho pre-trained model (tạo "ảnh" từ features)
        combined_features = self._reshape_for_pretrained_model(combined_features, n_query, n_support)
        
        # Pass qua pre-trained backbone
        backbone_out = self.relation_backbone(combined_features)
        backbone_out = backbone_out.view(n_query * n_support, -1)
        
        # Pass qua FC layers
        relation_scores = self.fc(backbone_out)
        relation_scores = relation_scores.view(n_query, n_support)
        
        return relation_scores
    
    def _reshape_for_pretrained_model(self, combined_features: torch.Tensor, n_query: int, n_support: int) -> torch.Tensor:
        """Reshape features để tạo "ảnh" cho pre-trained model"""
        # Tính kích thước ảnh từ feature dimension
        total_features = self.feature_dim * 2
        
        # Tìm kích thước ảnh phù hợp (bội số của 3)
        img_size = int(total_features ** 0.5)
        while (img_size * img_size) % 3 != 0:
            img_size += 1
        
        # Pad features để phù hợp với kích thước ảnh
        target_size = img_size * img_size
        if total_features < target_size:
            padding_size = target_size - total_features
            combined_features = F.pad(combined_features, (0, padding_size))
        elif total_features > target_size:
            # Cắt bớt nếu quá lớn
            combined_features = combined_features[:, :target_size]
        
        # Reshape thành "ảnh" 3 channels (RGB-like)
        reshaped = combined_features.view(n_query * n_support, img_size * img_size)
        
        # Chia thành 3 channels đều nhau
        features_per_channel = (img_size * img_size) // 3
        
        # Tạo 3 channels từ features
        channel1 = reshaped[:, :features_per_channel]
        channel2 = reshaped[:, features_per_channel:features_per_channel*2]
        channel3 = reshaped[:, features_per_channel*2:features_per_channel*3]
        
        # Reshape mỗi channel thành 2D (H x W)
        h = img_size
        w = features_per_channel // img_size
        if w == 0:
            w = 1
            h = features_per_channel
        
        channel1 = channel1[:, :h*w].view(n_query * n_support, h, w)
        channel2 = channel2[:, :h*w].view(n_query * n_support, h, w)
        channel3 = channel3[:, :h*w].view(n_query * n_support, h, w)
        
        # Pad để có cùng kích thước
        max_h = max(h, img_size)
        max_w = max(w, img_size)
        
        if h < max_h or w < max_w:
            channel1 = F.pad(channel1, (0, max_w - w, 0, max_h - h))
            channel2 = F.pad(channel2, (0, max_w - w, 0, max_h - h))
            channel3 = F.pad(channel3, (0, max_w - w, 0, max_h - h))
        
        # Stack thành 3 channels
        img_features = torch.stack([channel1, channel2, channel3], dim=1)
        
        return img_features

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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Tính khoảng cách Euclidean giữa 2 tensor
    
    Args:
        a: (n, feature_dim)
        b: (m, feature_dim)
        
    Returns:
        distances: (n, m) - pairwise distances
    """
    n = a.size(0)
    m = b.size(0)
    a_expanded = a.unsqueeze(1).expand(n, m, -1)
    b_expanded = b.unsqueeze(0).expand(n, m, -1)
    return torch.pow(a_expanded - b_expanded, 2).sum(2)


def compute_prototypes(support_embeddings: torch.Tensor, 
                      support_labels: torch.Tensor, 
                      n_classes: int) -> torch.Tensor:
    """
    Tính prototypes cho từng class
    
    Args:
        support_embeddings: (n_support, feature_dim)
        support_labels: (n_support,)
        n_classes: Số class
        
    Returns:
        prototypes: (n_classes, feature_dim) - prototype cho mỗi class
    """
    prototypes = []
    for c in range(n_classes):
        class_embeddings = support_embeddings[support_labels == c]
        if class_embeddings.size(0) > 0:
            prototype = class_embeddings.mean(0)
        else:
            # Nếu không có sample nào cho class này, dùng zero vector
            prototype = torch.zeros(support_embeddings.size(1), device=support_embeddings.device)
        prototypes.append(prototype)
    return torch.stack(prototypes)


class FlexibleDistanceModel(nn.Module):
    """
    Model linh hoạt có thể chọn giữa Euclidean Distance và Relation Network
    
    Hỗ trợ nhiều kiến trúc Transformer và phương pháp đo khoảng cách
    """
    
    def __init__(self, 
                 embed_dim: int = 512, 
                 relation_dim: int = 64, 
                 distance_method: str = "relation_network", 
                 transformer_model: str = "swin_base_patch4_window7_224",
                 relation_pretrained_model: str = "mobilenet_v2"):
        super().__init__()
        self.embed_dim = embed_dim
        self.relation_dim = relation_dim
        self.distance_method = distance_method
        
        # Khởi tạo backbone và relation network
        self.backbone = TransformerBackbone(embed_dim, transformer_model)
        self.relation_net = RelationNetwork(embed_dim, relation_dim, relation_pretrained_model)
        
        # Cập nhật tên model thực tế (sau fallback nếu có)
        self.transformer_model = getattr(self.backbone, 'model_name', transformer_model)
        
        # Hiển thị thông tin model
        self._print_model_info()
    
    def _print_model_info(self) -> None:
        """In thông tin model"""
        transformer_names = {
            # Swin Transformer
            'swin_base_patch4_window7_224': 'Swin-Base',
            'swin_large_patch4_window12_384': 'Swin-Large',
            # ConvNeXt
            'convnext_base': 'ConvNeXt-Base',
            'convnext_large': 'ConvNeXt-Large',
            # Vision Transformer (ViT)
            'vit_base_patch16_224': 'ViT-Base',
            'vit_large_patch16_224': 'ViT-Large',
            'vit_huge_patch14_224': 'ViT-Huge',
            # Data-efficient Image Transformers (DeiT)
            'deit_base_patch16_224': 'DeiT-Base',
            'deit_large_patch16_224': 'DeiT-Large'
        }
        
        display_name = transformer_names.get(self.transformer_model, self.transformer_model)
        
        print(f"🎯 Khởi tạo model với transformer: {display_name}")
        print(f"   - Architecture: {self.transformer_model}")
        print(f"   - Phương pháp: {self.distance_method}")
        
        if self.distance_method == "relation_network":
            print(f"   - Sử dụng Relation Network (có thể học được)")
            print(f"   - Relation dimension: {self.relation_dim}")
            print(f"   - Relation backbone: {self.relation_net.pretrained_model}")
        else:
            print(f"   - Sử dụng Euclidean Distance (cố định)")
    
    def forward(self, support_imgs: torch.Tensor, query_imgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass với phương pháp được chọn
        
        Args:
            support_imgs: (n_support, 3, H, W)
            query_imgs: (n_query, 3, H, W)
            
        Returns:
            scores: (n_query, n_support) - similarity scores
        """
        if self.distance_method == "relation_network":
            return self._forward_relation_network(support_imgs, query_imgs)
        else:
            return self._forward_euclidean(support_imgs, query_imgs)
    
    def _forward_relation_network(self, support_imgs: torch.Tensor, query_imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass cho Relation Network"""
        support_features = self.backbone(support_imgs)
        query_features = self.backbone(query_imgs)
        return self.relation_net(query_features, support_features)
    
    def _forward_euclidean(self, support_imgs: torch.Tensor, query_imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass cho Euclidean Distance"""
        support_features = self.backbone(support_imgs)
        query_features = self.backbone(query_imgs)
        
        # Compute Euclidean distances và chuyển thành similarity scores
        distances = euclidean_distance(query_features, support_features)
        return 1.0 / (1.0 + distances)
    
    def compute_class_scores(self, 
                            support_imgs: torch.Tensor, 
                            support_labels: torch.Tensor, 
                            query_imgs: torch.Tensor, 
                            n_classes: int) -> torch.Tensor:
        """
        Tính scores cho từng class với phương pháp được chọn
        
        Args:
            support_imgs: (n_support, 3, H, W)
            support_labels: (n_support,)
            query_imgs: (n_query, 3, H, W)
            n_classes: Số class
            
        Returns:
            class_scores: (n_query, n_classes) - scores cho từng class
        """
        if self.distance_method == "relation_network":
            return self._compute_class_scores_relation(support_imgs, support_labels, query_imgs, n_classes)
        else:
            return self._compute_class_scores_euclidean(support_imgs, support_labels, query_imgs, n_classes)
    
    def _compute_class_scores_relation(self, 
                                     support_imgs: torch.Tensor, 
                                     support_labels: torch.Tensor, 
                                     query_imgs: torch.Tensor, 
                                     n_classes: int) -> torch.Tensor:
        """Tính class scores sử dụng Relation Network"""
        relation_scores = self._forward_relation_network(support_imgs, query_imgs)
        return self._aggregate_scores_by_class(relation_scores, support_labels, n_classes, query_imgs.device)
    
    def _compute_class_scores_euclidean(self, 
                                      support_imgs: torch.Tensor, 
                                      support_labels: torch.Tensor, 
                                      query_imgs: torch.Tensor, 
                                      n_classes: int) -> torch.Tensor:
        """Tính class scores sử dụng Euclidean Distance"""
        support_features = self.backbone(support_imgs)
        query_features = self.backbone(query_imgs)
        
        # Tính prototypes cho từng class
        prototypes = compute_prototypes(support_features, support_labels, n_classes)
        
        # Tính Euclidean distances với prototypes và chuyển thành similarity
        distances = euclidean_distance(query_features, prototypes)
        return 1.0 / (1.0 + distances)
    
    def _aggregate_scores_by_class(self, 
                                 scores: torch.Tensor, 
                                 support_labels: torch.Tensor, 
                                 n_classes: int, 
                                 device: torch.device) -> torch.Tensor:
        """Aggregate scores theo class"""
        class_scores = torch.zeros(scores.size(0), n_classes, device=device)
        
        for c in range(n_classes):
            class_mask = (support_labels == c)
            if class_mask.sum() > 0:
                class_relations = scores[:, class_mask]
                class_scores[:, c] = class_relations.mean(dim=1)
        
        return class_scores
