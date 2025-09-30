"""
Module xử lý augmentation theo class cụ thể
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Dict, Tuple, Optional

class ClassSpecificAugmentation:
    """
    Class để xử lý augmentation theo class cụ thể
    """
    def __init__(self, config: Dict):
        self.config = config
        self.class_aug_config = config.get('CLASS_AUGMENTATION', {})
        self.enable_selective = self.class_aug_config.get('enable_selective', False)
        self.augment_classes = self.class_aug_config.get('augment_classes', [])
        self.skip_classes = self.class_aug_config.get('skip_classes', [])
        self.augment_ratio = self.class_aug_config.get('augment_ratio', 1.5)
        self.min_images_per_class = self.class_aug_config.get('min_images_per_class', 5)
        
        # Tạo transforms cho augmentation
        self.augment_transforms = self._create_augment_transforms()
        
        print(f"🎯 Class-Specific Augmentation:")
        print(f"   - Enable selective: {self.enable_selective}")
        if self.enable_selective:
            print(f"   - Augment classes: {self.augment_classes}")
            print(f"   - Skip classes: {self.skip_classes}")
            print(f"   - Augment ratio: {self.augment_ratio}x")
            print(f"   - Min images per class: {self.min_images_per_class}")
    
    def _create_augment_transforms(self) -> transforms.Compose:
        """
        Tạo transforms cho augmentation
        """
        aug_config = self.config.get('AUGMENTATION_CONFIG', {})
        
        return transforms.Compose([
            transforms.RandomCrop(aug_config.get('random_crop_size', 32)),
            transforms.RandomHorizontalFlip(p=aug_config.get('flip_probability', 0.5)),
            transforms.RandomRotation(degrees=aug_config.get('rotation_degrees', 15)),
            transforms.ColorJitter(
                brightness=aug_config.get('color_jitter', {}).get('brightness', 0.2),
                contrast=aug_config.get('color_jitter', {}).get('contrast', 0.2),
                saturation=aug_config.get('color_jitter', {}).get('saturation', 0.2),
                hue=aug_config.get('color_jitter', {}).get('hue', 0.1)
            ),
            transforms.RandomGrayscale(p=aug_config.get('grayscale_probability', 0.1)),
            transforms.Resize((self.config.get('IMAGE_SIZE', 224), self.config.get('IMAGE_SIZE', 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def should_augment_class(self, class_id: int) -> bool:
        """
        Kiểm tra xem class có nên được augment không
        """
        if not self.enable_selective:
            return True  # Augment tất cả class nếu không bật selective
        
        # Debug: In thông tin để kiểm tra
        print(f"🔍 Debug - Class ID {class_id}: augment_classes={self.augment_classes}, skip_classes={self.skip_classes}")
        
        # Kiểm tra trong danh sách augment_classes
        if class_id in self.augment_classes:
            print(f"✅ Class {class_id} sẽ được augment (có trong augment_classes)")
            return True
        
        # Kiểm tra trong danh sách skip_classes
        if class_id in self.skip_classes:
            print(f"⏭️ Class {class_id} sẽ bị skip (có trong skip_classes)")
            return False
        
        # Nếu không có trong cả hai danh sách, mặc định là augment
        print(f"✅ Class {class_id} sẽ được augment (mặc định)")
        return True
    
    def calculate_augmentation_needs(self, class_distribution: Dict[str, int]) -> Dict[str, Dict]:
        """
        Tính toán nhu cầu augmentation cho từng class
        
        Args:
            class_distribution: Dict với key là tên class, value là số ảnh hiện tại
        
        Returns:
            Dict chứa thông tin augmentation cho từng class
        """
        augmentation_plan = {}
        
        # Tạo mapping từ tên class sang index
        class_names = list(class_distribution.keys())
        class_name_to_index = {name: idx for idx, name in enumerate(class_names)}
        
        for class_name, current_count in class_distribution.items():
            # Lấy class_id từ index của class trong danh sách
            class_id = class_name_to_index[class_name]
            
            should_augment = self.should_augment_class(class_id)
            
            if should_augment:
                # Tính số ảnh cần augment
                target_count = max(
                    int(current_count * self.augment_ratio),
                    self.min_images_per_class
                )
                images_to_generate = max(0, target_count - current_count)
                
                augmentation_plan[class_name] = {
                    'should_augment': True,
                    'current_count': current_count,
                    'target_count': target_count,
                    'images_to_generate': images_to_generate,
                    'augment_ratio': self.augment_ratio
                }
            else:
                augmentation_plan[class_name] = {
                    'should_augment': False,
                    'current_count': current_count,
                    'target_count': current_count,
                    'images_to_generate': 0,
                    'augment_ratio': 1.0
                }
        
        return augmentation_plan
    
    def augment_class_images(self, class_images: List[str], class_name: str, 
                           augmentation_plan: Dict) -> List[torch.Tensor]:
        """
        Augment ảnh cho một class cụ thể
        
        Args:
            class_images: List đường dẫn ảnh của class
            class_name: Tên class
            augmentation_plan: Kế hoạch augmentation cho class này
        
        Returns:
            List các ảnh đã được augment (dạng tensor)
        """
        if not augmentation_plan[class_name]['should_augment']:
            return []  # Không augment class này
        
        images_to_generate = augmentation_plan[class_name]['images_to_generate']
        if images_to_generate <= 0:
            return []
        
        augmented_images = []
        
        # Tạo ảnh augment bằng cách chọn ngẫu nhiên từ ảnh gốc
        for _ in range(images_to_generate):
            # Chọn ngẫu nhiên một ảnh gốc
            source_image_path = random.choice(class_images)
            
            try:
                # Load và augment ảnh
                source_image = Image.open(source_image_path).convert('RGB')
                augmented_image = self.augment_transforms(source_image)
                augmented_images.append(augmented_image)
            except Exception as e:
                print(f"⚠️ Lỗi khi augment ảnh {source_image_path}: {e}")
                continue
        
        return augmented_images
    
    def get_augmentation_summary(self, class_distribution: Dict[str, int]) -> Dict:
        """
        Lấy thống kê tổng quan về augmentation
        
        Args:
            class_distribution: Dict với key là tên class, value là số ảnh hiện tại
        
        Returns:
            Dict chứa thống kê augmentation
        """
        augmentation_plan = self.calculate_augmentation_needs(class_distribution)
        
        total_original = sum(class_distribution.values())
        total_augmented = sum(plan['images_to_generate'] for plan in augmentation_plan.values())
        total_final = total_original + total_augmented
        
        classes_to_augment = [name for name, plan in augmentation_plan.items() 
                            if plan['should_augment']]
        classes_to_skip = [name for name, plan in augmentation_plan.items() 
                          if not plan['should_augment']]
        
        return {
            'total_original_images': total_original,
            'total_augmented_images': total_augmented,
            'total_final_images': total_final,
            'augmentation_ratio_overall': total_final / total_original if total_original > 0 else 1.0,
            'classes_to_augment': classes_to_augment,
            'classes_to_skip': classes_to_skip,
            'num_classes_to_augment': len(classes_to_augment),
            'num_classes_to_skip': len(classes_to_skip),
            'augmentation_plan': augmentation_plan
        }
    
    def print_augmentation_plan(self, class_distribution: Dict[str, int]):
        """
        In kế hoạch augmentation
        """
        augmentation_plan = self.calculate_augmentation_needs(class_distribution)
        summary = self.get_augmentation_summary(class_distribution)
        
        print("\n🎯 KẾ HOẠCH CLASS-SPECIFIC AUGMENTATION:")
        print("=" * 60)
        
        for class_name, plan in augmentation_plan.items():
            status = "🔄 AUGMENT" if plan['should_augment'] else "⏭️ SKIP"
            print(f"{status} {class_name}:")
            print(f"   - Ảnh hiện tại: {plan['current_count']}")
            print(f"   - Ảnh mục tiêu: {plan['target_count']}")
            print(f"   - Ảnh sẽ tạo: {plan['images_to_generate']}")
            if plan['should_augment']:
                print(f"   - Tỷ lệ augment: {plan['augment_ratio']}x")
            print()
        
        print("📊 THỐNG KÊ TỔNG QUAN:")
        print(f"   - Tổng ảnh gốc: {summary['total_original_images']:,}")
        print(f"   - Tổng ảnh augment: {summary['total_augmented_images']:,}")
        print(f"   - Tổng ảnh cuối: {summary['total_final_images']:,}")
        print(f"   - Tỷ lệ tổng thể: {summary['augmentation_ratio_overall']:.2f}x")
        print(f"   - Class được augment: {summary['num_classes_to_augment']}")
        print(f"   - Class bỏ qua: {summary['num_classes_to_skip']}")
        print("=" * 60)
