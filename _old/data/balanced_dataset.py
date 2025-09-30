"""
Module cho balanced dataset - đảm bảo tất cả class được đánh giá đều đặn
"""
import os
import random
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import numpy as np

class BalancedFewShotDataset:
    """
    Dataset cho few-shot learning với đánh giá cân bằng tất cả class
    """
    def __init__(self, root_dir, transform_train=None, transform_test=None):
        self.root_dir = root_dir
        self.transform_train = transform_train
        self.transform_test = transform_test
        
        # Tạo dataset với transform cơ bản để lấy thông tin
        self.dataset = ImageFolder(root_dir, transform=transforms.ToTensor())
        self.class_to_indices = self._group_by_class()
        self.all_classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.all_classes)
        
        print(f"📊 Dataset có {self.num_classes} class: {self.all_classes}")
        print(f"📈 Số ảnh mỗi class:")
        for class_id in self.all_classes:
            class_name = self.dataset.classes[class_id]
            num_images = len(self.class_to_indices[class_id])
            print(f"   - {class_name}: {num_images} ảnh")

    def _group_by_class(self):
        """
        Nhóm indices theo class
        """
        class_to_idx = {}
        for idx, (img_path, label) in enumerate(self.dataset.samples):
            class_to_idx.setdefault(label, []).append(idx)
        return class_to_idx

    def load_image_with_transform(self, img_path, use_augmentation=True):
        """
        Load ảnh với transform tương ứng
        """
        img = Image.open(img_path).convert('RGB')
        
        # Sử dụng logic augmentation dựa trên tham số
        if use_augmentation and self.transform_train:
            return self.transform_train(img)
        elif self.transform_test:
            return self.transform_test(img)
        else:
            return transforms.ToTensor()(img)

    def sample_episode_balanced(self, n_way, k_shot, q_query, q_valid=0, use_augmentation=True):
        """
        Sample một episode với n_way classes, đảm bảo tất cả class được sử dụng đều đặn
        """
        # Chọn n_way class từ tất cả class có sẵn
        if n_way >= self.num_classes:
            # Nếu n_way >= số class có sẵn, sử dụng tất cả
            selected_classes = self.all_classes.copy()
            print(f"🎯 Sử dụng tất cả {len(selected_classes)} class: {[self.dataset.classes[c] for c in selected_classes]}")
        else:
            # Nếu n_way < số class, chọn ngẫu nhiên
            selected_classes = random.sample(self.all_classes, n_way)
            print(f"🎯 Chọn ngẫu nhiên {n_way} class: {[self.dataset.classes[c] for c in selected_classes]}")
        
        support_idx, query_idx, valid_idx = [], [], []
        label_map = {}
        
        # Lưu thông tin class được chọn
        self.selected_class_names = [self.dataset.classes[class_id] for class_id in selected_classes]

        for new_label, class_id in enumerate(selected_classes):
            indices = self.class_to_indices[class_id]
            total_needed = k_shot + q_query + q_valid
            
            # Kiểm tra xem class có đủ ảnh không
            if len(indices) < total_needed:
                print(f"⚠️ Class {self.dataset.classes[class_id]} chỉ có {len(indices)} ảnh, cần {total_needed}")
                # Sử dụng tất cả ảnh có sẵn
                sampled = indices
                k_actual = min(k_shot, len(indices))
                q_actual = min(q_query, len(indices) - k_actual)
                v_actual = min(q_valid, len(indices) - k_actual - q_actual)
            else:
                sampled = random.sample(indices, total_needed)
                k_actual, q_actual, v_actual = k_shot, q_query, q_valid
            
            support = sampled[:k_actual]
            query = sampled[k_actual:k_actual + q_actual]
            valid = sampled[k_actual + q_actual:k_actual + q_actual + v_actual] if v_actual > 0 else []

            support_idx += support
            query_idx += query
            valid_idx += valid
            label_map[class_id] = new_label

        # Tạo support, query và validation sets
        support_set = []
        query_set = []
        valid_set = []
        
        for i in support_idx:
            img_path, label = self.dataset.samples[i]
            img_tensor = self.load_image_with_transform(img_path, use_augmentation=use_augmentation)
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

    def get_class_distribution_stats(self):
        """
        Lấy thống kê phân bố class
        """
        stats = {}
        for class_id in self.all_classes:
            class_name = self.dataset.classes[class_id]
            num_images = len(self.class_to_indices[class_id])
            stats[class_name] = num_images
        return stats

    def get_minimum_episodes_for_all_classes(self, n_way, k_shot, q_query, q_valid=0):
        """
        Tính số episode tối thiểu để mỗi class được sử dụng ít nhất một lần
        """
        images_per_episode = n_way * (k_shot + q_query + q_valid)
        total_images_needed = images_per_episode
        
        # Số episode tối thiểu để mỗi class được sử dụng
        min_episodes = max(1, self.num_classes // n_way)
        
        return min_episodes, total_images_needed
