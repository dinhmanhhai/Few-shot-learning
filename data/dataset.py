"""
Module cho dataset và episode sampling
"""
import os
import random
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

class FewShotDataset:
    """
    Dataset cho few-shot learning với episode sampling
    """
    def __init__(self, root_dir, transform_train=None, transform_test=None):
        self.root_dir = root_dir
        self.transform_train = transform_train
        self.transform_test = transform_test
        
        # Tạo dataset với transform cơ bản để lấy thông tin
        self.dataset = ImageFolder(root_dir, transform=transforms.ToTensor())
        self.class_to_indices = self._group_by_class()

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

    def sample_episode(self, n_way, k_shot, q_query, q_valid=0, use_augmentation=True):
        """
        Sample một episode với n_way classes, k_shot support, q_query query, q_valid validation
        """
        selected_classes = random.sample(list(self.class_to_indices.keys()), n_way)
        support_idx, query_idx, valid_idx = [], [], []
        label_map = {}
        
        # Lưu thông tin class được chọn
        self.selected_class_names = [self.dataset.classes[class_id] for class_id in selected_classes]

        for new_label, class_id in enumerate(selected_classes):
            indices = self.class_to_indices[class_id]
            total_needed = k_shot + q_query + q_valid
            sampled = random.sample(indices, total_needed)
            support = sampled[:k_shot]
            query = sampled[k_shot:k_shot + q_query]
            valid = sampled[k_shot + q_query:] if q_valid > 0 else []

            support_idx += support
            query_idx += query
            valid_idx += valid
            label_map[class_id] = new_label

        # Tạo support, query và validation sets với augmentation
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
