"""
Module để tìm và quản lý các dataset có sẵn trong hệ thống
"""
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

class DatasetFinder:
    """
    Class để tìm và quản lý các dataset có sẵn
    """
    
    def __init__(self, base_paths: List[str] = None):
        """
        Khởi tạo DatasetFinder
        
        Args:
            base_paths: Danh sách các đường dẫn cơ sở để tìm dataset
        """
        if base_paths is None:
            # Các đường dẫn mặc định để tìm dataset
            self.base_paths = [
                r'D:\AI',
                r'D:\Dataset',
                r'D:\Datasets',
                r'C:\Dataset',
                r'C:\Datasets',
                r'./dataset',
                r'./datasets',
                r'./data'
            ]
        else:
            self.base_paths = base_paths
            
        # Các định dạng ảnh được hỗ trợ
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
        
        # Các tên thư mục có thể chứa dataset
        self.dataset_keywords = ['dataset', 'datasets', 'data', 'images', 'train', 'test', 'val', 'validation']
    
    def find_datasets(self, min_images_per_class: int = 1) -> Dict[str, Dict]:
        """
        Tìm tất cả các dataset có sẵn trong hệ thống
        
        Args:
            min_images_per_class: Số ảnh tối thiểu mỗi class để được coi là dataset hợp lệ
            
        Returns:
            Dictionary chứa thông tin các dataset tìm được
        """
        datasets = {}
        
        print("🔍 Đang tìm kiếm datasets trong hệ thống...")
        print("=" * 60)
        
        for base_path in self.base_paths:
            if os.path.exists(base_path):
                print(f"📁 Quét thư mục: {base_path}")
                found_datasets = self._scan_directory(base_path, min_images_per_class)
                datasets.update(found_datasets)
            else:
                print(f"⚠️ Thư mục không tồn tại: {base_path}")
        
        print(f"\n✅ Tìm thấy {len(datasets)} dataset(s) hợp lệ")
        return datasets
    
    def _scan_directory(self, directory: str, min_images_per_class: int) -> Dict[str, Dict]:
        """
        Quét một thư mục để tìm dataset
        
        Args:
            directory: Đường dẫn thư mục cần quét
            min_images_per_class: Số ảnh tối thiểu mỗi class
            
        Returns:
            Dictionary chứa thông tin dataset tìm được
        """
        datasets = {}
        
        try:
            # Quét tất cả thư mục con
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isdir(item_path):
                    # Kiểm tra xem thư mục này có phải là dataset không
                    dataset_info = self._analyze_potential_dataset(item_path, min_images_per_class)
                    
                    if dataset_info:
                        # Tạo tên dataset duy nhất
                        dataset_name = f"{os.path.basename(directory)}_{item}"
                        datasets[dataset_name] = dataset_info
                        print(f"  ✅ Tìm thấy dataset: {dataset_name}")
                        print(f"     📊 Classes: {dataset_info['num_classes']}, Images: {dataset_info['total_images']}")
                    
                    # Quét sâu hơn vào các thư mục con
                    sub_datasets = self._scan_directory(item_path, min_images_per_class)
                    datasets.update(sub_datasets)
                    
        except PermissionError:
            print(f"  ❌ Không có quyền truy cập: {directory}")
        except Exception as e:
            print(f"  ⚠️ Lỗi khi quét {directory}: {str(e)}")
        
        return datasets
    
    def _analyze_potential_dataset(self, path: str, min_images_per_class: int) -> Optional[Dict]:
        """
        Phân tích một thư mục để xem có phải là dataset không
        
        Args:
            path: Đường dẫn thư mục cần phân tích
            min_images_per_class: Số ảnh tối thiểu mỗi class
            
        Returns:
            Thông tin dataset nếu hợp lệ, None nếu không
        """
        try:
            # Kiểm tra xem có phải là cấu trúc dataset không (thư mục con chứa ảnh)
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            
            if not subdirs:
                return None
            
            # Phân tích từng thư mục con (class)
            class_info = {}
            total_images = 0
            valid_classes = 0
            
            for subdir in subdirs:
                subdir_path = os.path.join(path, subdir)
                image_count = self._count_images_in_directory(subdir_path)
                
                if image_count >= min_images_per_class:
                    class_info[subdir] = image_count
                    total_images += image_count
                    valid_classes += 1
            
            # Chỉ coi là dataset nếu có ít nhất 2 class và tổng cộng có ảnh
            if valid_classes >= 2 and total_images > 0:
                return {
                    'path': path,
                    'name': os.path.basename(path),
                    'num_classes': valid_classes,
                    'total_images': total_images,
                    'class_distribution': class_info,
                    'avg_images_per_class': total_images / valid_classes,
                    'min_images_per_class': min(class_info.values()) if class_info else 0,
                    'max_images_per_class': max(class_info.values()) if class_info else 0,
                    'is_balanced': self._is_balanced_dataset(class_info),
                    'suitable_for_fewshot': self._is_suitable_for_fewshot(class_info, min_images_per_class)
                }
            
        except Exception as e:
            pass  # Bỏ qua lỗi và tiếp tục
        
        return None
    
    def _count_images_in_directory(self, directory: str) -> int:
        """
        Đếm số ảnh trong một thư mục
        
        Args:
            directory: Đường dẫn thư mục
            
        Returns:
            Số lượng ảnh
        """
        count = 0
        try:
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)):
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in self.image_extensions):
                        count += 1
        except:
            pass
        return count
    
    def _is_balanced_dataset(self, class_distribution: Dict[str, int]) -> bool:
        """
        Kiểm tra xem dataset có cân bằng không
        
        Args:
            class_distribution: Phân bố số ảnh theo class
            
        Returns:
            True nếu dataset cân bằng
        """
        if not class_distribution:
            return False
        
        counts = list(class_distribution.values())
        avg_count = sum(counts) / len(counts)
        
        # Coi là cân bằng nếu tất cả class có số ảnh trong khoảng 50%-150% của trung bình
        for count in counts:
            if count < avg_count * 0.5 or count > avg_count * 1.5:
                return False
        
        return True
    
    def _is_suitable_for_fewshot(self, class_distribution: Dict[str, int], min_images_per_class: int) -> bool:
        """
        Kiểm tra xem dataset có phù hợp cho few-shot learning không
        
        Args:
            class_distribution: Phân bố số ảnh theo class
            min_images_per_class: Số ảnh tối thiểu mỗi class
            
        Returns:
            True nếu phù hợp cho few-shot learning
        """
        if not class_distribution:
            return False
        
        # Cần ít nhất 5 class và mỗi class có ít nhất min_images_per_class ảnh
        if len(class_distribution) < 5:
            return False
        
        for count in class_distribution.values():
            if count < min_images_per_class:
                return False
        
        return True
    
    def display_datasets(self, datasets: Dict[str, Dict], show_details: bool = True):
        """
        Hiển thị danh sách các dataset tìm được
        
        Args:
            datasets: Dictionary chứa thông tin dataset
            show_details: Có hiển thị chi tiết không
        """
        if not datasets:
            print("❌ Không tìm thấy dataset nào!")
            return
        
        print(f"\n📋 DANH SÁCH DATASET TÌM ĐƯỢC ({len(datasets)} dataset):")
        print("=" * 80)
        
        for i, (name, info) in enumerate(datasets.items(), 1):
            print(f"\n{i}. {name}")
            print(f"   📁 Đường dẫn: {info['path']}")
            print(f"   📊 Số class: {info['num_classes']}")
            print(f"   🖼️ Tổng ảnh: {info['total_images']:,}")
            print(f"   📈 Trung bình ảnh/class: {info['avg_images_per_class']:.1f}")
            print(f"   📊 Min/Max ảnh/class: {info['min_images_per_class']}/{info['max_images_per_class']}")
            print(f"   ⚖️ Cân bằng: {'✅ Có' if info['is_balanced'] else '❌ Không'}")
            print(f"   🎯 Phù hợp Few-Shot: {'✅ Có' if info['suitable_for_fewshot'] else '❌ Không'}")
            
            if show_details and info['class_distribution']:
                print(f"   📋 Chi tiết class:")
                for class_name, count in sorted(info['class_distribution'].items()):
                    print(f"      • {class_name}: {count} ảnh")
    
    def get_dataset_by_path(self, datasets: Dict[str, Dict], path: str) -> Optional[Dict]:
        """
        Lấy thông tin dataset theo đường dẫn
        
        Args:
            datasets: Dictionary chứa thông tin dataset
            path: Đường dẫn dataset cần tìm
            
        Returns:
            Thông tin dataset nếu tìm thấy
        """
        for dataset_info in datasets.values():
            if dataset_info['path'] == path:
                return dataset_info
        return None
    
    def save_datasets_info(self, datasets: Dict[str, Dict], output_file: str = "datasets_info.json"):
        """
        Lưu thông tin dataset vào file JSON
        
        Args:
            datasets: Dictionary chứa thông tin dataset
            output_file: Tên file output
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(datasets, f, indent=2, ensure_ascii=False)
            print(f"💾 Đã lưu thông tin dataset vào: {output_file}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file: {str(e)}")
    
    def load_datasets_info(self, input_file: str = "datasets_info.json") -> Dict[str, Dict]:
        """
        Load thông tin dataset từ file JSON
        
        Args:
            input_file: Tên file input
            
        Returns:
            Dictionary chứa thông tin dataset
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                datasets = json.load(f)
            print(f"📂 Đã load thông tin dataset từ: {input_file}")
            return datasets
        except Exception as e:
            print(f"❌ Lỗi khi load file: {str(e)}")
            return {}

def find_and_display_datasets(min_images_per_class: int = 1, show_details: bool = True) -> Dict[str, Dict]:
    """
    Hàm tiện ích để tìm và hiển thị datasets
    
    Args:
        min_images_per_class: Số ảnh tối thiểu mỗi class
        show_details: Có hiển thị chi tiết không
        
    Returns:
        Dictionary chứa thông tin dataset
    """
    finder = DatasetFinder()
    datasets = finder.find_datasets(min_images_per_class)
    finder.display_datasets(datasets, show_details)
    return datasets

if __name__ == "__main__":
    # Test chức năng
    datasets = find_and_display_datasets(min_images_per_class=5, show_details=True)
    
    if datasets:
        # Lưu thông tin dataset
        finder = DatasetFinder()
        finder.save_datasets_info(datasets)
