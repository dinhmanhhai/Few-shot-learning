"""
Module cho data transforms và augmentation
"""
from torchvision import transforms

def create_transforms(config):
    """
    Tạo transforms sử dụng cấu hình
    """
    IMAGE_SIZE = config['IMAGE_SIZE']
    AUGMENTATION_CONFIG = config['AUGMENTATION_CONFIG']
    
    # Transform cơ bản cho validation/test
    transform_basic = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform với data augmentation cho training (chỉ khi USE_AUGMENTATION = True)
    if config.get('USE_AUGMENTATION', False):
        transform_augmented = transforms.Compose([
            transforms.Resize((IMAGE_SIZE + AUGMENTATION_CONFIG['random_crop_size'],
                              IMAGE_SIZE + AUGMENTATION_CONFIG['random_crop_size'])),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=AUGMENTATION_CONFIG['flip_probability']),
            transforms.RandomRotation(degrees=AUGMENTATION_CONFIG['rotation_degrees']),
            transforms.ColorJitter(
                brightness=AUGMENTATION_CONFIG['color_jitter']['brightness'],
                contrast=AUGMENTATION_CONFIG['color_jitter']['contrast'],
                saturation=AUGMENTATION_CONFIG['color_jitter']['saturation'],
                hue=AUGMENTATION_CONFIG['color_jitter']['hue']
            ),
            transforms.RandomGrayscale(p=AUGMENTATION_CONFIG['grayscale_probability']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Sử dụng transform cơ bản thay vì augmentation
        transform_augmented = transform_basic

    # Transform cho inference (không có augmentation)
    transform_inference = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform_basic, transform_augmented, transform_inference
