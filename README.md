### Siamese Transformer Few-Shot (PyTorch)

- Mục tiêu: Chạy few-shot image classification với backbone Transformer (ViT/Swin từ `timm`) theo kiểu Siamese/Prototypical.
- Dữ liệu: cấu trúc thư mục theo class-subfolders.

#### 1) Cài đặt
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_siamese.txt
```

#### 2) Chuẩn bị dữ liệu
Cấu trúc thư mục:
```
DATASET_ROOT/
  class_A/
    img1.jpg
    img2.jpg
  class_B/
  ...
```
Chỉnh `DATASET_ROOT` trong `src/config.py` hoặc truyền qua CLI.

#### 3) Chạy train/eval few-shot
```bash
python -m src --dataset_root D:\\AI\\Dataset \
  --n_way 5 --k_shot 1 --q_query 5 --episodes 200 \
  --backbone vit_base_patch16_224 --image_size 224 --embed_dim 512
```

- Thêm validation (tùy chọn):
```bash
python -m src --dataset_root D:\\AI\\Dataset --val_root D:\\AI\\DatasetVal \
  --n_way 5 --k_shot 1 --q_query 5 --episodes 200 --episodes_val 100 \
  --backbone vit_base_patch16_224 --image_size 224 --embed_dim 512 --distance cosine
```
Tập validation sẽ in và lưu `fewshot_report_val.txt` (accuracy mean/std, classification report, confusion matrix).

Tham số chính:
- `--n_way`, `--k_shot`, `--q_query`: cấu hình episode.
- `--episodes`, `--episodes_val`: số episode train/eval và validation.
- `--backbone`: tên model từ `timm` (vd: `vit_base_patch16_224`, `swin_base_patch4_window7_224`).
- `--distance`: `cosine` | `euclidean`.

#### 4) Ghi chú
- Mặc định dùng embedding từ `timm` và metric không học được (Siamese/Proto). Có thể mở rộng thành Relation Network sau.

#### 5) Câu lệnh 
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_siamese.txt
python -m src --dataset_root D:\AI\Tea_Leaf_Disease --n_way 6 --k_shot 1 --q_query 5 --episodes 50 --backbone vit_base_patch16_224 --image_size 224 --embed_dim 512 --distance cosine