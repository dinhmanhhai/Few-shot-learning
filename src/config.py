import argparse
from dataclasses import dataclass


@dataclass
class Config:
	# Đường dẫn dữ liệu
	dataset_root: str = r"D:\AI\Dataset"
	val_root: str | None = None
	# Episode setup
	n_way: int = 5
	k_shot: int = 1
	q_query: int = 5
	episodes: int = 200
	episodes_val: int = 0
	# Ảnh và backbone
	image_size: int = 224
	backbone: str = "vit_base_patch16_224"
	embed_dim: int = 512
	# Metric
	distance: str = "cosine"  # cosine | euclidean
	# Huấn luyện
	device: str = "cuda"
	seed: int = 42


def build_config_from_cli() -> Config:
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_root", type=str)
	parser.add_argument("--val_root", type=str)
	parser.add_argument("--n_way", type=int)
	parser.add_argument("--k_shot", type=int)
	parser.add_argument("--q_query", type=int)
	parser.add_argument("--episodes", type=int)
	parser.add_argument("--episodes_val", type=int)
	parser.add_argument("--image_size", type=int)
	parser.add_argument("--backbone", type=str)
	parser.add_argument("--embed_dim", type=int)
	parser.add_argument("--distance", type=str, choices=["cosine", "euclidean"])
	parser.add_argument("--device", type=str)
	parser.add_argument("--seed", type=int)
	args = parser.parse_args()

	cfg = Config()
	for k, v in vars(args).items():
		if v is not None:
			setattr(cfg, k, v)
	return cfg
