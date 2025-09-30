import os
import random
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ClassFolderDataset:
	"""Index theo class -> list file paths."""
	def __init__(self, root: str):
		self.root = root
		self.class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
		self.class_to_files = {}
		for c in self.class_names:
			cdir = os.path.join(root, c)
			files = []
			for fname in os.listdir(cdir):
				p = os.path.join(cdir, fname)
				if os.path.isfile(p) and fname.lower().split(".")[-1] in {"jpg", "jpeg", "png", "bmp"}:
					files.append(p)
			self.class_to_files[c] = files

	def sample_episode(self, n_way: int, k_shot: int, q_query: int) -> Tuple[List[str], List[int], List[str], List[int]]:
		classes = random.sample(self.class_names, n_way)
		support_paths, support_labels = [], []
		query_paths, query_labels = [], []
		for idx, c in enumerate(classes):
			files = random.sample(self.class_to_files[c], k_shot + q_query)
			support_paths += files[:k_shot]
			support_labels += [idx] * k_shot
			query_paths += files[k_shot:]
			query_labels += [idx] * q_query
		return support_paths, support_labels, query_paths, query_labels


def build_transforms(image_size: int, train: bool = True):
	ops = [
		T.Resize((image_size, image_size)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]
	return T.Compose(ops)


def load_images(paths: List[str], transform) -> torch.Tensor:
	imgs = []
	for p in paths:
		img = Image.open(p).convert("RGB")
		imgs.append(transform(img))
	return torch.stack(imgs, dim=0)
