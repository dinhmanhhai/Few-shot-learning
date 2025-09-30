from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TimmEncoder(nn.Module):
	def __init__(self, backbone: str = "vit_base_patch16_224", embed_dim: int = 512):
		super().__init__()
		self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
		in_features = getattr(self.encoder, "num_features", None)
		if in_features is None and hasattr(self.encoder, "get_classifier"):
			in_features = self.encoder.get_classifier().in_features  # type: ignore
		if in_features is None:
			in_features = 768
		self.project = nn.Linear(in_features, embed_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		feat = self.encoder(x)
		return F.normalize(self.project(feat), dim=-1)


def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t()


def pairwise_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	a2 = (a ** 2).sum(dim=1, keepdim=True)
	b2 = (b ** 2).sum(dim=1, keepdim=True).t()
	ab = a @ b.t()
	d2 = a2 + b2 - 2 * ab
	return -d2  # negative distance as similarity


def prototypes(embeddings: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
	return torch.stack([embeddings[labels == c].mean(dim=0) for c in range(n_classes)], dim=0)


def proto_scores(query: torch.Tensor, protos: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
	if metric == "cosine":
		return pairwise_cosine(query, protos)
	else:
		return pairwise_euclidean(query, protos)
