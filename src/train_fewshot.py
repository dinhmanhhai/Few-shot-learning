import os
import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .config import build_config_from_cli
from .data import ClassFolderDataset, build_transforms, load_images
from .model import TimmEncoder, prototypes, proto_scores


def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def run_episode(encoder: TimmEncoder,
			   device: torch.device,
			   ds: ClassFolderDataset,
			   n_way: int, k_shot: int, q_query: int,
			   image_size: int, metric: str) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
	# sample
	s_paths, s_labels, q_paths, q_labels = ds.sample_episode(n_way, k_shot, q_query)
	transform = build_transforms(image_size)
	s_imgs = load_images(s_paths, transform).to(device)
	q_imgs = load_images(q_paths, transform).to(device)
	s_labels_t = torch.tensor(s_labels, device=device)
	q_labels_t = torch.tensor(q_labels, device=device)

	# encode
	with torch.no_grad():
		s_emb = encoder(s_imgs)
		q_emb = encoder(q_imgs)
		protos = prototypes(s_emb, s_labels_t, n_way)
		scores = proto_scores(q_emb, protos, metric=metric)
		pred = scores.argmax(dim=1)
		acc = (pred == q_labels_t).float().mean().item()

	# simple NLL-style loss with softmax over classes
	logits = scores
	loss = F.cross_entropy(logits, q_labels_t)
	return loss.item(), acc, q_labels_t.detach().cpu(), pred.detach().cpu()


def evaluate_split(encoder: TimmEncoder, device: torch.device, ds: ClassFolderDataset,
				  episodes: int, n_way: int, k_shot: int, q_query: int,
				  image_size: int, metric: str, title: str):
	losses, accs = [], []
	all_true, all_pred = [], []
	for _ in tqdm(range(episodes), desc=title):
		loss, acc, y_true, y_pred = run_episode(
			encoder, device, ds, n_way, k_shot, q_query, image_size, metric
		)
		losses.append(loss)
		accs.append(acc)
		all_true.append(y_true.numpy())
		all_pred.append(y_pred.numpy())

	all_true_np = np.concatenate(all_true, axis=0)
	all_pred_np = np.concatenate(all_pred, axis=0)
	mean_acc = float(np.mean(accs))
	std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
	print(f"{title} -> Avg loss: {sum(losses)/len(losses):.4f} | Avg acc: {mean_acc:.4f} (+/- {std_acc:.4f})")
	print("\nClassification report (per class):")
	print(classification_report(all_true_np, all_pred_np, digits=4, output_dict=False))
	cm = confusion_matrix(all_true_np, all_pred_np)
	print("Confusion matrix:\n", cm)
	return {
		"losses": losses,
		"accs": accs,
		"y_true": all_true_np,
		"y_pred": all_pred_np,
		"cm": cm,
		"mean_acc": mean_acc,
		"std_acc": std_acc,
	}


def main():
	cfg = build_config_from_cli()
	set_seed(cfg.seed)
	device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

	encoder = TimmEncoder(cfg.backbone, cfg.embed_dim).to(device)
	encoder.eval()  # metric-based inference

	# Train episodes (thực chất là evaluate trên tập train split, vì metric không học được)
	train_ds = ClassFolderDataset(cfg.dataset_root)
	train_res = evaluate_split(
		encoder, device, train_ds,
		episodes=cfg.episodes, n_way=cfg.n_way, k_shot=cfg.k_shot, q_query=cfg.q_query,
		image_size=cfg.image_size, metric=cfg.distance, title="Train/Eval"
	)

	# Validation (nếu cung cấp val_root và số episode > 0)
	if cfg.val_root and cfg.episodes_val and cfg.episodes_val > 0:
		val_ds = ClassFolderDataset(cfg.val_root)
		val_res = evaluate_split(
			encoder, device, val_ds,
			episodes=cfg.episodes_val, n_way=cfg.n_way, k_shot=cfg.k_shot, q_query=cfg.q_query,
			image_size=cfg.image_size, metric=cfg.distance, title="Validation"
		)
		# Lưu file riêng cho val
		out_txt_val = os.path.join(os.getcwd(), "fewshot_report_val.txt")
		with open(out_txt_val, "w", encoding="utf-8") as f:
			f.write(f"Avg acc: {val_res['mean_acc']:.4f} (+/- {val_res['std_acc']:.4f})\n")
			f.write("Classification report (per class):\n")
			f.write(classification_report(val_res["y_true"], val_res["y_pred"], digits=4, output_dict=False))
			f.write("\n\nConfusion matrix:\n")
			f.write(np.array2string(val_res["cm"]))
		print(f"Saved validation report to: {out_txt_val}")

	# Lưu train report như trước
	out_txt = os.path.join(os.getcwd(), "fewshot_report.txt")
	with open(out_txt, "w", encoding="utf-8") as f:
		f.write(f"Avg acc: {train_res['mean_acc']:.4f} (+/- {train_res['std_acc']:.4f})\n")
		f.write("Classification report (per class):\n")
		f.write(classification_report(train_res["y_true"], train_res["y_pred"], digits=4, output_dict=False))
		f.write("\n\nConfusion matrix:\n")
		f.write(np.array2string(train_res["cm"]))
	print(f"Saved report to: {out_txt}")


if __name__ == "__main__":
	main()
