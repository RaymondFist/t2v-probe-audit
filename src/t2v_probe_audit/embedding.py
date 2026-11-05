"""DINOv2 frame embeddings and pairwise similarity statistics.

The encoder is DINOv2 ViT-S/14 loaded from a local torch-hub clone. Inputs are
RGB in [0, 1], converted to ImageNet-normalised tensors.
"""

from pathlib import Path

import numpy as np
import torch

from .config import DINO_REPO

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_embedder(repo: Path = DINO_REPO):
    model = torch.hub.load(str(repo), "dinov2_vits14", source="local", verbose=False)
    model.eval()
    return model


@torch.no_grad()
def embed(model, rgb):
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)
    x = (x - MEAN) / STD
    return torch.nn.functional.normalize(model(x), dim=1)


@torch.no_grad()
def embedding_stats(model, rgb):
    """Adjacent-pair mean/min/std and the all-pairs mean.

    The all-pairs mean averages cosine similarity over every unordered pair of
    frames.
    """
    f = embed(model, rgb)
    adjacent = (f[:-1] * f[1:]).sum(dim=1).numpy().astype(np.float64)
    sim = (f @ f.T).numpy().astype(np.float64)
    iu = np.triu_indices(len(f), k=1)
    all_pairs = sim[iu]
    return {
        "dino_pair_mean": float(np.mean(adjacent)),
        "dino_pair_min": float(np.min(adjacent)),
        "dino_pair_std": float(np.std(adjacent)),
        "dino_allpairs_mean": float(np.mean(all_pairs)),
    }
