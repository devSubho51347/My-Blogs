"""Dataset generators/loaders for the Streamlit playground."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, make_blobs, make_circles, make_moons


Dataset2DName = Literal["blobs_binary", "blobs_multiclass", "moons", "circles"]


@dataclass(frozen=True)
class Dataset2DConfig:
    name: Dataset2DName
    n_samples: int = 300
    noise: float = 0.2
    seed: int = 42
    centers: int = 3  # multiclass blobs


def make_2d_dataset(cfg: Dataset2DConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.name == "moons":
        X, y = make_moons(n_samples=cfg.n_samples, noise=cfg.noise, random_state=cfg.seed)
        return X.astype(float), y.astype(int)

    if cfg.name == "circles":
        X, y = make_circles(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            factor=0.5,
            random_state=cfg.seed,
        )
        return X.astype(float), y.astype(int)

    if cfg.name == "blobs_binary":
        centers = np.array([[-2.0, -2.0], [2.0, 2.0]])
        X, y = make_blobs(
            n_samples=cfg.n_samples,
            centers=centers,
            cluster_std=max(0.1, cfg.noise),
            random_state=cfg.seed,
        )
        return X.astype(float), y.astype(int)

    if cfg.name == "blobs_multiclass":
        X, y = make_blobs(
            n_samples=cfg.n_samples,
            centers=cfg.centers,
            cluster_std=max(0.1, cfg.noise),
            random_state=cfg.seed,
        )
        return X.astype(float), y.astype(int)

    raise ValueError(f"Unknown dataset name: {cfg.name}")


def load_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    return data.data.astype(float), data.target.astype(int)
