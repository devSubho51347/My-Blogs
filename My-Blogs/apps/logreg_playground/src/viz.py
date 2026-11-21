"""Visualization helpers: decision boundaries and surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Mesh:
    xx: np.ndarray
    yy: np.ndarray
    grid: np.ndarray  # (n_points, 2)


def make_mesh(X: np.ndarray, *, res: int = 250, pad: float = 0.75) -> Mesh:
    x_min, x_max = float(X[:, 0].min() - pad), float(X[:, 0].max() + pad)
    y_min, y_max = float(X[:, 1].min() - pad), float(X[:, 1].max() + pad)

    xs = np.linspace(x_min, x_max, int(res))
    ys = np.linspace(y_min, y_max, int(res))
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return Mesh(xx=xx, yy=yy, grid=grid)


def decision_surface(model, mesh: Mesh) -> np.ndarray:
    Z = model.predict(mesh.grid)
    return Z.reshape(mesh.xx.shape)


def plot_decision_boundary(*, X: np.ndarray, y: np.ndarray, model, mesh: Mesh, title: str):
    import matplotlib.pyplot as plt

    Z = decision_surface(model, mesh)
    n_classes = int(np.unique(y).size)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(mesh.xx, mesh.yy, Z, alpha=0.25, levels=n_classes, cmap="tab10")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap="tab10", edgecolor="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig
