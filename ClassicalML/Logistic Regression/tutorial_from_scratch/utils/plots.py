"""Small plotting helpers."""

from __future__ import annotations

from typing import Dict, List


def plot_history(history: Dict[str, List[float]], *, title: str = "Training history"):
    """Plot scalar metrics stored in a dict of lists."""

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for k, v in history.items():
        if len(v) == 0:
            continue
        plt.plot(v, label=k)
    plt.title(title)
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
