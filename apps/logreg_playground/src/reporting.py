"""Snapshot + PDF reporting utilities.

Snapshots are stored under:
apps/logreg_playground/runs/<run_id>/
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional


RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def _jsonify(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable forms."""
    if is_dataclass(obj):
        return _jsonify(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]

    # numpy arrays, pandas series, etc.
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # numpy scalars
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass

    return obj


def make_run_id(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(c for c in prefix if c.isalnum() or c in ("-", "_"))[:40]
    return f"{ts}_{safe_prefix}"


def save_snapshot(
    *,
    run_id: str,
    meta: Dict[str, Any],
    boundary_fig=None,
    confusion_fig=None,
) -> Path:
    """Save snapshot assets.

    Parameters
    ----------
    run_id: folder name
    meta: JSON-serializable dict
    boundary_fig: matplotlib figure (optional)
    confusion_fig: matplotlib figure (optional)
    """

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(_jsonify(meta), indent=2),
        encoding="utf-8",
    )

    if boundary_fig is not None:
        boundary_fig.savefig(out_dir / "decision_boundary.png", dpi=200, bbox_inches="tight")

    if confusion_fig is not None:
        confusion_fig.savefig(out_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")

    return out_dir


def build_pdf_report(run_dir: Path, *, title: str = "LogReg Playground Report") -> Path:
    """Create a PDF report for a run directory."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    pdf_path = run_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: summary text
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait inches
        fig.suptitle(title, fontsize=16, y=0.98)

        lines = []
        lines.append(f"run_id: {run_dir.name}")
        for key in ["timestamp", "page", "dataset", "standardize", "hyperparameters", "metrics"]:
            if key in meta:
                lines.append("")
                lines.append(f"{key}:")
                val = meta[key]
                # pretty print dicts
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        lines.append(f"  - {k2}: {v2}")
                else:
                    lines.append(f"  {val}")

        fig.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # Optional images
        boundary = run_dir / "decision_boundary.png"
        if boundary.exists():
            img = plt.imread(boundary)
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Decision boundary")
            pdf.savefig(fig)
            plt.close(fig)

        cm = run_dir / "confusion_matrix.png"
        if cm.exists():
            img = plt.imread(cm)
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Confusion matrix")
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)
