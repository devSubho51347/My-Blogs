from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.reporting import RUNS_DIR, build_pdf_report, list_runs


st.set_page_config(page_title="Reports", layout="wide")
st.title("Saved Snapshots / Reports")

st.markdown(
    """
This page lists previously captured snapshots stored under:

`apps/logreg_playground/runs/`

Each run contains:
- `meta.json`
- images (`decision_boundary.png` for 2D, `confusion_matrix.png`)
- `report.pdf`
"""
)


runs = list_runs()
st.caption(f"Found {len(runs)} run(s) in {RUNS_DIR}")

if len(runs) == 0:
    st.info("No snapshots yet. Go to a page and click **Capture snapshot**.")
    st.stop()

run_names = [p.name for p in runs]
selected = st.selectbox("Select a run", run_names)
run_dir = RUNS_DIR / selected

meta_path = run_dir / "meta.json"
st.subheader("meta.json")
st.code(meta_path.read_text(encoding="utf-8"), language="json")

st.subheader("Artifacts")
cols = st.columns(2)

boundary = run_dir / "decision_boundary.png"
cm = run_dir / "confusion_matrix.png"

with cols[0]:
    if boundary.exists():
        st.image(str(boundary), caption="Decision boundary")
    else:
        st.info("No decision boundary for this run (likely non-2D dataset).")

with cols[1]:
    if cm.exists():
        st.image(str(cm), caption="Confusion matrix")

st.subheader("PDF")
pdf_path = run_dir / "report.pdf"
if not pdf_path.exists():
    if st.button("Generate PDF"):
        build_pdf_report(run_dir)
        st.rerun()
else:
    pdf_bytes = pdf_path.read_bytes()
    st.download_button(
        label="Download report.pdf",
        data=pdf_bytes,
        file_name=f"{run_dir.name}.pdf",
        mime="application/pdf",
    )
