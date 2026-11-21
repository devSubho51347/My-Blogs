from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="LogReg Playground",
    page_icon="📈",
    layout="wide",
)

st.title("Logistic Regression Hyperparameter Playground")

st.markdown(
    """
This app lets you interactively explore **Logistic Regression** hyperparameters.

### Pages
- **2D Decision Boundary**: pick a 2D dataset (linear + non-linear), tune hyperparameters, and watch the boundary evolve over frames.
- **Breast Cancer (metrics)**: run on the real dataset (higher dimensional) and inspect metrics.

Use the left sidebar to navigate pages.
"""
)
