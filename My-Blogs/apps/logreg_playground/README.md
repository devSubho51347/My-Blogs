# Streamlit — Logistic Regression Playground

Interactive Streamlit app to explore **Logistic Regression hyperparameters**.

## Features
- 2D datasets (decision boundary visualization):
  - linearly separable blobs (binary)
  - moons (non-linear)
  - circles (non-linear)
  - multiclass blobs
- Hyperparameters:
  - `C`, `penalty`, `solver`, `max_iter`, `tol`, `class_weight`, `l1_ratio` (elasticnet)
- Animated training (best-effort):
  - uses `warm_start=True` + repeated `fit()` calls with `max_iter=1`
- Breast Cancer dataset page:
  - metrics only (no boundary plot)

## Run
From the repo root:

```bash
pip install -r apps/logreg_playground/requirements.txt
streamlit run apps/logreg_playground/app.py
```

## Notes on animation
The animation is an approximation because sklearn solvers do not expose per-iteration states directly.
The warm-start loop is good for visually understanding the effect of hyperparameters, but different
solvers may converge in fewer frames.
