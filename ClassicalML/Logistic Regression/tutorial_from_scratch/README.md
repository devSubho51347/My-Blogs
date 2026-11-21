# Logistic Regression — From Scratch (Tutorial Notebooks)

This folder contains a **step-by-step** implementation of **binary logistic regression** built from scratch.

## Rules followed in this tutorial
- `sklearn` is used **only to load the dataset** (`load_breast_cancer`).
  Splitting, scaling, metrics, training loop: **from scratch**.
- Implementation is shown incrementally: minimal → better engineering → better accuracy.

## Notebooks (recommended order)
All notebooks are under: `notebooks/`

1. **00_setup_and_dataset.ipynb**
   - dataset loading
   - scratch train/test split
   - why feature scaling matters

2. **01_minimal_logreg_naive_gd.ipynb**
   - minimal end-to-end logistic regression
   - gradients with explicit loops

3. **02_vectorized_logreg_gd.ipynb**
   - same model, vectorized gradients
   - loss curve

4. **03_feature_scaling_and_minibatch.ipynb**
   - standardization (scratch)
   - mini-batch GD
   - usually a *big* accuracy jump

5. **04_regularization_L2_and_thresholding.ipynb**
   - validation split
   - L2 regularization
   - threshold tuning

6. **05_metrics_and_learning_curves.ipynb**
   - confusion matrix
   - precision/recall/F1
   - learning curves

7. **06_early_stopping.ipynb**
   - early stopping using validation loss

## Quick sanity check (accuracy progression)
Run:

```bash
python "ClassicalML/Logistic Regression/tutorial_from_scratch/scripts/run_all_and_summarize.py"
```

Example output (your numbers may vary slightly by machine/versions):

```
01 naive GD (unscaled)               ~0.88
02 vectorized GD (unscaled)          ~0.88
03 scaled + mini-batch               ~0.96
04 scaled + (optional) L2            ~0.97
06 early stopping                    ~0.97
```

## Implementation files
- `utils/`:
  - `data.py`: scratch splitting + standardization
  - `metrics.py`: scratch metrics
  - `plots.py`: lightweight plotting
- `scripts/`:
  - `logreg_scratch.py`: reusable logistic regression implementation
  - `generate_notebooks.py`: writes the notebooks programmatically
  - `run_all_and_summarize.py`: quick accuracy progression check
