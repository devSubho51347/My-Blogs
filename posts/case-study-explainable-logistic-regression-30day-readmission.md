# Explainable Logistic Regression in Healthcare Triage (30‑Day Readmission): A Case Study With the Math

*A practical, math-first walkthrough of why logistic regression remains one of the most explainable—and deployable—models in high-stakes settings.*

---

## Table of contents

1. [Case study: why we are predicting 30‑day readmission](#case-study-why-we-are-predicting-30-day-readmission)
2. [What “explainability” needs to mean in a hospital](#what-explainability-needs-to-mean-in-a-hospital)
3. [The logistic regression model: interpretable by construction](#the-logistic-regression-model-interpretable-by-construction)
4. [The math: from probabilities → log-odds → a linear model](#the-math-from-probabilities--log-odds--a-linear-model)
5. [Training as maximum likelihood (cross-entropy)](#training-as-maximum-likelihood-cross-entropy)
6. [Gradients, Hessian, and why the objective is convex](#gradients-hessian-and-why-the-objective-is-convex)
7. [Regularization: stability, generalization, and a Bayesian lens](#regularization-stability-generalization-and-a-bayesian-lens)
8. [Global explainability: coefficients, odds ratios, and confidence intervals](#global-explainability-coefficients-odds-ratios-and-confidence-intervals)
9. [Local explainability: explaining a single patient prediction](#local-explainability-explaining-a-single-patient-prediction)
10. [Thresholding and triage: explainability meets operations](#thresholding-and-triage-explainability-meets-operations)
11. [Calibration: when “0.82 risk” must mean ~82%](#calibration-when-082-risk-must-mean-82)
12. [Common pitfalls (and how to explain them honestly)](#common-pitfalls-and-how-to-explain-them-honestly)
13. [A deployment checklist for explainable logistic regression](#a-deployment-checklist-for-explainable-logistic-regression)
14. [Closing thoughts: when logistic regression is the right model](#closing-thoughts-when-logistic-regression-is-the-right-model)

---

## Case study: why we are predicting 30‑day readmission

Imagine a hospital network that wants to reduce **avoidable 30‑day readmissions**.

Readmissions are expensive, often preventable, and frequently used as quality metrics. But operationally, the hospital has limited capacity:

- care coordinators can only handle a fixed number of follow-up plans per day,
- clinicians need reasons, not just scores,
- leadership needs aggregated insight (which factors drive readmission risk?),
- and compliance teams need decisions to be *auditable*.

### The prediction task

For each discharge (or near-discharge) patient, we want to predict:

> **Will this patient be readmitted within 30 days?**

We’ll encode the label as:

- \(y = 1\): *readmitted within 30 days* (“high risk”)
- \(y = 0\): *not readmitted within 30 days* (“low risk”)

### Why not a black-box model?

Deep models and boosted trees can score well, but in many hospitals the most immediate bottleneck is not raw AUC—it is **trust and workflow adoption**. A model that clinicians cannot interrogate is often a model that never makes it into practice.

Logistic regression is a strong default because:

1. It produces **probabilities** (not just class labels).
2. It can be **calibrated** and validated.
3. Its parameters have a direct **explanatory meaning**.
4. It’s easy to monitor and maintain.

---

## What explainability needs to mean in a hospital

“Explainability” is overloaded. In this case study, we’ll be precise.

### Stakeholders and the explanations they need

1. **Clinicians** (local explanations)
   - *Why* is *this* patient flagged high risk?
   - What factors are pushing the risk up/down?
   - What actions would reduce risk (within reason)?

2. **Care management / operations** (thresholding explanations)
   - If we intervene on the top 5% highest risk, how many readmissions do we likely prevent?
   - What will be the workload? What’s the trade-off between sensitivity and precision?

3. **Quality / compliance** (auditability)
   - Can we document the model form and the training objective?
   - Are we using sensible features? Any leakage?
   - Can we quantify uncertainty and stability?

4. **ML team** (diagnostics)
   - Is the model calibrated?
   - Are coefficients stable across time/hospitals?
   - Are there dataset shifts that change feature meaning?

### Two kinds of explanations: global vs local

- **Global explanation**: what the model has learned *in general*.
  - Which factors most increase/decrease risk?
  - Are learned directions clinically plausible?

- **Local explanation**: why the model predicted what it did for *one patient*.
  - Which inputs contributed most to this score?
  - What would need to change to reduce the probability?

Logistic regression naturally supports both because it is **additive** in a meaningful scale (log-odds).

---

## The logistic regression model: interpretable by construction

Logistic regression is not “interpretable because it’s simple.” It’s interpretable because it models a **linear relationship in the log-odds**, which decomposes into **additive feature contributions**.

### Notation

- Dataset \(\{(x_i, y_i)\}_{i=1}^{n}\)
- Features \(x_i \in \mathbb{R}^d\)
- Label \(y_i \in \{0,1\}\)
- Parameter vector \(w \in \mathbb{R}^d\)
- Intercept (bias) \(b \in \mathbb{R}\)

Define the linear score:

\[
z_i = w^\top x_i + b
\]

and the predicted probability:

\[
\hat{p}_i = P(y_i=1\mid x_i) = \sigma(z_i)
\]

where \(\sigma\) is the sigmoid:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}.
\]

---

## The math: from probabilities → log-odds → a linear model

The key object is the **odds** and **log-odds**.

### Odds and log-odds

If an event has probability \(p\), its **odds** are:

\[
\text{odds}(p) = \frac{p}{1-p}
\]

and its **log-odds** (also called the **logit**) are:

\[
\text{logit}(p) = \log\frac{p}{1-p}.
\]

Logistic regression assumes:

\[
\log\frac{P(y=1\mid x)}{1-P(y=1\mid x)} = w^\top x + b.
\]

This statement is the heart of interpretability.

### Why does this create explainability?

Because the log-odds are **additive**:

\[
w^\top x + b = \sum_{j=1}^{d} w_j x_j + b.
\]

Each feature contributes **linearly** to the log-odds. That makes explanations straightforward:

- sign of \(w_j\) tells direction (risk up/down)
- magnitude of \(w_j\) tells strength (subject to feature scale)
- \(w_j x_j\) tells the *patient-specific* contribution in log-odds space

### The logistic link

If the log-odds are linear, the probability is:

\[
p = \sigma(w^\top x + b).
\]

This yields a probability in \([0,1]\), and the nonlinearity lives *only* in the final mapping from log-odds to probability.

---

## Training as maximum likelihood (cross-entropy)

Logistic regression is usually trained by **maximum likelihood estimation (MLE)**.

### Bernoulli likelihood

For one sample \((x_i, y_i)\), the Bernoulli model gives:

\[
P(y_i\mid x_i) = \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}
\]

where \(\hat{p}_i = \sigma(w^\top x_i + b)\).

Assuming i.i.d. samples, the likelihood of all data is:

\[
\mathcal{L}(w,b) = \prod_{i=1}^{n} \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}.
\]

### Log-likelihood

Taking log (easier to optimize):

\[
\ell(w,b) = \sum_{i=1}^{n} \Big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big].
\]

MLE means maximizing \(\ell\). Equivalently, we minimize the **negative log-likelihood** (NLL):

\[
J(w,b) = -\ell(w,b) = -\sum_{i=1}^{n} \Big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big].
\]

If we average over \(n\), we obtain the familiar **binary cross-entropy**:

\[
\text{BCE}(w,b) = -\frac{1}{n}\sum_{i=1}^{n} \Big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big].
\]

### Why this objective is explainability-friendly

Because every part is explicit:

- the model form \(\hat{p} = \sigma(w^\top x + b)\)
- the probability assumptions (Bernoulli)
- the training criterion (maximize likelihood / minimize NLL)

This matters in audits: you can literally write down the objective the model optimized.

---

## Gradients, Hessian, and why the objective is convex

Explainability isn’t just “how you interpret coefficients.” It also includes whether the training procedure is well-behaved.

Logistic regression has a major advantage:

> The negative log-likelihood is **convex** in \((w,b)\). There is a unique global optimum (up to degenerate feature spaces).

That is a strong property when you need a model that is stable and reproducible.

### Vectorized setup

Let:

- \(X \in \mathbb{R}^{n\times d}\) be the design matrix (rows are \(x_i^\top\))
- \(y \in \mathbb{R}^{n}\) be the label vector
- \(w \in \mathbb{R}^{d}\)
- \(\mathbf{1}\in\mathbb{R}^{n}\) be an all-ones vector

Define:

\[
z = Xw + b\mathbf{1}, \quad \hat{p} = \sigma(z)
\]

where \(\sigma\) is applied elementwise.

### Gradient (derivation sketch)

For one datapoint, the negative log-likelihood is:

\[
J_i = -\Big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big]
\]

Using \(\frac{d}{dz}\sigma(z)=\sigma(z)(1-\sigma(z))\) and chaining through, you get:

\[
\frac{\partial J}{\partial w} = \frac{1}{n}X^\top(\hat{p}-y)
\]

\[
\frac{\partial J}{\partial b} = \frac{1}{n}\mathbf{1}^\top(\hat{p}-y) = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i-y_i).
\]

Interpretation:

- the residual \((\hat{p}-y)\) drives learning
- gradients are linear combinations of features weighted by residuals

### Hessian

Define a diagonal matrix \(R\in\mathbb{R}^{n\times n}\) with:

\[
R_{ii} = \hat{p}_i(1-\hat{p}_i).
\]

Then the Hessian w.r.t. \(w\) is:

\[
H = \nabla_w^2 J = \frac{1}{n}X^\top R X.
\]

### Convexity intuition (why this is PSD)

For any vector \(v\in\mathbb{R}^d\):

\[
v^\top H v = \frac{1}{n}v^\top X^\top R X v = \frac{1}{n}(Xv)^\top R (Xv).
\]

Since \(R\) is diagonal with nonnegative entries \(\hat{p}_i(1-\hat{p}_i)\ge 0\), we have \((Xv)^\top R (Xv) \ge 0\). Therefore \(H\) is **positive semidefinite**, hence \(J\) is convex.

### Newton’s method / IRLS (optional but insightful)

The Newton update is:

\[
w_{t+1} = w_t - H^{-1}\nabla_w J.
\]

For logistic regression, this leads to the classical **Iteratively Reweighted Least Squares (IRLS)** algorithm.

Many libraries use quasi-Newton solvers (L-BFGS) because they scale well.

In practice, simple gradient descent also works and is easy to implement from scratch.

---

## Regularization: stability, generalization, and a Bayesian lens

In healthcare data, features can be correlated, noisy, and sparse. Regularization is essential for both **performance** and **interpretability**.

### L2-regularized logistic regression (ridge)

Add a penalty to discourage large weights:

\[
J_{\text{L2}}(w,b) = J(w,b) + \lambda\lVert w\rVert_2^2.
\]

The gradient becomes:

\[
\nabla_w J_{\text{L2}} = \frac{1}{n}X^\top(\hat{p}-y) + 2\lambda w.
\]

**Interpretability benefit:** L2 tends to make coefficients **more stable** under multicollinearity (e.g., lab measures that move together), reducing “sign flipping” across retrains.

### L1-regularized logistic regression (lasso)

\[
J_{\text{L1}}(w,b) = J(w,b) + \lambda\lVert w\rVert_1.
\]

L1 encourages **sparsity** (many weights become exactly zero), which can be appealing for clinical simplicity.

Trade-off: L1 can be less stable when multiple correlated features could “share credit.” The model may arbitrarily pick one.

### MAP interpretation (why regularization is a prior)

Regularization has a clean probabilistic interpretation:

- Minimizing NLL corresponds to maximizing likelihood.
- Adding an L2 penalty corresponds to putting a **Gaussian prior** on weights.

Specifically, if \(w \sim \mathcal{N}(0, \sigma^2 I)\), then:

\[
-\log P(w) \propto \frac{1}{2\sigma^2}\lVert w\rVert_2^2.
\]

So minimizing \(J + \lambda\lVert w\rVert_2^2\) is equivalent to **maximum a posteriori (MAP)** estimation.

In explainability discussions, this is a powerful story:

> “We assume, *a priori*, that extreme effects are unlikely unless supported by data.”

---

## Global explainability: coefficients, odds ratios, and confidence intervals

Global explainability means answering: *“What variables matter overall, and in what direction?”*

### 1) Coefficients as directions in log-odds space

From:

\[
\log\frac{p}{1-p} = w^\top x + b
\]

If feature \(x_j\) increases by one unit (holding others fixed), the log-odds change by \(w_j\).

This statement is exact.

### 2) Odds ratios

Exponentiating log-odds yields an “odds multiplier.” For a +1 unit change in \(x_j\):

\[
\frac{\text{odds}(p\;\text{after})}{\text{odds}(p\;\text{before})} = e^{w_j}.
\]

So \(e^{w_j}\) is the **odds ratio (OR)** for a one-unit increase in \(x_j\).

Example interpretations (hypothetical):

- If \(w_{\text{prior_admissions}} = 0.55\), then OR \(= e^{0.55}\approx 1.73\).
  - “Each additional prior admission multiplies the odds of 30‑day readmission by ~1.73 (all else equal).”

- If \(w_{\text{followup_scheduled}} = -0.80\), then OR \(= e^{-0.8}\approx 0.45\).
  - “Scheduling follow-up reduces the odds by ~55% (all else equal).”

### 3) The critical caveat: scaling changes interpretation

Coefficient magnitudes depend on feature scale.

- If you measure age in years vs decades, \(w\) changes by a factor of 10.
- If you standardize features (mean 0, std 1), then \(w_j\) corresponds to **a +1 standard deviation change**.

In practice, a common pattern for explainability:

1. standardize continuous variables,
2. keep binary indicators as 0/1,
3. then compare coefficient magnitudes more fairly.

### 4) Uncertainty: standard errors and confidence intervals

In high-stakes decisions, stakeholders will ask:

> “Is this effect real, or noise?”

For MLE, under standard regularity conditions, the estimator is approximately normal:

\[
\hat{w} \approx \mathcal{N}(w,\; (X^\top R X)^{-1})
\]

where \(R\) uses \(\hat{p}_i(1-\hat{p}_i)\).

So an approximate standard error for coefficient \(j\) is:

\[
\text{SE}(\hat{w}_j) = \sqrt{\big[(X^\top R X)^{-1}\big]_{jj}}.
\]

The (approximate) 95% Wald CI is:

\[
\hat{w}_j \pm 1.96\;\text{SE}(\hat{w}_j).
\]

And for the odds ratio \(e^{w_j}\), the CI is:

\[
\exp(\hat{w}_j \pm 1.96\;\text{SE}(\hat{w}_j)).
\]

This is one of the most underused *and* most persuasive explainability tools in linear models.

---

## Local explainability: explaining a single patient prediction

Now we answer the clinician question:

> “Why did the model flag this patient?”

### The additive decomposition (log-odds contributions)

For a specific patient feature vector \(x\):

\[
z = w^\top x + b = \sum_{j=1}^{d} w_j x_j + b.
\]

Each term \(w_j x_j\) is a **feature contribution** to the log-odds.

This is as direct as local explanations get.

### A worked mini-example (hypothetical)

Suppose we fit a model with these (simplified) standardized features:

- \(x_1\): age (standardized)
- \(x_2\): prior admissions in last year (standardized)
- \(x_3\): length of stay (standardized)
- \(x_4\): follow-up scheduled within 7 days (binary 0/1)

Assume learned parameters:

```text
b  = -0.40
w1 = +0.30   (age)
w2 = +0.80   (prior admissions)
w3 = +0.25   (length of stay)
w4 = -0.90   (follow-up scheduled)
```

Patient A has:

```text
x1 = +0.6   (older than average)
x2 = +1.2   (many prior admissions)
x3 = +0.3   (slightly longer stay)
x4 =  0     (no follow-up scheduled)
```

Compute contributions:

```text
age contribution                = 0.30 * 0.6  = +0.18
prior admissions contribution   = 0.80 * 1.2  = +0.96
length-of-stay contribution     = 0.25 * 0.3  = +0.075
follow-up contribution          = -0.90 * 0   =  0
intercept                       = -0.40

logit z                         = -0.40 + 0.18 + 0.96 + 0.075 = 0.815
```

Convert logit to probability:

\[
\hat{p} = \sigma(0.815) \approx 0.693.
\]

So the model says: ~69% readmission risk.

**Local explanation in plain language:**

- “The strongest driver is prior admissions (+0.96 log-odds).”
- “Age contributes moderately (+0.18 log-odds).”
- “No follow-up plan removes a protective effect (you’re missing a -0.90 term).”

### Turning contributions into action (counterfactual thinking)

If a follow-up were scheduled (\(x_4=1\)):

```text
new z = 0.815 + (-0.90) = -0.085
new p = sigmoid(-0.085) ≈ 0.479
```

This is a compelling care-management story:

> “Scheduling follow-up is associated with a major reduction in predicted risk, all else equal.”

**Important warning:** association is not causation. You must communicate this carefully (see pitfalls).

---

## Thresholding and triage: explainability meets operations

Hospitals rarely act on raw probability. They act on **decisions**: who gets an intervention.

### Why the default 0.5 threshold is often wrong

Readmission can be imbalanced (say 10–20% positive). If you use 0.5:

- you might flag too few patients,
- you may miss clinically important cases,
- and the threshold has no operational meaning.

### Choosing thresholds in a workflow

Common strategies:

1. **Capacity-based threshold**
   - “We can intervene on 30 patients/day.”
   - Choose the threshold so the top 30 scores are flagged.

2. **Constraint-based threshold**
   - “Sensitivity must be at least 0.80.”
   - Choose the lowest threshold meeting the constraint.

3. **Cost-based threshold**
   - Set a threshold based on the relative cost of false negatives vs false positives.
   - (Often hard to quantify precisely in healthcare, but still useful conceptually.)

### Explaining the confusion matrix in clinical terms

- **True positive (TP):** correctly flagged readmission risk → intervention likely helpful.
- **False positive (FP):** flagged but no readmission → wasted effort / alarm fatigue.
- **False negative (FN):** not flagged but readmission happens → missed opportunity.
- **True negative (TN):** not flagged and no readmission → efficient.

Because logistic regression is simple, the threshold decision becomes a transparent policy layer above the model.

---

## Calibration: when “0.82 risk” must mean ~82%

Explainability collapses if probabilities are not trustworthy.

### What calibration is

If among patients assigned \(\hat{p} \approx 0.20\), about 20% are actually readmitted, then the model is calibrated around 0.20.

### Why logistic regression often calibrates well

Logistic regression is a proper probabilistic model trained with a proper scoring rule (NLL). That tends to help calibration.

But calibration is not guaranteed—dataset shift and regularization can affect it.

### How to measure calibration

- **Reliability diagram** (calibration curve)
- **Brier score** (mean squared error of probabilities)

### How to fix calibration if needed

Post-hoc calibration methods:

- **Platt scaling** (fit another logistic regression on the scores)
- **Isotonic regression** (nonparametric, flexible)

In a hospital, calibrated probabilities are valuable because:

- they enable consistent triage rules (“flag if risk > 0.35”),
- they support communication (“risk decreased from 0.62 to 0.41 after intervention plan”),
- they facilitate monitoring (drift shows up as miscalibration).

---

## Common pitfalls (and how to explain them honestly)

Logistic regression is explainable, but explanations can still mislead if you ignore the data realities.

### 1) Multicollinearity (correlated features)

If two features measure similar underlying concepts (e.g., multiple kidney markers), coefficients can become unstable.

Symptoms:

- big coefficients with wide confidence intervals
- sign flips across retrains

Mitigations:

- L2 regularization
- feature grouping or dimensionality reduction
- clinical review of redundant variables

### 2) Missingness is itself a signal

In EHR data, “missing lab” can imply “not ordered,” which can correlate with severity.

You must choose a missing-data policy (imputation + missingness indicators) and document it, because it affects explainability.

### 3) Leakage (the silent killer of trust)

If you include features that are only available *after discharge* (or reflect future care decisions), performance can look great and explanations can look clinically plausible—but the model is invalid.

Explainability demands strict time-aware feature definitions.

### 4) “Coefficient = causality” (no)

Logistic regression is associative. A negative coefficient on “follow-up scheduled” does not prove follow-up prevents readmission.

How to communicate responsibly:

- “This factor is *predictive* in the observed data.”
- “We should validate clinically and consider causal study design if acting on it.”

### 5) Nonlinear effects and interactions

Many clinical effects are nonlinear (e.g., risk rises sharply after a threshold).

Logistic regression can still handle this *without losing interpretability*:

- add **splines** or **binning** for continuous variables
- include clinically motivated **interaction terms** (e.g., age × comorbidity index)

The key: the model remains linear in the expanded feature space, so contributions are still additive and explainable.

---

## A deployment checklist for explainable logistic regression

Use this as an “explainability-ready” checklist.

### Data + features

- [ ] Feature definitions are time-consistent (no leakage).
- [ ] Continuous features are scaled (or interpretation explicitly accounts for scale).
- [ ] Missing data strategy is documented.
- [ ] Protected attributes and fairness considerations are addressed.

### Model training

- [ ] Objective written down (NLL / cross-entropy), solver documented.
- [ ] Regularization chosen intentionally (L2 for stability; L1 for sparsity).
- [ ] Class imbalance handled (reweighting, threshold policy, metrics).

### Explainability artifacts

- [ ] Global coefficient table with odds ratios.
- [ ] Confidence intervals for key coefficients (where feasible).
- [ ] Per-patient “top contributors” explanation in log-odds space.
- [ ] Clear disclaimer on association vs causation.

### Evaluation

- [ ] Discrimination: ROC-AUC / PR-AUC.
- [ ] Calibration: reliability curve + Brier score.
- [ ] Operating point: threshold chosen with clinical/ops input.
- [ ] Subgroup performance analysis.

### Monitoring

- [ ] Drift monitoring (feature distribution shifts).
- [ ] Calibration monitoring over time.
- [ ] Periodic coefficient stability checks.

---

## Closing thoughts: when logistic regression is the right model

Logistic regression is often the best first deployable model in healthcare triage because it aligns with the real-world constraints:

- explanations must be immediate and defensible,
- probabilities must be interpretable and calibratable,
- updates must be stable,
- and the training objective must be easy to audit.

If you need more accuracy later, logistic regression still provides value as:

- a baseline,
- a calibration layer,
- or a distilled explanation model.

---

## Appendix A: one-page mathematical summary

**Model:**

\[
\hat{p} = \sigma(Xw + b\mathbf{1}), \quad \sigma(z)=\frac{1}{1+e^{-z}}
\]

**Loss (average NLL / BCE):**

\[
J(w,b)= -\frac{1}{n}\sum_{i=1}^{n} \Big[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big]
\]

**Gradients:**

\[
\nabla_w J = \frac{1}{n}X^\top(\hat{p}-y), \quad \frac{\partial J}{\partial b} = \frac{1}{n}\sum_i(\hat{p}_i-y_i)
\]

**Hessian:**

\[
\nabla_w^2 J = \frac{1}{n}X^\top R X, \quad R_{ii}=\hat{p}_i(1-\hat{p}_i)
\]

**Log-odds (interpretability core):**

\[
\log\frac{\hat{p}}{1-\hat{p}} = w^\top x + b
\]

**Odds ratio for feature \(j\):**

\[
\text{OR}_j = e^{w_j}
\]

---

### Related code in this repo

If you want a scratch implementation (useful for teaching or internal review), see:

- `ClassicalML/Logistic Regression/tutorial_from_scratch/`

This case study is domain-focused (readmissions) while the scratch tutorial is dataset-focused; the math and the explainability story are the same.
