# Supervised ML: Classification – Key Takeaways

Source: codebasics.io Takeaways – Supervised Machine Learning: Classification.  
**Industrial relevance added:** production metrics, class imbalance handling, and real-world trade-offs.

## Regression vs Classification

- Regression: continuous target.  
- Classification: discrete classes.  
  - Binary (two classes)  
  - Multiclass (more than two)

## Logistic Regression (Binary)

Sigmoid function maps any real number to (0, 1):

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

Used to turn a linear combination of features into a probability.

**Industrial relevance:**  
- Extremely common as an interpretable baseline (credit scoring, risk, marketing propensity).  
- Coefficients can be converted to odds ratios for business stakeholders.  
- Always calibrate probabilities if you need reliable scores (Platt scaling or isotonic regression).

## Evaluation Metrics

- **Accuracy** = (TP + TN) / Total  
  Misleading when classes are imbalanced.
- **Precision** = TP / (TP + FP)  
  How many of the predicted positives are correct. Critical when false positives are expensive.
- **Recall** = TP / (TP + FN)  
  How many of the actual positives were found. Critical when false negatives are expensive.
- **F1** = \( 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \)  
  Harmonic mean; balances precision and recall.
- **Confusion matrix**: table of TP, FP, TN, FN. Best single view of errors.
- **AUC-ROC**: ranking quality across thresholds.

**Industrial relevance:**  
- Choose the metric based on business cost. Fraud detection usually optimizes recall or a cost-weighted metric. Spam filtering often optimizes precision.  
- Always look at the confusion matrix and segment performance (by customer type, region, etc.).

## Cost Function – Log Loss

MSE is not suitable for logistic regression (non-convex).  
Use **Log Loss** (Binary Cross-Entropy / Logistic Loss):

\[ -\frac{1}{N}\sum [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)] \]

## SVM

- Finds the optimal separating hyperplane.
- Kernel trick maps data to higher dimensions (linear, poly, RBF, sigmoid).
- **Gamma**: how far the influence of a single training example reaches.
- Regularization (C) trades off margin size vs. training errors.
- Kernel choice strongly affects compute cost.

**Industrial relevance:**  
- Less common than gradient boosting or logistic regression on large tabular data today because of scaling and interpretability.  
- Still useful for high-dimensional text or when a clear margin is desired.

## Scaling

Important for distance-based and gradient-based methods.  
- Min-Max scaling  
- Standard scaling (z-score)

## Sklearn Pipeline

Chains preprocessing steps + estimator into one object.  
Ensures the same transformations are applied at train and predict time; cleaner code and fewer data-leakage bugs. Strongly preferred in production code.

## Naive Bayes

Bayes’ theorem:

\[ P(A|B) = \frac{P(B|A) P(A)}{P(B)} \]

“Naive” because it assumes features are conditionally independent given the class.  
Works well for text (e.g., MultinomialNB + CountVectorizer for spam classification). Fast and surprisingly strong baseline.

## Decision Trees

Tree of decisions and outcomes.  
Splitting criteria (Gini, entropy) choose the best feature and threshold at each node.  
Prone to overfitting unless pruned or used inside ensembles.

## Class Imbalance

- Undersample majority class  
- Oversample minority (simple duplication or **SMOTE** – synthetic examples via k-NN)  
- SMOTE + Tomek links (clean borderline points)  
- Ensemble methods that balance classes  
- Class weights / cost-sensitive learning

**Industrial relevance:**  
- In production, prefer class weights or proper threshold tuning over aggressive oversampling if the original distribution matters at inference time.  
- Always evaluate on the real (imbalanced) distribution, not the resampled one.  
- Business often cares more about precision-recall at a specific operating point than overall accuracy.

## Interview Questions

1. Why is accuracy a bad metric for imbalanced classification problems? What would you use instead?
2. Explain the difference between precision and recall. Give a business example for each.
3. What is the log-loss / binary cross-entropy and why is it used for logistic regression?
4. How does SMOTE work and what are its limitations?
5. When would you choose logistic regression over a more complex model like XGBoost?
6. How do you handle class imbalance in a production system?
7. What is the role of the confusion matrix in model evaluation?
