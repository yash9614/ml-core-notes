# Supervised ML: Classification – Key Takeaways

Source: codebasics.io Takeaways – Supervised Machine Learning: Classification.

## Regression vs Classification

- Regression: continuous target.  
- Classification: discrete classes.  
  - Binary (two classes)  
  - Multiclass (more than two)

## Logistic Regression (Binary)

Sigmoid function maps any real number to (0, 1):

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

where \( e \approx 2.71828 \).

Used to turn a linear combination of features into a probability.

## Evaluation Metrics

- **Accuracy** = (TP + TN) / Total  
  Misleading when classes are imbalanced.
- **Precision** = TP / (TP + FP)  
  How many of the predicted positives are correct.
- **Recall** = TP / (TP + FN)  
  How many of the actual positives were found.
- **F1** = \( 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \)  
  Harmonic mean; balances precision and recall.
- **Confusion matrix**: table of TP, FP, TN, FN. Best single view of errors.

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

## Scaling

Important for distance-based and gradient-based methods.  
- Min-Max scaling  
- Standard scaling (z-score)

## Sklearn Pipeline

Chains preprocessing steps + estimator into one object.  
Ensures the same transformations are applied at train and predict time; cleaner code and fewer data-leakage bugs.

## Naive Bayes

Bayes’ theorem:

\[ P(A|B) = \frac{P(B|A) P(A)}{P(B)} \]

“Naive” because it assumes features are conditionally independent given the class.  
Works well for text (e.g., MultinomialNB + CountVectorizer for spam classification).

## Decision Trees

Tree of decisions and outcomes.  
Splitting criteria (Gini, entropy) choose the best feature and threshold at each node.

## Class Imbalance

- Undersample majority class  
- Oversample minority (simple duplication or **SMOTE** – synthetic examples via k-NN)  
- SMOTE + Tomek links (clean borderline points)  
- Ensemble methods that balance classes
