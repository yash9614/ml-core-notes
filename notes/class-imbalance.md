# Class Imbalance Handling – Practical Notes

Source: codebasics breast-cancer recurrence exercise (undersampling, SMOTE, SMOTE-Tomek) + industrial practice.  
**Industrial relevance added:** production trade-offs and recommended approaches.

Class imbalance is common in fraud, churn, rare-disease, and defect detection. Accuracy becomes misleading; focus on precision, recall, F1, or cost-weighted metrics.

## Techniques demonstrated in the exercise

1. **No handling** — train on the natural (imbalanced) distribution.
2. **Random Undersampling** — reduce the majority class.
3. **SMOTE** — generate synthetic minority examples using k-nearest neighbors.
4. **SMOTE + Tomek Links** — SMOTE followed by removal of borderline (Tomek) pairs to clean the decision boundary.

### Example pattern

```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# After train/test split (apply resampling ONLY on training data)
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

smt = SMOTETomek(random_state=42)
X_smt, y_smt = smt.fit_resample(X_train, y_train)

# Train and evaluate each on the original (imbalanced) test set
for name, Xt, yt in [("Original", X_train, y_train),
                     ("Undersample", X_rus, y_rus),
                     ("SMOTE", X_smote, y_smote),
                     ("SMOTE-Tomek", X_smt, y_smt)]:
    model = LogisticRegression(max_iter=2000)
    model.fit(Xt, yt)
    print(name)
    print(classification_report(y_test, model.predict(X_test)))
    print("---")
```

## Industrial relevance

- Prefer **class weights** (`class_weight='balanced'` or custom) or **threshold tuning** over resampling when the real-world prior must be preserved at inference time.
- Always evaluate on the original (imbalanced) test distribution — never on the resampled set.
- SMOTE can create unrealistic samples and increase overfitting risk; monitor validation carefully.
- In high-stakes domains (fraud, medical), often combine a high-recall model with a secondary review process rather than chasing perfect precision.
- Modern alternatives: focal loss (deep learning), cost-sensitive learning, or anomaly-detection formulations for extreme imbalance.

## Interview questions

1. Why is accuracy a poor metric under class imbalance? What should you report instead?
2. Explain how SMOTE works. What are its main limitations?
3. When would you prefer undersampling over SMOTE?
4. Why must resampling be applied only to the training set?
5. How does `class_weight='balanced'` differ from SMOTE?
6. Describe a production system for detecting rare fraud events. How would you handle imbalance and evaluation?
7. What is the role of Tomek links in the SMOTE-Tomek approach?
