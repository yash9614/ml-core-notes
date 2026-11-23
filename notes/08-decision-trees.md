# Decision trees

Source: codebasics pet-adoption Decision Tree exercise + core concepts.  
**Industrial relevance added:** why single trees are rarely deployed alone, interpretability use cases, and production practices.

## Split criteria

**Classification**
- Gini impurity: \(1 - \sum p_k^2\)
- Entropy: \(-\sum p_k \log p_k\)

**Regression**
Variance reduction. The leaf prediction is the mean (or median) of the targets in that leaf.

## Numeric splits

Sort the column. Candidate thresholds lie halfway between consecutive distinct values. Rows with feature ≤ threshold go left.

## Scoring a split

Size-weighted average of the children's impurities:

\[
\text{score} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R
\]

Lower is better.

## Why trees overfit

A single tree can keep splitting until every leaf is pure. The top split also decides every split below it, so the tree is unstable (high variance).

## Stopping overfitting

- `max_depth`
- `min_samples_leaf` / `min_samples_split`
- Cost-complexity pruning (`ccp_alpha`)
- Ensemble methods (Random Forest, Gradient Boosting)

## Extrapolation

A tree cannot predict past the range of leaf means it saw. It produces a step function.

## Gini vs entropy

Results are usually very similar. Gini is slightly cheaper to compute.

### Practical code (pet adoption)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("pet_adoption_data.csv")
target = "adoption_likelihood"
X = df.drop(target, axis=1)
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("tree", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Industrial relevance

- Single decision trees are highly interpretable and still used in regulated domains (credit, insurance, healthcare) where stakeholders need to see the exact rules.
- Almost never the best pure predictive model — Random Forest or gradient boosting almost always outperform them.
- Useful for feature importance exploration and for generating business rules that can be implemented without a model serving layer.
- In production, prefer shallow trees (`max_depth` 3–6) or tree ensembles; deep single trees overfit and are brittle to data shifts.

## Interview questions

1. How does a decision tree choose the best split?
2. Why do single decision trees overfit so easily? How do you control it?
3. Compare Gini impurity and entropy. Does the choice matter much in practice?
4. Why can’t a decision tree extrapolate beyond the training data range?
5. When would you deploy a single decision tree instead of a Random Forest?
6. Explain how `min_samples_leaf` and `max_depth` affect bias and variance.
7. Walk through building a Decision Tree pipeline with categorical encoding and evaluation.
8. What is cost-complexity pruning and when would you use it?
