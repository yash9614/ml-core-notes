# Naive Bayes

Source: codebasics Titanic Gaussian NB exercise + core concepts.  
**Industrial relevance added:** when it is (and is not) used in production, plus practical pitfalls.

## What it is

Bayes rule plus a strong assumption: features are independent given the class.

\[
P(y \mid x) \propto P(y) \prod_j P(x_j \mid y)
\]

The assumption is false for almost every real table. The classifier still works when the dependence is mild, especially on text, because only the ranking of classes matters.

## Variants

- **Gaussian**: continuous features, each class has its own mean and variance per column. Used on the Titanic exercise.
- **Multinomial**: counts, such as word frequencies. The classic text baseline.
- **Bernoulli**: binary flags, word present or absent.
- **Categorical**: discrete levels that are not counts.

Zero probability kills the product. Laplace smoothing adds a fake count so an unseen word does not zero the whole class.

### Practical code (Titanic – Gaussian NB)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("titanic.csv")
# Drop low-value columns, handle missing, encode categoricals (simplified)
df = df.drop(columns=["passenger_id", "name", "sib_sp", "parch", "ticket", "cabin", "embarked"], errors="ignore")
df = df.dropna()

# Example encoding
for col in ["sex", "p_class"]:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## Industrial relevance

- Extremely fast to train and predict; still a strong baseline for text classification (spam, intent, document categorization).
- Rarely the final model on tabular data today (gradient boosting usually wins), but useful when you need probabilistic outputs quickly or have very little data.
- Gaussian NB assumes normality per feature per class — check distributions or transform skewed features (e.g., log income) before using it.
- In production pipelines it is sometimes kept as a lightweight fallback or for online learning scenarios.

## Interview questions

1. Why is it called “naive”? The conditional independence assumption.
2. Why does Naive Bayes still work well on text even though words are not independent?
3. What is Laplace smoothing and why is it necessary?
4. When would Gaussian NB perform poorly? (Skewed features, strong feature dependence, multimodal distributions.)
5. Compare Multinomial NB vs Gaussian NB. When would you use each?
6. How does Naive Bayes handle missing values or continuous features?
7. Walk through training a Gaussian NB model on a tabular dataset with sklearn and evaluating it with a classification report.
