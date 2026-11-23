# Support vector machines

Source: codebasics weather classification exercise (linear vs RBF) + core concepts.  
**Exercise notebook:** [svm_weather_classification.ipynb](07-svm/svm_weather_classification.ipynb)

## Hyperplane

In d dimensions a hyperplane is defined by

\[
w \cdot x + b = 0
\]

The signed distance from a point to the plane is proportional to \(w \cdot x + b\). SVM looks for the plane that maximizes the margin between the two classes.

## Classifier

Find the hyperplane with the widest margin. Points on the margin (or inside it when soft-margin) are the support vectors. The other points do not define the boundary.

Hard margin requires perfect separation. Soft margin allows violations, controlled by `C`.  
- Large `C` → fewer violations, can overfit.  
- Small `C` → wider, simpler margin.

## Kernels

A kernel is a similarity. It lets the boundary curve without you building the extra features.

- **Linear**: same family as logistic regression, different loss. Fast and interpretable.
- **Polynomial**: curved boundary. Degree is a hyperparameter. Degree 2 or 3 is common; higher degrees overfit quickly.
- **RBF (Gaussian)**: local similarity. The default when you do not know the shape. `gamma` sets how far a point’s influence reaches. Large gamma → each point only affects a tiny neighborhood → overfit. Small gamma → almost linear.

**Always scale features.** SVM margins are geometric; unscaled features distort the geometry.

### Practical code (weather classification – linear vs RBF)

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume df is loaded and preprocessed (one-hot + scale done, or use pipeline)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear kernel
svm_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear"))
])
svm_linear.fit(X_train, y_train)
print("Linear accuracy:", svm_linear.score(X_test, y_test))
print(classification_report(y_test, svm_linear.predict(X_test)))

# RBF kernel with custom hyperparameters
svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=0.5, gamma="auto"))
])
svm_rbf.fit(X_train, y_train)
print("RBF accuracy:", svm_rbf.score(X_test, y_test))
print(classification_report(y_test, svm_rbf.predict(X_test)))
```

## SVR

Regression with an epsilon-tube. Errors inside the tube are ignored. Same kernel story.

## Probabilities

SVM does not output probabilities by default. sklearn can fit a Platt scaling step (`probability=True`), but it is a separate calibration and adds cost. Logistic regression gives probabilities more naturally.

## Interview questions

1. SVM vs logistic regression? Both can draw a linear boundary. SVM optimizes the margin. Logistic optimizes likelihood. Logistic gives probabilities more naturally. SVM needs a separate calibration step.
2. What is a support vector? A training point on or inside the margin. Delete a non-support point and the hard-margin boundary stays the same.
3. Explain the effect of `C` and `gamma` in an RBF SVM.
4. Why must you scale features before training an SVM?
5. When would you prefer a linear kernel over RBF?
6. How does the kernel trick work at a high level?
7. Walk through building an sklearn Pipeline that scales data and trains an RBF SVM.
8. What are the practical drawbacks of SVM on large datasets?
