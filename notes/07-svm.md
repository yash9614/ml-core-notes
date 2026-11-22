# Support vector machines

Source: `6-Support+Vector+Classifier.pdf`, `6.1-SVR.pdf`, `6.2-SVM+Kernels.pdf`, `2.3-+Equation+Of+a+straight+Line,Hyperplane.pdf`, and the matching notebooks.

## Hyperplane

In d dimensions a hyperplane is defined by

```math
w \cdot x + b = 0
```

The signed distance from a point to the plane is proportional to `w · x + b`. SVM looks for the plane that maximizes the margin between the two classes.

## Classifier

Find the hyperplane with the widest margin. Points on the margin, or inside it when the problem is soft, are the support vectors. The other points do not define the boundary.

Hard margin requires perfect separation. Soft margin allows violations, controlled by `C`. Large `C` punishes violations and can overfit. Small `C` allows a wider, simpler margin.

## Kernels

A kernel is a similarity. It lets the boundary curve without you building the extra features.

- **Linear**: same family as logistic regression, different loss. Fast and interpretable.
- **Polynomial**: curved boundary. Degree is a hyperparameter. Degree 2 or 3 is common; higher degrees overfit quickly.
- **RBF (Gaussian)**: local similarity. The default when you do not know the shape. `gamma` sets how far a point’s influence reaches. Large gamma → each point only affects a tiny neighborhood → overfit. Small gamma → almost linear.

Scale features. SVM margins are geometric; unscaled features distort the geometry.

## SVR

Regression with an epsilon-tube. Errors inside the tube are ignored. Errors outside are penalized. Same kernel story. Useful when you want a model that is insensitive to small residuals.

## Probabilities

SVM does not output probabilities by default. sklearn can fit a Platt scaling step (`probability=True`), but it is a separate calibration and adds cost. Logistic regression gives probabilities more naturally.

## Interview questions

1. SVM vs logistic regression? Both can draw a linear boundary. SVM optimizes the margin. Logistic optimizes likelihood. Logistic gives probabilities more naturally. SVM needs a separate calibration step.
2. What is a support vector? A training point on or inside the margin. Delete a non-support point and the hard-margin boundary stays.
3. When does RBF fail? Lots of features, lots of noise, no scaling, gamma left at a bad default.
4. What does the hyperplane equation mean? `w · x + b = 0` is the decision boundary. The sign of `w · x + b` tells you the side.

## Kaggle

Digit Recognizer, linear kernel against RBF. If RBF only wins after scaling, that is the lesson.
