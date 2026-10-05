# Support vector machines

Source: `6-Support+Vector+Classifier.pdf`, `6.1-SVR.pdf`, `6.2-SVM+Kernels.pdf`, `Basic+SVC+Implementation.ipynb`, `SVM+Kernels+Implementation.ipynb`, `Support+Vector+Regression+Implementation.ipynb`.

## Classifier

Find the hyperplane with the widest margin between classes. Points on the margin, or inside it when the problem is soft, are the support vectors. The other points do not define the boundary.

Hard margin requires perfect separation. Soft margin allows violations, controlled by `C`. Large `C` punishes violations and can overfit. Small `C` allows a wider, simpler margin.

## Kernels

A kernel is a similarity. It lets the boundary curve without you building the extra features.

- Linear: same family as logistic regression, different loss.
- Polynomial: curved, degree is a hyperparameter.
- RBF: local similarity. The default when you do not know the shape. `gamma` sets how far a point's influence reaches. Large gamma overfits.

Scale features. SVM margins are geometric.

## SVR

Regression with an epsilon-tube. Errors inside the tube are ignored. Errors outside are penalized. Same kernel story.

## Interview questions

1. SVM vs logistic regression? Both can draw a linear boundary. SVM optimizes the margin. Logistic optimizes likelihood. Logistic gives probabilities more naturally. SVM needs a separate calibration step.
2. What is a support vector? A training point on or inside the margin. Delete a non-support point and the hard-margin boundary stays.
3. When does RBF fail? Lots of features, lots of noise, no scaling, gamma left at a bad default.

## Kaggle

Digit Recognizer, linear kernel against RBF. If RBF only wins after scaling, that is the lesson.
