# Decision trees

Source: `Decision+Tree+Classifier+.pdf`, `Decision+Tree+Regressor.pdf`, `Decision+Tree+split+For+Numerical+Features.pdf`, `Decision+Tree+Classifier+Practical+Implementation.ipynb`, `Diabetes+Prediction+Using+Decision+Tree+Regressor.ipynb`.

## What it is

A sequence of if/else splits. Each internal node tests one feature. Each leaf holds a class or a mean value.

Classification leaf: majority class, or the class frequencies if you want a probability.
Regression leaf: mean of the training rows that landed there. That is why a tree regressor predicts a step function, not a smooth curve.

## How a split is chosen

Try thresholds on each feature. Keep the split that most reduces impurity.

- Gini: $1 - \sum_k p_k^2$. Default in sklearn. Cheaper than entropy, usually the same tree.
- Entropy: $-\sum_k p_k \log p_k$. Information gain is the drop in entropy.
- Regression: variance reduction, or equivalently MSE.

Numeric features are split by sorting unique values and testing midpoints. Categorical features are split by grouping levels. The course note on numerical splits is this sorting step.

## Why trees overfit

A deep tree can isolate single rows. Pure leaves look perfect on train and fail on test.

Controls: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`, or cost-complexity pruning (`ccp_alpha`). Pruning beats "I will just set depth to 3" once the data is not tiny.

## Strengths and failures

Strengths: no scaling needed, mixed feature types, a path you can read to a business user.

Failures: axis-aligned splits only, high variance (a small data change can rebuild the tree), weak on linear relationships unless the tree is deep.

## Interview questions

1. Gini vs entropy? Both measure impurity. Gini does not need a log. In practice the trees are close. Do not claim one is always more accurate.
2. How does a tree handle a numeric feature? Sort it, try thresholds between adjacent values, pick the threshold with the best impurity drop.
3. Why is a single tree unstable? The top split decides everything below it. A few flipped rows can change that split. Random forests exist because of this.
4. Can a tree extrapolate? No. A regressor cannot predict above the highest leaf mean it saw. Linear regression can.
5. How do you get probabilities? Class fraction in the leaf. They are coarsely calibrated. Depth and leaf size change them a lot.

Full set: [interview/questions.md](../interview/questions.md#08-decision-trees).

## Kaggle

- [Titanic](https://www.kaggle.com/competitions/titanic): fit a depth-3 tree and read the splits. Sex and class should appear near the top.
- [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) or the course diabetes notebook: tree regressor, then compare RMSE against a linear model.
- [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset): imbalanced classification. A tree will look accurate and still miss the positive class. Use recall and PR-AUC.
