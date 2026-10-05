# Naive Bayes

Source: `1.0-Naive+Bayes+Classifier+Indepth+Intuition.pdf`, `2.0-Variants+of+Naive+Bayes+.pdf`, `3.0-Naive+Bayes+Implementation.ipynb`.

## What it is

Bayes rule plus a strong assumption: features are independent given the class.

$$
P(y \mid x) \propto P(y) \prod_j P(x_j \mid y)
$$

The assumption is false for almost every real table. The classifier still works when the dependence is mild, especially on text, because only the ranking of classes matters.

## Variants

- Gaussian: continuous features, each class has its own mean and variance per column.
- Multinomial: counts, such as word frequencies. The text baseline.
- Bernoulli: binary flags, word present or absent.
- Categorical: discrete levels that are not counts.

Zero probability kills the product. Laplace smoothing adds a fake count so an unseen word does not zero the whole class.

## Interview questions

1. Why naive? The independence assumption.
2. Why does it work on text anyway? Words are not independent, but the model only needs a decent ranking, and each word probability is a stable estimate.
3. Gaussian NB on raw income? A bad fit if income is skewed. Log-transform, or do not use Gaussian NB.

Full set: [interview/questions.md](../interview/questions.md#06-naive-bayes).

## Kaggle

[Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started). Multinomial NB on bag-of-words is the baseline before any neural encoder.
