# Decision trees

## Split criteria

**Classification**

- Gini impurity: `1 - Σ p_k²`
- Entropy: `-Σ p_k log p_k`

**Regression**

Variance reduction. The leaf prediction is the mean (or median) of the targets in that leaf.

## Numeric splits

Sort the column. Candidate thresholds lie halfway between consecutive distinct values. Rows with feature ≤ threshold go left.

## Scoring a split

Size-weighted average of the children's impurities:

```math
\text{score} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R
```

Lower is better. Ties are broken by smaller feature index, then smaller threshold.

## Why trees overfit

A single tree can keep splitting until every leaf is pure. The top split also decides every split below it, so the tree is unstable.

## Stopping overfitting

- `max_depth`
- `min_samples_leaf` / `min_samples_split`
- Cost-complexity pruning (`ccp_alpha`)

## Extrapolation

A tree cannot predict past the range of leaf means it saw. It produces a step function.

## Gini vs entropy

Results are usually very similar. Gini is slightly cheaper to compute.

## Coding reference

Best decision-stump split by size-weighted Gini: [interview/coding_questions.md](../interview/coding_questions.md).
