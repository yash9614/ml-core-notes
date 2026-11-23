# Feature Engineering

Source: codebasics.io Takeaways – Feature Engineering.  
**Industrial relevance added:** production practices, common pitfalls, and what actually moves the needle.

Feature engineering creates or selects features so the model can learn better patterns. In industry this is often the highest-ROI activity after getting clean data—especially on tabular problems.

## Three main ways

1. **Feature cleaning & transformation**  
   Improve quality so the model can use the data:
   - Handle missing values (imputation strategy must match production)
   - Remove duplicates
   - Treat outliers (winsorize, clip, or model them explicitly)
   - Scaling / Normalization
   - Encoding (label / one-hot / target / embedding)

2. **Feature selection**  
   Keep only the most relevant features. Reduces complexity, improves interpretability, and can reduce overfitting and inference latency.

3. **Feature creation**  
   Derive new features from existing ones (ratios, aggregates, interactions, time-based features, domain-specific transforms). This is where domain knowledge pays off most.

**Industrial relevance:**  
- Feature engineering is iterative and driven by error analysis.  
- In production, features must be computed identically at training and serving time (training-serving skew is a top cause of production failures).  
- Prefer features that are stable, available in real-time, and not easily gamed.  
- Feature stores (Feast, Tecton, or custom) are increasingly common for consistency and reuse.

## Feature selection using correlation

- Measures linear relationship between features (and with the target).
- Remove highly correlated redundant features.
- **Do not rely on correlation alone when**:
  - Relationship is non-linear
  - There are strong outliers
  - Variables are categorical
  - You confuse correlation with causation

## Feature selection using VIF (Variance Inflation Factor)

- VIF measures how much a feature’s variance is inflated because of multicollinearity.
- High VIF → feature is redundant with others; coefficients become unstable.
- Multicollinearity may not hurt predictive accuracy much, but it hurts interpretation and coefficient stability.

### Practical process

1. Compute VIF for all features.
2. Drop the feature with the highest VIF.
3. Recompute VIF on the remaining features.
4. Repeat until all VIF values are acceptable (commonly < 5 or < 10).

**Industrial relevance:**  
- In pure predictive settings (e.g., gradient boosting) multicollinearity is often ignored.  
- In interpretable models (logistic regression for credit, risk, pricing) VIF and coefficient stability matter a lot.  
- Always validate that dropping features does not hurt business metrics on a proper hold-out set.

## Interview Questions

1. What is training-serving skew and how do you prevent it in feature engineering?
2. When would you prefer target encoding over one-hot encoding? What are the risks?
3. How do you handle high-cardinality categorical features in production?
4. Explain a time when feature engineering gave a bigger lift than changing the algorithm.
5. How would you decide which features to keep when you have hundreds of them?
6. What is the difference between correlation and causation in the context of feature selection? Why does it matter?
