# Feature Engineering

Source: codebasics.io Takeaways – Feature Engineering.

Feature engineering creates or selects features so the model can learn better patterns. It can be driven by business understanding or by statistics.

## Three main ways

1. **Feature cleaning & transformation**  
   Improve quality so the model can use the data:
   - Handle missing values
   - Remove duplicates
   - Treat outliers
   - Scaling
   - Normalization
   - Encoding (label / one-hot)

2. **Feature selection**  
   Keep only the most relevant features. Reduces complexity and can improve performance and interpretability.

3. **Feature creation**  
   Derive new features from existing ones (ratios, aggregates, interactions, domain-specific transforms).

## Feature selection using correlation

- Measures linear relationship between features (and with the target).
- Remove highly correlated redundant features.
- **Do not rely on correlation alone when**:
  - Relationship is non-linear
  - There are strong outliers
  - Variables are categorical
  - You confuse correlation with causation

## Feature selection using VIF (Variance Inflation Factor)

- VIF measures how much a feature’s variance is inflated because of multicollinearity (features highly correlated with each other).
- High VIF → feature is redundant with others; coefficients become unstable.
- Multicollinearity may not hurt accuracy much, but it hurts interpretation and coefficient stability.

### Practical process (example: salary prediction)

1. Compute VIF for all features.
2. Drop the feature with the highest VIF.
3. Recompute VIF on the remaining features.
4. Repeat until all VIF values are acceptable (commonly < 5 or < 10).
