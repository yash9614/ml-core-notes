# ML Project Lifecycle

Source: codebasics.io Takeaways – ML Project Life Cycle.  
**Industrial relevance added:** real-world practices, common pitfalls, and what actually matters in production.

A typical ML project has ~10 stages. In industry the hard parts are rarely the model itself—they are data quality, stakeholder alignment, evaluation design, and monitoring after deployment.

## 10 Stages (high-level)

1. Requirements gathering and Scope of Work (SOW)
2. Data collection
3. Data preparation & EDA
4. Feature engineering
5. Model selection & training
6. Model evaluation
7. Model fine-tuning
8. Deployment
9. Monitoring
10. Feedback loop / retraining

**Industry tip:** Most projects fail in stages 1–3 or 9–10, not in modeling.

## Requirements and SOW

- SOW mixes business requirements with the technology approach. It is the reference for all stakeholders.
- Early estimates are rough because requirements are vague; they improve over time.
- Tools: JIRA (very common), Confluence, Notion, Asana, Excel.

**Industrial relevance:**  
- Explicitly define the business metric (e.g., “reduce false positives by 20%”, “improve conversion by 3%”) before any modeling.  
- Document data ownership, privacy constraints (GDPR/CCPA), latency requirements, and acceptable error rates.  
- Get sign-off from business stakeholders on the evaluation criteria early—otherwise the project will be judged on the wrong metric later.

## Data Collection

- Mostly internal organizational data.
- Sometimes purchased from external vendors.
- Web scraping: Beautiful Soup (Python) or tools like BrightData.

**Industrial relevance:**  
- Data quality issues (missing, delayed, inconsistent, biased) kill more projects than algorithm choice.  
- Always check for data leakage (future information leaking into features) and label leakage.  
- Prefer feature stores or well-governed data pipelines over one-off CSVs.

## Data Preparation & EDA

- Cleaning: missing values, duplicates, outliers.
- Transforming: encoding categoricals, scaling/normalizing.
- EDA: understand quality, distributions, relationships relevant to the model.
- Jupyter for quick analysis and plots.
- Apache Spark for distributed computation on large data.

**Industrial relevance:**  
- EDA should answer: Is the data representative of production? Are there segments where the model will fail? What is the label quality?  
- Document every assumption and transformation so it can be reproduced in the production pipeline.

## Feature Engineering

- Creating additional features from existing data to help the model predict better. (See dedicated note.)

## Model Selection & Training

- Choice depends on the use case.
- Naive Bayes: small datasets, especially text.
- Decision trees: problems that involve sequential decisions.
- Random Forest: complex datasets, good baseline.
- XGBoost / LightGBM / CatBoost: when predictive accuracy is critical and you have tabular data.
- Always evaluate on a held-out test set (ideally time-based or user-based split if applicable).

**Industrial relevance:**  
- Start with a simple baseline (linear model or even a business rule).  
- Gradient boosting is the default strong baseline for tabular data in industry.  
- Track experiments properly (MLflow, Weights & Biases, or at least a spreadsheet).

## Model Fine Tuning

- Hyperparameter tuning is essential.
- GridSearchCV / RandomizedSearchCV (sklearn) or Optuna / Hyperopt.
- Transfer learning can speed up training by starting from a pre-trained model (more common in deep learning).

## Monitoring and MLOps Feedback

- Development → staging/testing → production.
- Once in production, monitor performance and collect feedback.
- Feedback comes from both humans and automated signals.
- Retrain regularly with new data to combat data drift.

**Industrial relevance:**  
- Monitor data drift (feature distributions) and concept drift (relationship between features and target).  
- Set up alerts on key business metrics, not just model metrics.  
- Have a rollback plan and shadow/canary deployment strategy.  
- Common failure modes: silent data pipeline breaks, training-serving skew, and unmonitored drift.

## Interview Questions

1. Walk me through the end-to-end lifecycle of an ML project you have worked on. What was the biggest bottleneck?
2. How do you prevent data leakage when building a model?
3. What is the difference between data drift and concept drift? How would you detect each in production?
4. Why do most ML projects fail after deployment rather than during modeling?
5. How would you design the evaluation strategy for a model that predicts customer churn?
6. What tools and practices do you use for experiment tracking and model versioning?
