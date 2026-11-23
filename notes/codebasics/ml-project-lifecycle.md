# ML Project Lifecycle

Source: codebasics.io Takeaways – ML Project Life Cycle.

A typical ML project has about 10 stages, from requirements to monitoring. Think of it like making a pizza: gather ingredients (data), prepare, cook (train), taste-test (evaluate), serve (deploy), and improve based on feedback.

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

## Requirements and SOW

- SOW mixes business requirements with the technology approach. It is the reference for all stakeholders.
- Early estimates are rough because requirements are vague; they improve over time.
- Tools: JIRA (very common), Confluence, Notion, Asana, Excel.

## Data Collection

- Mostly internal organizational data.
- Sometimes purchased from external vendors.
- Web scraping: Beautiful Soup (Python) or tools like BrightData.

## Data Preparation & EDA

- Cleaning: missing values, duplicates, outliers.
- Transforming: encoding categoricals, scaling/normalizing.
- EDA: understand quality, distributions, relationships relevant to the model.
- Jupyter for quick analysis and plots.
- Apache Spark for distributed computation on large data (feels local to the user).

## Feature Engineering

- Creating additional features from existing data to help the model predict better. (See dedicated note.)

## Model Selection & Training

- Choice depends on the use case.
- Naive Bayes: small datasets, especially text.
- Decision trees: problems that involve sequential decisions (e.g., salary rules).
- Random Forest: complex datasets.
- XGBoost: when predictive accuracy is critical.
- Always evaluate on a held-out test set by comparing predictions to true values.

## Model Fine Tuning

- Hyperparameter tuning is essential.
- GridSearchCV (sklearn) systematically searches the parameter grid with cross-validation.
- Transfer learning can speed up training by starting from a pre-trained model.

## Monitoring and MLOps Feedback

- Development → staging/testing → production.
- Once in production, monitor performance and collect feedback.
- Feedback comes from both humans and automated signals.
- Retrain regularly with new data to combat data drift.
