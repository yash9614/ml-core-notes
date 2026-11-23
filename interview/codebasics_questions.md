# Codebasics ML Interview Questions

Extracted from the codebasics interview question bank. Grouped by difficulty. Use these for practice; detailed answers live in the topic notes.

## Beginner

1. What is Machine Learning?
2. Explain the difference between supervised and unsupervised learning.
3. What is the difference between classification and regression?
4. What are the Machine Learning Project steps?
5. Can you explain the steps involved in the data pre-processing process?
6. Explain the term 'bias' in the context of machine learning models.
7. What is the importance of feature scaling in machine learning?
8. Can you explain the concept of regularisation in machine learning?
9. What is the difference between L1 and L2 regularisation?
10. What is the purpose of a confusion matrix in classification tasks?
11. How do you handle a situation where the data is too imbalanced?
12. What is the purpose of the Support Vector Machine (SVM) algorithm?
13. What is the purpose of the F1 score metric in evaluating classification models?

## Intermediate

1. How do you handle a large volume of data that cannot fit into memory?
2. How do you handle a situation where there are too many features compared to the number of observations?
3. How to find the best fit line in Linear Regression using Gradient Descent?
4. Why is MSE preferred over MAE in Linear Regression?
5. How are nominal and ordinal categorical variables encoded in machine learning, and why are different techniques used for each?
6. Suppose you're building a linear regression model and observe that two of your independent variables are highly correlated with each other. What potential issues could this cause, and how would you address them?
7. Explain the concepts of bias and variance, and how they affect model performance. How would you identify and address underfitting or overfitting?
8. You’re working on a marketing campaign to predict whether a user will click on a promotional email using Logistic Regression. How would you approach this problem and what are the key considerations?
9. You’ve built a classification model for detecting fraud transactions. The dataset is highly imbalanced (1% fraud). Your model reports 99% accuracy. Would you consider this a good model? Why or why not? What alternative metrics would you use?
10. What is a kernel in SVM? When would you use RBF over linear?

## Advanced

1. What happens when we use MSE as the cost function for Logistic Regression (which uses the sigmoid function)?
2. You’ve built a decision tree that perfectly fits training data but performs poorly on unseen data. Why are decision trees prone to overfitting, and how do ensemble methods (Random Forest, Gradient Boosting) help? How does feature importance differ between a single tree and an ensemble?
3. You're working with a small and imbalanced dataset using standard K-Fold. What challenges might this pose, and how can Stratified K-Fold help?
4. You're tuning a Random Forest for loan default prediction. GridSearchCV is too slow. Propose a faster tuning strategy. Discuss trade-offs.
5. You’ve applied K-Means to customer behaviour data with outliers and features on different scales. How would you choose the right number of clusters? What are the limitations of K-Means and how would you address them? How would you evaluate clustering quality?
6. How does DBSCAN overcome the limitations of K-Means?
7. How would you design an efficient and secure API to serve a trained machine learning model? (model loading, input/output handling, validation, security)
8. Why do we need Kubernetes for orchestration?
9. What is a CI/CD pipeline, and how does it facilitate production-grade ML deployment?
10. What is Data Drift and Concept Drift?

## Industrial notes for advanced questions

- **API serving**: Load model once at startup, validate input schema (Pydantic), return probabilities + metadata, add authentication, rate limiting, and logging.
- **Kubernetes**: Useful when you need horizontal scaling, rolling updates, and resource isolation for multiple models or high traffic.
- **CI/CD for ML**: Automated testing of data schemas, model performance thresholds, and deployment pipelines (Jenkins, GitHub Actions, or dedicated ML platforms).
- **Drift**: Data drift = feature distribution changes; Concept drift = relationship between features and target changes. Both require monitoring and retraining.
