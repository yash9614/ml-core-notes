# Kaggle problems worth doing

One problem per idea. Finish a dumb baseline before a tuned model. Write down the metric and the leak you almost committed.

| Concept | Problem | Why this one | Done when |
|---|---|---|---|
| Linear regression | [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | Classic tabular regression, RMSLE | Linear model with a residual plot, before any tree |
| Linear regression, small | [Medical Cost Personal](https://www.kaggle.com/datasets/mirichoi0218/insurance) | Coefficients you can explain | You can say what one year of age does to predicted cost |
| Logistic regression | [Titanic](https://www.kaggle.com/competitions/titanic) | Small, leaked-feature traps, a public baseline | Logistic beats "predict the majority class" |
| Imbalanced classification | [Porto Seguro](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction) | Rare events, Gini metric | You stop reporting accuracy |
| Decision tree | [Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) | A readable tree on a health table | Depth-3 tree drawn, recall checked |
| Tree regressor | [Diabetes dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) or the course diabetes notebook | Compare tree steps vs a line | RMSE of both models written down |
| Random forest / boosting | House Prices again, then [Tabular Playground](https://www.kaggle.com/competitions) from any recent season | Same data, stronger model | Forest beats the linear baseline, and you know why |
| XGBoost | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | Real fraud table, heavy imbalance | A single boosted model with early stopping |
| KNN | [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) | Distance-based, scaling matters | k chosen on a validation curve |
| Naive Bayes | [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) | Text counts, independence assumption | Multinomial NB baseline before any embedding |
| SVM | Digit Recognizer again | Linear SVM vs RBF on pixels | You can say which kernel won and why |
| K-means | [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | Elbow and silhouette on two spending features | k chosen with silhouette, not by eye alone |
| DBSCAN | Same mall data, or the course DBSCAN notebook | Density vs centroid | Noise points listed, eps justified |
| PCA | Digit Recognizer | Variance vs reconstruction | 95% variance cutoff, then a classifier on the components |
| Anomaly detection | [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 0.17% positive, PCA features already | Isolation Forest or LOF compared with logistic regression on PR-AUC |

Course datasets if you do not want an account yet: `height-weight.csv`, `economic_index.csv`, `cardekho_imputated.csv`, `Travel.csv`, Algerian forest fires, and the phishing tables in `NETWORKsecurity`.

Skip leaderboard chasing. A written error analysis is the deliverable.
