# ML core notes

Study notes for the algorithms in [yash9614/Nrish_Karik_SataDience/ML](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/ML).

That folder is the raw course dump: PDFs, notebooks, and two end-to-end apps. This repo is the readable layer. Code notebooks will be added later. Nothing here copies the slide decks.

## How to use this

1. Read the note for one algorithm.
2. Glance at the highlighted formulas in [formulae.md](formulae.md).
3. Answer the interview questions out loud before looking at the short answers.
4. Read the coding concepts in [notes/13-coding-concepts.md](notes/13-coding-concepts.md) and implement the primitives in [interview/coding_questions.md](interview/coding_questions.md).
5. Do the linked Kaggle problem. The point is the split, metric, and error analysis, not the leaderboard.
6. Only then open the matching notebook in the course dump.

## Map

| Topic | Note | Course files |
|---|---|---|
| Core formulas (highlighted) | [formulae.md](formulae.md) | all notes |
| Types of ML, instance vs model | [notes/01-ml-landscape.md](notes/01-ml-landscape.md) | `2-Typesof+ML+technqiues.pdf`, `instance+based+vs+model+absed+learning.pdf` |
| Simple and multiple linear regression | [notes/02-linear-regression.md](notes/02-linear-regression.md) | `1-Simple+Linear+Regression.pdf`, `2-Multiple+Linear+Regression.pdf`, `height-weight.csv`, `economic_index.csv` |
| Polynomial, Ridge, Lasso, Elastic Net | [notes/03-regularization.md](notes/03-regularization.md) | `9-Polynomialregression.pdf`, `Ridge,Lasso+And+Elasticnet.pdf`, Algerian forest-fires practical |
| Logistic regression | [notes/04-logistic-regression.md](notes/04-logistic-regression.md) | `5-Logistic+Regression.pdf`, `Logistic+Regression+Implementation.ipynb` |
| KNN | [notes/05-knn.md](notes/05-knn.md) | `1,0-KNN+Classification+And+Regression.pdf`, `2.0-KD+Tree+Ball+Tree+.pdf` |
| Naive Bayes | [notes/06-naive-bayes.md](notes/06-naive-bayes.md) | `1.0-Naive+Bayes+Classifier+Indepth+Intuition.pdf`, `2.0-Variants+of+Naive+Bayes+.pdf` |
| SVM | [notes/07-svm.md](notes/07-svm.md) | `6-Support+Vector+Classifier.pdf`, `6.1-SVR.pdf`, `6.2-SVM+Kernels.pdf` |
| Decision trees | [notes/08-decision-trees.md](notes/08-decision-trees.md) | classifier and regressor PDFs, diabetes regressor notebook |
| Random forest, AdaBoost, GBM, XGBoost | [notes/09-ensembles.md](notes/09-ensembles.md) | AdaBoost, gradient boosting, XGBoost PDFs and notebooks |
| Class imbalance handling | [notes/class-imbalance.md](notes/class-imbalance.md) | breast-cancer exercise (SMOTE, undersampling) |
| K-means, hierarchical, DBSCAN | [notes/10-clustering.md](notes/10-clustering.md) | `Kmeans+clustering.pdf`, `Hierarichal+Clustering.pdf`, `DBCAN.pdf` |
| Anomaly detection and PCA | [notes/11-anomaly-and-pca.md](notes/11-anomaly-and-pca.md) | Isolation Forest, LOF, `PCA.pdf` |
| Metrics, bias/variance, cross-validation | [notes/12-metrics-and-validation.md](notes/12-metrics-and-validation.md) | performance-metrics PDFs, `Types+Of+Cross+Validation.pdf` |
| Coding concepts (stable softmax, BCE, cosine, LoRA, …) | [notes/13-coding-concepts.md](notes/13-coding-concepts.md) | from coding questions |
| Interview bank (conceptual) | [interview/questions.md](interview/questions.md) | all of the above |
| Interview bank (coding) | [interview/coding_questions.md](interview/coding_questions.md) | top-k cosine, stable softmax, BCE, top-k accuracy, Gini stump, LoRA, logistic GD, confusion matrix, cross-entropy |
| Codebasics interview questions | [interview/codebasics_questions.md](interview/codebasics_questions.md) | beginner / intermediate / advanced |
| Kaggle problems | [kaggle/must-try.md](kaggle/must-try.md) | practice only |

## Codebasics.io Takeaways

Clean, structured study notes extracted from the codebasics ML takeaway PDFs. These complement the main algorithm notes above.

| Topic | Note |
|---|---|
| ML Project Lifecycle | [notes/codebasics/ml-project-lifecycle.md](notes/codebasics/ml-project-lifecycle.md) |
| Feature Engineering | [notes/codebasics/feature-engineering.md](notes/codebasics/feature-engineering.md) |
| Supervised Regression | [notes/codebasics/supervised-regression-takeaways.md](notes/codebasics/supervised-regression-takeaways.md) |
| Supervised Classification | [notes/codebasics/supervised-classification-takeaways.md](notes/codebasics/supervised-classification-takeaways.md) |
| Ensemble Learning | [notes/codebasics/ensemble-learning-takeaways.md](notes/codebasics/ensemble-learning-takeaways.md) |
| Unsupervised Learning | [notes/codebasics/unsupervised-learning-takeaways.md](notes/codebasics/unsupervised-learning-takeaways.md) |
| Model Evaluation & Fine-Tuning | [notes/codebasics/model-evaluation-fine-tuning.md](notes/codebasics/model-evaluation-fine-tuning.md) |

## Projects already in the course dump

These stay in the old repo until we port code.

- `mlproject-main`: student exam-score regression, Flask app, CatBoost.
- `NETWORKsecurity`: phishing classifier, FastAPI, MongoDB, Docker.
- `Ridge Lassso Elastic Regression Practicals`: Algerian forest-fires dataset.

## Suggested order

Metrics first if you already know the model names. Otherwise: linear regression, logistic regression, decision trees, ensembles, then KNN, Naive Bayes, SVM, clustering, anomaly detection, PCA. Finish with the coding concepts and implementations.
