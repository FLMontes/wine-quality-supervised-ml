# Project Summary - Wine Quality Supervised ML

This project applies supervised Machine Learning to the **UCI Wine Quality dataset** to solve two predictive tasks: wine type classification and wine quality regression.

The goal is to demonstrate a complete supervised learning workflow, including data preparation, exploratory analysis, preprocessing, model training, hyperparameter tuning, model comparison and interpretation of results.

---

## Objectives

1. Predict whether a wine is **red** or **white** using physicochemical variables.
2. Estimate the wine **quality score** using regression models.
3. Compare baseline models against optimized ensemble models.
4. Evaluate model performance using appropriate classification and regression metrics.
5. Interpret why classification performs better than regression in this dataset.

---

## Dataset

The project uses the public **Wine Quality dataset** from the UCI Machine Learning Repository.

The combined dataset contains:

* **1,599** red wine samples.
* **4,898** white wine samples.
* **6,497** total samples.
* **11** physicochemical input features.

Main features include acidity, residual sugar, chlorides, sulfur dioxide, density, pH, sulphates and alcohol.

---

## Machine Learning Workflow

The project follows this workflow:

1. Load and combine red and white wine datasets.
2. Perform exploratory data analysis.
3. Separate features and target variables.
4. Apply feature scaling with `StandardScaler`.
5. Train classification models for wine type prediction.
6. Train regression models for wine quality estimation.
7. Optimize selected models using `GridSearchCV`.
8. Compare results using objective metrics.
9. Interpret the difference between classification and regression performance.

---

## Models Evaluated

### Classification Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier with `GridSearchCV`
* XGBoost Classifier with `GridSearchCV`

### Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Random Forest Regressor with `GridSearchCV`
* XGBoost Regressor with `GridSearchCV`

---

## Main Results

### Classification - Wine Type

The best classification model was an optimized **XGBoost Classifier**.

| Model                        |   Accuracy | Weighted F1-score |
| ---------------------------- | ---------: | ----------------: |
| Logistic Regression          |     0.9914 |            0.9914 |
| Decision Tree                |     0.9852 |            0.9853 |
| Random Forest + GridSearchCV |     0.9951 |            0.9951 |
| XGBoost + GridSearchCV       | **0.9975** |        **0.9975** |

The classification task achieved near-perfect performance because red and white wines have clearly different physicochemical profiles.

---

### Regression - Wine Quality

The best regression model was an optimized **XGBoost Regressor**.

| Model                        |        MAE |         R2 |
| ---------------------------- | ---------: | ---------: |
| Linear Regression            |     0.5703 |     0.2589 |
| Decision Tree Regressor      |     0.5961 |     0.2444 |
| Random Forest Regressor      |     0.5799 |     0.2659 |
| Random Forest + GridSearchCV |     0.4977 |     0.4271 |
| XGBoost + GridSearchCV       | **0.4515** | **0.4865** |

The regression task was more difficult than classification because wine quality is subjective, ordinal and influenced by factors not fully captured by physicochemical variables alone.

---

## Key Interpretation

The results show a clear difference between the two supervised learning tasks.

Wine type classification is highly reliable because the input variables contain strong signals that separate red and white wines.

Wine quality prediction is harder because the quality score is a subjective evaluation. Even with optimized ensemble models, the available physicochemical variables only explain part of the variability in quality.

This makes the project useful for comparing a highly separable classification problem against a more complex regression problem using the same dataset.

---

## Portfolio Value

This project demonstrates practical skills in:

* Python data analysis.
* Exploratory Data Analysis.
* Supervised Machine Learning.
* Classification and regression modeling.
* Hyperparameter tuning.
* Model evaluation.
* Critical interpretation of results.
* GitHub project organization and documentation.
