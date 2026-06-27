# Wine Quality Prediction - Supervised Machine Learning

[Spanish version](README.es.md)

Portfolio project applying supervised Machine Learning to the **UCI Wine Quality dataset**.

The project focuses on two predictive tasks:

1. **Wine type classification**: predict whether a wine is red or white.
2. **Wine quality regression**: estimate the wine quality score from physicochemical variables.

This repository was reorganized from a Data Science / Machine Learning diploma project into a cleaner GitHub portfolio project.

---

## Project Overview

This project demonstrates a complete supervised Machine Learning workflow:

* Data loading and preparation.
* Exploratory Data Analysis.
* Feature scaling.
* Classification and regression modeling.
* Hyperparameter tuning with `GridSearchCV`.
* Model comparison using objective metrics.
* Critical interpretation of results.

The main goal is not only to train models, but also to compare their behavior and explain the difference between an easier classification task and a more complex regression task.

---

## Dataset

The project uses the public **Wine Quality dataset** from the UCI Machine Learning Repository.

| Dataset    | Samples | Description                                         |
| ---------- | ------: | --------------------------------------------------- |
| Red wine   |   1,599 | Physicochemical properties of red wines             |
| White wine |   4,898 | Physicochemical properties of white wines           |
| Combined   |   6,497 | Full dataset used for classification and regression |

The dataset includes physicochemical variables such as acidity, residual sugar, chlorides, sulfur dioxide, density, pH, sulphates and alcohol.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## Machine Learning Tasks

### 1. Classification - Wine Type

The objective is to predict whether a wine is **red** or **white** based only on its physicochemical characteristics.

Models evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier with `GridSearchCV`
* XGBoost Classifier with `GridSearchCV`

| Model                        |   Accuracy | Weighted F1-score |
| ---------------------------- | ---------: | ----------------: |
| Logistic Regression          |     0.9914 |            0.9914 |
| Decision Tree                |     0.9852 |            0.9853 |
| Random Forest + GridSearchCV |     0.9951 |            0.9951 |
| XGBoost + GridSearchCV       | **0.9975** |        **0.9975** |

**Best model:** XGBoost Classifier.

The classification task achieved near-perfect performance because red and white wines show clearly different physicochemical profiles.

---

### 2. Regression - Wine Quality

The objective is to estimate the quality score assigned to each wine.

Models evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Random Forest Regressor with `GridSearchCV`
* XGBoost Regressor with `GridSearchCV`

| Model                        |        MAE |         R2 |
| ---------------------------- | ---------: | ---------: |
| Linear Regression            |     0.5703 |     0.2589 |
| Decision Tree Regressor      |     0.5961 |     0.2444 |
| Random Forest Regressor      |     0.5799 |     0.2659 |
| Random Forest + GridSearchCV |     0.4977 |     0.4271 |
| XGBoost + GridSearchCV       | **0.4515** | **0.4865** |

**Best model:** XGBoost Regressor.

The regression task was significantly harder than classification. Wine quality is subjective and is not fully explained by physicochemical variables alone.

---

## Key Findings

* Wine type classification reached near-perfect performance.
* XGBoost produced the best results for both classification and regression.
* Regression was more difficult than classification due to the subjective nature of wine quality.
* Hyperparameter tuning improved model performance, especially in the regression task.
* The dataset is useful for comparing supervised learning approaches across classification and regression problems.

---

## Repository Structure

```text
wine-quality-supervised-ml/
├── .github/workflows/
│   └── python-check.yml
├── data/
│   └── README.md
├── docs/
│   └── project_summary.md
├── notebooks/
│   └── 01_wine_quality_supervised_ml.ipynb
├── src/
│   ├── data_loader.py
│   └── train_supervised.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/FLMontes/wine-quality-supervised-ml.git
cd wine-quality-supervised-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train_supervised.py
```

To explore the full analysis, open the notebook:

```bash
jupyter notebook notebooks/01_wine_quality_supervised_ml.ipynb
```

---

## Project Status

This project is currently being improved as a portfolio repository.

Planned improvements:

* Clean and reorganize the notebook structure.
* Add visual model comparison charts.
* Add confusion matrix and regression error plots.
* Improve documentation with business-oriented insights.
* Add saved figures to `reports/figures/`.

---

## Author

**Franco Leonel Montes**
Computer Engineering student - Universidad Nacional de Córdoba
Data Science & Machine Learning diploma graduate

* GitHub: [FLMontes](https://github.com/FLMontes)
* LinkedIn: [franco-leonel-montes](https://www.linkedin.com/in/franco-leonel-montes/)
