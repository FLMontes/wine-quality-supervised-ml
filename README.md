# Wine Quality Prediction — Supervised Machine Learning

Portfolio project applying supervised Machine Learning to the **UCI Wine Quality dataset**.

The project solves two tasks:

1. **Wine type classification**: predict whether a wine is red or white.
2. **Wine quality regression**: estimate the quality score from physicochemical variables.

This repository was reorganized from a Data Science / Machine Learning diploma project into a cleaner GitHub portfolio project.

---

## Why this project matters

This project shows a complete supervised ML workflow:

- Data loading and preparation.
- Exploratory Data Analysis.
- Feature scaling.
- Classification and regression modeling.
- Hyperparameter tuning with `GridSearchCV`.
- Model comparison using objective metrics.
- Critical interpretation of results.

---

## Dataset

The project uses the public Wine Quality dataset from the UCI Machine Learning Repository.

| Dataset | Samples | Description |
|---|---:|---|
| Red wine | 1,599 | Physicochemical properties of red wines |
| White wine | 4,898 | Physicochemical properties of white wines |
| Combined | 6,497 | Full dataset used for classification and regression |

Features include acidity, residual sugar, chlorides, sulfur dioxide, density, pH, sulphates and alcohol.

---

## Models evaluated

### Classification — Wine Type

| Model | Accuracy | Weighted F1-score |
|---|---:|---:|
| Logistic Regression | 0.9914 | 0.9914 |
| Decision Tree | 0.9852 | 0.9853 |
| Random Forest + GridSearchCV | 0.9951 | 0.9951 |
| XGBoost + GridSearchCV | **0.9975** | **0.9975** |

### Regression — Wine Quality

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | 0.5703 | 0.2589 |
| Decision Tree Regressor | 0.5961 | 0.2444 |
| Random Forest Regressor | 0.5799 | 0.2659 |
| Random Forest + GridSearchCV | 0.4977 | 0.4271 |
| XGBoost + GridSearchCV | **0.4515** | **0.4865** |

---

## Key conclusions

- Wine type classification achieved near-perfect performance.
- XGBoost produced the best results for both classification and regression.
- Regression was significantly harder than classification.
- Wine quality is subjective and not fully explained by physicochemical variables alone.
- Hyperparameter tuning improved model performance, especially in regression.

---

## Repository structure

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
├── reports/
│   └── figures/
├── src/
│   ├── data_loader.py
│   └── train_supervised.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## How to run

```bash
git clone https://github.com/FLMontes/wine-quality-supervised-ml.git
cd wine-quality-supervised-ml
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
python src/train_supervised.py
```

To explore the full analysis:

```bash
jupyter notebook notebooks/01_wine_quality_supervised_ml.ipynb
```

---

## Author

**Franco Leonel Montes**  
Computer Engineering student — Universidad Nacional de Córdoba  
Data Science & Machine Learning diploma graduate

- GitHub: [FLMontes](https://github.com/FLMontes)
- LinkedIn: [franco-leonel-montes](https://www.linkedin.com/in/franco-leonel-montes/)
