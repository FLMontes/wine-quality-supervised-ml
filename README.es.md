# Predicción de Calidad de Vinos - Machine Learning Supervisado

Proyecto de portfolio aplicando Machine Learning supervisado al **dataset Wine Quality de UCI**.

El proyecto se enfoca en dos tareas predictivas:

1. **Clasificación del tipo de vino**: predecir si un vino es tinto o blanco.
2. **Regresión de calidad del vino**: estimar el puntaje de calidad a partir de variables fisicoquímicas.

Este repositorio fue reorganizado a partir de un proyecto de Diplomatura en Data Science / Machine Learning para convertirlo en un proyecto más limpio y profesional para GitHub.

---

## Descripción del proyecto

Este proyecto demuestra un flujo completo de trabajo de Machine Learning supervisado:

* Carga y preparación de datos.
* Análisis exploratorio de datos.
* Escalado de variables.
* Modelado de clasificación y regresión.
* Optimización de hiperparámetros con `GridSearchCV`.
* Comparación de modelos usando métricas objetivas.
* Interpretación crítica de resultados.

El objetivo principal no es solamente entrenar modelos, sino también comparar su comportamiento y explicar la diferencia entre una tarea de clasificación más simple y una tarea de regresión más compleja.

---

## Dataset

El proyecto utiliza el dataset público **Wine Quality** del repositorio UCI Machine Learning Repository.

| Dataset     | Muestras | Descripción                                               |
| ----------- | -------: | --------------------------------------------------------- |
| Vino tinto  |    1.599 | Propiedades fisicoquímicas de vinos tintos                |
| Vino blanco |    4.898 | Propiedades fisicoquímicas de vinos blancos               |
| Combinado   |    6.497 | Dataset completo utilizado para clasificación y regresión |

El dataset incluye variables fisicoquímicas como acidez, azúcar residual, cloruros, dióxido de azufre, densidad, pH, sulfatos y alcohol.

---

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## Tareas de Machine Learning

### 1. Clasificación - Tipo de vino

El objetivo es predecir si un vino es **tinto** o **blanco** usando solamente sus características fisicoquímicas.

Modelos evaluados:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier con `GridSearchCV`
* XGBoost Classifier con `GridSearchCV`

| Modelo                       |   Accuracy | F1-score ponderado |
| ---------------------------- | ---------: | -----------------: |
| Logistic Regression          |     0.9914 |             0.9914 |
| Decision Tree                |     0.9852 |             0.9853 |
| Random Forest + GridSearchCV |     0.9951 |             0.9951 |
| XGBoost + GridSearchCV       | **0.9975** |         **0.9975** |

**Mejor modelo:** XGBoost Classifier.

La tarea de clasificación alcanzó un rendimiento casi perfecto porque los vinos tintos y blancos presentan perfiles fisicoquímicos claramente diferenciados.

---

### 2. Regresión - Calidad del vino

El objetivo es estimar el puntaje de calidad asignado a cada vino.

Modelos evaluados:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Random Forest Regressor con `GridSearchCV`
* XGBoost Regressor con `GridSearchCV`

| Modelo                       |        MAE |         R2 |
| ---------------------------- | ---------: | ---------: |
| Linear Regression            |     0.5703 |     0.2589 |
| Decision Tree Regressor      |     0.5961 |     0.2444 |
| Random Forest Regressor      |     0.5799 |     0.2659 |
| Random Forest + GridSearchCV |     0.4977 |     0.4271 |
| XGBoost + GridSearchCV       | **0.4515** | **0.4865** |

**Mejor modelo:** XGBoost Regressor.

La tarea de regresión fue significativamente más difícil que la clasificación. La calidad del vino es una variable subjetiva y no queda completamente explicada solo por variables fisicoquímicas.

---

## Hallazgos principales

* La clasificación del tipo de vino alcanzó un rendimiento casi perfecto.
* XGBoost obtuvo los mejores resultados tanto en clasificación como en regresión.
* La regresión fue más difícil que la clasificación por la naturaleza subjetiva de la calidad del vino.
* La optimización de hiperparámetros mejoró el rendimiento de los modelos, especialmente en la tarea de regresión.
* El dataset es útil para comparar enfoques de aprendizaje supervisado en problemas de clasificación y regresión.

---

## Estructura del repositorio

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
├── README.es.md
└── requirements.txt
```

---

## Cómo ejecutar el proyecto

Clonar el repositorio:

```bash
git clone https://github.com/FLMontes/wine-quality-supervised-ml.git
cd wine-quality-supervised-ml
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Windows:

```bash
.venv\Scripts\activate
```

Activarlo en Linux / macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el script de entrenamiento:

```bash
python src/train_supervised.py
```
> Nota: Los resultados oficiales reportados en este proyecto son los documentados en el notebook.  
> El script `src/train_supervised.py` se incluye solo como un pipeline liviano para verificar la configuración del proyecto.
Para explorar el análisis completo, abrir el notebook:

```bash
jupyter notebook notebooks/01_wine_quality_supervised_ml.ipynb
```

---

## Estado del proyecto

Este proyecto se encuentra actualmente en proceso de mejora como repositorio de portfolio.

Mejoras planificadas:

* Limpiar y reorganizar la estructura del notebook.
* Agregar gráficos comparativos de modelos.
* Agregar matriz de confusión y gráficos de error de regresión.
* Mejorar la documentación con conclusiones orientadas al negocio.
* Guardar figuras en `reports/figures/`.

---

## Autor

**Franco Leonel Montes**
Estudiante de Ingeniería en Computación - Universidad Nacional de Córdoba
Diplomado en Data Science & Machine Learning

* GitHub: [FLMontes](https://github.com/FLMontes)
* LinkedIn: [franco-leonel-montes](https://www.linkedin.com/in/franco-leonel-montes/)
