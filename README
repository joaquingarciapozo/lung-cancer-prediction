# Lung Disease Prediction Using Machine Learning

A machine learning project focused on **predicting pulmonary disease using clinical, demographic, and environmental exposure variables**. Several classification algorithms are evaluated and compared, with a stacking ensemble selected to combine the predictions of multiple classifiers.

**Author:** Joaquín García Pozo

## 📋 Project Overview

The objective is to develop and evaluate classification models capable of predicting the presence of pulmonary disease (`PULMONARY_DISEASE`) based on the patient characteristics available in the dataset.

The project follows a data science workflow that includes:

- Exploratory Data Analysis (EDA).
- Univariate and bivariate analysis.
- Data quality assessment.
- Association analysis between categorical variables using Cramér's V.
- Data preprocessing and feature transformation.
- Model training and hyperparameter optimization using cross-validation.
- Performance comparison using classification metrics.
- Selection and evaluation of an ensemble classification model.

## 📂 Project Structure

```text
LungCancer/
├── LungCancerFINAL_JGP.ipynb
├── lung-cancer-dataset.csv
└── README.md
```

- `LungCancerFINAL_JGP.ipynb`: Main notebook containing exploratory data analysis, model training, and performance evaluation.
- `lung-cancer-dataset.csv`: Dataset used by the notebook. It must be downloaded separately.
- `README.md`: Project documentation.

## 📊 Dataset

The project uses the following Kaggle dataset:

[Lung Cancer Prediction Dataset](https://www.kaggle.com/datasets/shantanugarg274/lung-cancer-prediction-dataset)

Download the dataset and place `lung-cancer-dataset.csv` in the working directory from which the notebook is executed.

### Features

The dataset includes variables such as:

| Feature | Description |
|---|---|
| `AGE` | Patient age. |
| `GENDER` | Recorded gender. |
| `SMOKING` | Smoking-related indicator. |
| `FINGER_DISCOLORATION` | Finger discoloration. |
| `MENTAL_STRESS` | Mental stress indicator. |
| `EXPOSURE_TO_POLLUTION` | Exposure to pollution. |
| `LONG_TERM_ILLNESS` | Long-term illness indicator. |
| `IMMUNE_WEAKNESS` | Immune weakness indicator. |
| `ENERGY_LEVEL` | Energy level. |
| `BREATHING_ISSUE` | Breathing difficulties. |
| `ALCOHOL_CONSUMPTION` | Alcohol consumption indicator. |
| `THROAT_DISCOMFORT` | Throat discomfort. |
| `CHEST_TIGHTNESS` | Chest tightness. |
| `FAMILY_HISTORY` | Family history of disease. |
| `SMOKING_FAMILY_HISTORY` | Family history related to smoking. |
| `STRESS_IMMUNE` | Indicator associated with stress and the immune system. |
| `OXYGEN_SATURATION` | Oxygen saturation. |
| `PULMONARY_DISEASE` | Target variable indicating the presence (`YES`) or absence (`NO`) of pulmonary disease. |

The target variable is encoded as a binary classification label:

- `YES` → `1`
- `NO` → `0`

The initial analysis reports an approximate target distribution of 40.74% positive cases and 59.26% negative cases.

## 🛠️ Technologies and Dependencies

The project is implemented in Python and uses the following libraries:

- **NumPy** and **Pandas**: Data manipulation and analysis.
- **Matplotlib** and **Seaborn**: Data visualization.
- **SciPy**: Statistical analysis.
- **Scikit-learn**: Data preprocessing, model training, cross-validation, and evaluation.
- **XGBoost**: Gradient boosting classification.
- **CatBoost**: Boosting with support for categorical features.
- **Tabulate**: Tabular result presentation.
- **Jupyter Notebook**: Interactive execution of the analysis.

### Installation

Create a virtual environment to isolate the project dependencies.

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required libraries:

```bash
python -m pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost catboost tabulate jupyter
```

The original notebook does not specify pinned dependency versions. For strict reproducibility, record the versions used in a `requirements.txt` file.

## 🚀 Getting Started

### 1. Download the dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shantanugarg274/lung-cancer-prediction-dataset) and place `lung-cancer-dataset.csv` in the project directory.

### 2. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 3. Run the notebook

Open `LungCancerFINAL_JGP.ipynb` and execute the cells in order, from data loading through final model evaluation.

The notebook loads the dataset using:

```python
df = pd.read_csv("lung-cancer-dataset.csv")
```

Ensure that the CSV file is available at the expected relative path.

## 🔬 Methodology

### 1. Exploratory Data Analysis

The dataset is examined to understand feature distributions, descriptive statistics, unique values, missing values, and duplicate records.

Histograms, box plots, frequency charts, and heatmaps are used to explore the dataset.

Associations between categorical variables are analyzed using Cramér's V coefficient.

### 2. Data Preprocessing

The input features (`X`) and target variable (`y`) are separated.

The dataset is split into training and test sets using `train_test_split`, with 80% of the samples allocated to training and 20% to testing. The split uses `random_state=42`.

Preprocessing includes:

- **Standardization:** `StandardScaler` for numerical features.
- **Categorical encoding:** `OneHotEncoder(drop='first')` for categorical features in the corresponding pipelines.
- **Target encoding:** Conversion of `YES` and `NO` into binary values.

Preprocessing transformations are integrated into pipelines for the models that use them.

### 3. Classification Models

Eight classification approaches are evaluated:

1. Logistic Regression (`LogisticRegression`).
2. K-Nearest Neighbors (`KNeighborsClassifier`).
3. Decision Tree (`DecisionTreeClassifier`).
4. Support Vector Machine (`SVC`).
5. Random Forest (`RandomForestClassifier`).
6. XGBoost (`XGBClassifier`).
7. CatBoost (`CatBoostClassifier`).
8. Stacking (`StackingClassifier`).

Hyperparameter optimization is performed using `GridSearchCV` with five-fold cross-validation (`cv=5`) and the F1-score as the model selection criterion.

### 4. Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Proportion of correctly predicted positive cases among all predicted positive cases.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to discriminate between the two classes.

F1-score is the primary metric used during hyperparameter optimization. Confusion matrices and ROC curves are also analyzed.

## 📈 Results

The following results were obtained on the test set, according to the results table generated by the notebook:

| Model | Accuracy | F1-score | Recall | Precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.911 | 0.8983 | 0.9161 | 0.8812 | 0.9373 |
| KNN | 0.868 | 0.8537 | 0.8974 | 0.8140 | 0.9262 |
| Decision Tree | 0.882 | 0.8641 | 0.8741 | 0.8542 | 0.8723 |
| SVM | 0.902 | 0.8858 | 0.8858 | 0.8858 | 0.9347 |
| Random Forest | 0.917 | 0.9029 | 0.8998 | 0.9061 | 0.9333 |
| XGBoost | 0.914 | 0.8991 | 0.8928 | 0.9054 | 0.9324 |
| CatBoost | 0.915 | 0.9006 | 0.8974 | 0.9038 | 0.9371 |
| **Stacking** | **0.918** | **0.9042** | **0.9021** | **0.9063** | **0.9362** |

Values are rounded for presentation.

### Selected Model: StackingClassifier

The selected model combines four base classifiers:

- Logistic Regression.
- Random Forest.
- XGBoost.
- Support Vector Classifier.

Their predictions are combined using a final Logistic Regression estimator, which learns how to integrate the outputs of the base models.

In this experiment, the stacking model achieves:

- **Accuracy:** 0.918
- **F1-score:** 0.9042
- **Recall:** 0.9021
- **Precision:** 0.9063
- **ROC-AUC:** 0.9362

Among the eight evaluated approaches, Stacking achieves the highest reported accuracy and F1-score on the test set.

## ⚠️ Limitations and Considerations

This project is intended for academic and experimental purposes. The reported results do not establish that the model is suitable for clinical use.

Before considering any real-world application, further work would be required to:

- Validate the model on independent external datasets.
- Assess potential biases and performance differences across patient subgroups.
- Evaluate probability calibration and appropriate decision thresholds.
- Analyze the clinical consequences of false positives and false negatives.
- Document the provenance, representativeness, and quality of the dataset.
- Establish clinical oversight and performance monitoring procedures.

A high recall alone does not guarantee that a detection system is safe. Model selection should also account for the required clinical sensitivity, specificity, and the consequences of classification errors.

**This project is not a medical device and must not be used to diagnose, rule out, or treat pulmonary disease.**

## 📚 References

- [Lung Cancer Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/shantanugarg274/lung-cancer-prediction-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/en/docs)

## 📄 License

No license has been specified for this project. If the repository is published publicly, an explicit license should be selected after reviewing the usage and redistribution conditions of the original dataset.
