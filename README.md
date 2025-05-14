# 🫀 Heart Disease Classification with Feature Selection and Model Optimization

---
## Introduction

Heart disease is one of the leading causes of death worldwide. It occurs when blood is unable to flow properly to the heart (Yadav, D.C., et al., 2020). According to the World Health Organization (WHO), approximately 17.9 million people die from heart disease each year. Therefore, early detection during initial examinations is crucial to prevent more serious complications. In a previous study by Yadav, D.C., et al., heart disease prediction was performed using feature selection with Pearson correlation and classification using the Random Forest method. The selected features were cp, exang, and oldpeak.

This project focuses on predicting heart disease using machine learning classification techniques. Feature selection is done using Variance Inflation Factor (VIF), and models are evaluated through k-Fold cross-validation and multiple performance metrics.

---

## 📌 Problems

1. What are the selected features in the heart disease dataset using the Variance Inflation Factor?
2. What are the hyperparameter optimizations using the Random Forest, Multilayer Perceptron, and XGBoost Classifier methods?
3. How is k-fold cross validation applied to the heart disease dataset?
4. What are the evaluation metric results (accuracy, recall, precision, and F1 score) for the three classification methods used?

## 📌 Objectives
1. To identify the selected features in the heart disease dataset using the Variance Inflation Factor.
2. To determine the hyperparameter optimization for the Random Forest, Multilayer Perceptron, and XGBoost Classifier methods.
3. To understand the implementation of k-fold cross validation on the heart disease dataset.
4. To analyze the evaluation metrics (accuracy, recall, precision, and F1 score) of the three classification methods used.

---

## 📂 Dataset

- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- The dataset includes 14 attributes such as:

| Feature    | Description                                                                                                                                              |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `age`      | Age of the patient (in years)                                                                                                                            |
| `sex`      | Sex of the patient (1 = male, 0 = female)                                                                                                                |
| `cp`       | Chest pain type (0–3):<br>0 = typical angina<br>1 = atypical angina<br>2 = non-anginal pain<br>3 = asymptomatic                                          |
| `trestbps` | Resting blood pressure (in mm Hg) on admission to the hospital                                                                                           |
| `chol`     | Serum cholesterol in mg/dl                                                                                                                               |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)                                                                                                    |
| `restecg`  | Resting electrocardiographic results:<br>0 = normal<br>1 = having ST-T wave abnormality<br>2 = showing probable or definite left ventricular hypertrophy |
| `thalach`  | Maximum heart rate achieved                                                                                                                              |
| `exang`    | Exercise-induced angina (1 = yes; 0 = no)                                                                                                                |
| `oldpeak`  | ST depression induced by exercise relative to rest                                                                                                       |
| `slope`    | The slope of the peak exercise ST segment:<br>0 = upsloping<br>1 = flat<br>2 = downsloping                                                               |
| `ca`       | Number of major vessels (0–3) colored by fluoroscopy                                                                                                     |
| `thal`     | Thalassemia:<br>1 = normal<br>2 = fixed defect<br>3 = reversible defect                                                                                  |
| `target`   | Diagnosis of heart disease (1 = disease present, 0 = no disease)                                                                                         |

---

## 🧪 Methodology

1. **Data Preprocessing:**
   - Null value check
   - Correlation analysis
   - Data normalization (MinMaxScaler)
   - Handling Outlier

2. **Feature Selection:**
   - Applied **Variance Inflation Factor (VIF)** to reduce multicollinearity

3. **Modeling:**
   - Machine learning models used:
     - Random Forest Classifier
     - Multilayer Perceptron (MLP)
     - XGBoost Classifier

4. **Hyperparameter Tuning:**
   - Performed for each model using GridSearchCV

5. **Validation:**
   - **10-fold Cross Validation**

6. **Evaluation Metrics:**
   - Accuracy
   - Precision
   - Recall
   - F1-Score

---

## 🧾 VIF Feature Selection

```python
# Calculate Variance Inflation Factor (VIF)
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

vif_result = calculate_vif(X_train)
print(vif_result)

```
| Feature   | age     | sex     | cp      | trestbps | chol    | fbs     | restecg | thalach | exang   | oldpeak | slope   | ca      | thal    |
|-----------|---------|---------|---------|----------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| VIF       | 10.07   | 3.59    | 2.38    | 7.78     | 8.98    | 1.28    | 2.04    | 15.97   | 2.08    | 3.15    | 9.24    | 1.92    | 15.04   |

## 🔍 Selected Features
The selected features were chosen based on a Variance Inflation Factor (VIF) threshold, where only features with VIF < 5 were retained.
`sex`,`cp`,`fbs`,`restecg`,`exang`,`oldpeak`,`ca`

## ⚙️ Best Hyperparameter each classification

 - Random Forest Best Hyperparameters
{
  'max_depth': None,
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 100
}

  - MLP Best Hyperparameters
{
  'activation': 'relu',
  'alpha': 0.01,
  'hidden_layer_sizes': (30, 20, 10)
}

  - XGBoost Best Hyperparameters
{
  'learning_rate': 0.2,
  'max_depth': 7,
  'n_estimators': 200
}

## 📊 Evaluation Metrics Each Classification (Average)
| Classification | Accuracy | Recall | Precicions | F1 Score | Computational Time (s) |
| -------------- | -------- | ------ | ---------- | -------- | ---------------------- |
| Random Forest  | 0.95     | 0.97   | 0.94       | 0.95     | 1.2                    |
| MLP            | 0.87     | 0.90   | 0.85       | 0.90     | 8.3                    |
| XG Boost       | 0.95     | 0.97   | 0.94       | 0.96     | 1.5                    |

Among the three classification methods above:
 - XGBoost achieved the highest accuracy.
 - Random Forest achieved the highest recall, precision, and F1-Score.
 - Based on the average computation time during 25 iterations with 10-fold cross-validation, Random Forest was the fastest, while Multilayer Perceptron was the slowest.
