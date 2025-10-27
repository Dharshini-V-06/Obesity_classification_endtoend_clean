#  Obesity Level Classification — End-to-End Machine Learning Project

##  Project Overview

This project predicts **Obesity Levels** of individuals based on lifestyle, demographic, and dietary habits using various **supervised machine learning algorithms**.
The dataset contains medical and behavioral attributes like age, height, weight, family history of obesity, food habits, physical activity, and more.

---

##  Objective

To build and compare multiple machine learning models that classify individuals into one of **seven obesity levels**:

* Insufficient Weight
* Normal Weight
* Overweight Level I
* Overweight Level II
* Obesity Type I
* Obesity Type II
* Obesity Type III

---

##  Dataset Information

**Dataset Name:** ObesityDataset.csv
**Total Records:** 2,111
**Total Features:** 17
**Target Variable:** `NObeyesdad` (renamed to `Obesity_Level`)

**Dataset link:** [Obesity dataset](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster)


### 📑 Feature Description

| Feature                        | Type        | Description                            |
| ------------------------------ | ----------- | -------------------------------------- |
| Gender                         | Categorical | Male / Female                          |
| Age                            | Numeric     | Age of individual                      |
| Height                         | Numeric     | Height (meters)                        |
| Weight                         | Numeric     | Weight (kg)                            |
| family_history_with_overweight | Categorical | Whether obesity runs in family         |
| FAVC                           | Categorical | Frequent high-calorie food consumption |
| FCVC                           | Numeric     | Frequency of vegetable consumption     |
| NCP                            | Numeric     | Number of main meals per day           |
| CAEC                           | Categorical | Consumption of food between meals      |
| SMOKE                          | Categorical | Smoking habits                         |
| CH2O                           | Numeric     | Daily water intake                     |
| SCC                            | Categorical | Caloric drink consumption              |
| FAF                            | Numeric     | Physical activity frequency            |
| TUE                            | Numeric     | Time spent on electronic devices       |
| CALC                           | Categorical | Alcohol consumption                    |
| MTRANS                         | Categorical | Mode of transportation                 |
| Obesity_Level                  | Target      | Obesity category                       |

---

##  Data Preprocessing

###  Steps Performed

1. **Removed duplicates:**

   * 24 duplicates dropped → Final records: 2,087

2. **Checked for missing values:**

   * No missing data found

3. **Outlier Detection (IQR Method):**

   * Outliers found in `Age`, `Height`, `Weight`, `NCP`
   * Capped outliers only in `NCP` column

4. **Skewness Correction:**

   * Applied `PowerTransformer` (Yeo-Johnson) on `Age` to normalize distribution

5. **Encoding:**

   * Label Encoding for binary columns (`Gender`, `FAVC`, `SCC`, etc.)
   * One-Hot Encoding for multi-category columns (`CAEC`, `CALC`, `MTRANS`)
   * Label Encoding for target variable (`Obesity_Level`)

6. **Feature Scaling:**

   * Standardization using `StandardScaler`

---

##  Exploratory Data Analysis (EDA)

* **Countplots & Pie Charts:** Visualized categorical feature distributions (Gender, Alcohol consumption, Obesity levels, etc.)
* **Heatmap:** Examined correlations among numerical variables
* **Boxplots:** Identified outliers across numeric features

---

##  Model Building & Evaluation

###  Models Used

| Model                            | Accuracy      | Precision | Recall | F1 Score |
| -------------------------------- | ------------- | --------- | ------ | -------- |
| **Random Forest**                | 0.9498        | 0.9523    | 0.9480 | 0.9488   |
| **Logistic Regression**          | 0.9091        | 0.9053    | 0.9067 | 0.9053   |
| **Decision Tree**                | 0.9282        | 0.9294    | 0.9258 | 0.9263   |
| **K-Nearest Neighbors (KNN)**    | 0.8158        | 0.8051    | 0.8090 | 0.8048   |
| **Support Vector Machine (SVM)** | 0.8612        | 0.8645    | 0.8584 | 0.8599   |
| **XGBoost**                      |  **0.9713** | 0.9706    | 0.9708 | 0.9706   |

---

##  Key Insights

* The dataset was **well-balanced** across obesity categories.
* **XGBoost** achieved the **best overall performance** (97% accuracy).
* **Random Forest** also performed very well with 95% accuracy.
* **KNN** showed the lowest performance, indicating high sensitivity to feature scaling or outliers.

---

##  Confusion Matrices

Each model’s performance was visualized using confusion matrices to assess class-wise predictions.

---

##  Tech Stack

| Category                            | Tools Used                                              |
| ----------------------------------- | ------------------------------------------------------- |
| **Programming Language**            | Python                                                  |
| **Libraries (EDA & Visualization)** | Pandas, NumPy, Matplotlib, Seaborn                      |
| **ML Modeling**                     | Scikit-learn, XGBoost                                   |
| **Evaluation Metrics**              | Accuracy, Precision, Recall, F1 Score, Confusion Matrix |
| **Feature Scaling & Encoding**      | StandardScaler, LabelEncoder, OneHotEncoder             |

---

##  Conclusion

The project successfully demonstrates an **end-to-end machine learning pipeline** — from data preprocessing, EDA, feature transformation, model training, and evaluation.
Among all models tested, **XGBoost** proved to be the **most robust and accurate** classifier for predicting obesity levels.

---

