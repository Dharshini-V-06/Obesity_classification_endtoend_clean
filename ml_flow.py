# ==========================================================
# OBESITY LEVEL CLASSIFICATION WITH MLFLOW (Simplified)
# ==========================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

import mlflow
import mlflow.sklearn

# ----------------------------------------------------------
# STEP 1: Load and preprocess dataset
# ----------------------------------------------------------
df = pd.read_csv("ObesityDataset.csv")
df.drop_duplicates(inplace=True)
df.rename(columns={'NObeyesdad': 'Obesity_Level'}, inplace=True)

# Encode binary categorical columns
binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-class columns
multi_class_cols = ['CAEC', 'CALC', 'MTRANS']
df = pd.get_dummies(df, columns=multi_class_cols, drop_first=False)

# Correct skewness
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Obesity_Level')
pt = PowerTransformer(method='yeo-johnson')
df[num_cols] = pt.fit_transform(df[num_cols])

# Split features and target
X = df.drop('Obesity_Level', axis=1)
y = le.fit_transform(df['Obesity_Level'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# STEP 2: Setup MLflow Experiment
# ----------------------------------------------------------
EXPERIMENT_NAME = "Obesity_Classification"
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------------------------------------
# STEP 3: Define function for training + logging
# ----------------------------------------------------------
def train_and_log_model(model, model_name):
    """
    Train model, evaluate, and log metrics & plots to MLflow.
    """
    with mlflow.start_run(run_name=model_name, nested=True):
        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        # Confusion matrix visualization (logged directly to MLflow)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{model_name} - Confusion Matrix")

        # Log plot directly
        mlflow.log_figure(plt.gcf(), f"{model_name}_confusion_matrix.png")
        plt.close()

        # Print results
        print(f"✅ {model_name} Results:")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall   : {rec:.4f}")
        print(f"   F1-score : {f1:.4f}\n")

# ----------------------------------------------------------
# STEP 4: Train and log multiple models
# ----------------------------------------------------------
models = {
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    )
}

# Main parent run to group all child runs
with mlflow.start_run(run_name="All_Models_Training"):
    for name, mdl in models.items():
        train_and_log_model(mdl, name)

print("\n🎯 All models trained and logged successfully to MLflow experiment:", EXPERIMENT_NAME)
print("👉 Run this command in terminal to open MLflow UI:\n")
print("   mlflow ui --backend-store-uri mlruns")
print("\nThen open http://127.0.0.1:5000 in your browser.")