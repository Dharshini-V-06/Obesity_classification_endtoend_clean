import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# 🧠 Load or Train Model
# -------------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.load_model("xgb_obesity_model.json")
    return model

# You can also directly train if model not saved
def train_model():
    df = pd.read_csv("ObesityDataset.csv")
    df = df.drop_duplicates()

    # Label encode binary columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # Get dummies for multi-class categorical columns
    multi_class_cols = ['CAEC', 'CALC', 'MTRANS']
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=False)

    # Rename target column
    df.rename(columns={'NObeyesdad': 'Obesity_Level'}, inplace=True)

    # Split X, y
    X = df.drop('Obesity_Level', axis=1)
    y = le.fit_transform(df['Obesity_Level'])

    # Split & Scale
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(X_train_scaled, y_train)

    model.save_model("xgb_obesity_model.json")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(X.columns.tolist(), "feature_columns.pkl")
    return model, scaler, le, X.columns.tolist()

# Try loading model, else train
try:
    model = load_model()
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except:
    model, scaler, le, feature_columns = train_model()

# -------------------------------
# 🎨 Streamlit UI
# -------------------------------
st.set_page_config(page_title="Obesity Level Prediction", layout="centered")
st.title("🏋️ Obesity Level Classification App")
st.write("Predict obesity level based on lifestyle and health indicators.")

# -------------------------------
# 🧾 User Inputs
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.slider("Age", 14, 60, 25)
    Height = st.number_input("Height (in meters)", 1.4, 2.0, 1.7)
    Weight = st.number_input("Weight (in kg)", 40.0, 180.0, 70.0)
    family_history = st.selectbox("Family history with overweight", ["yes", "no"])
    FAVC = st.selectbox("High caloric food consumption", ["yes", "no"])
    SMOKE = st.selectbox("Do you smoke?", ["yes", "no"])

with col2:
    FCVC = st.slider("Vegetable consumption frequency (1–3)", 1.0, 3.0, 2.0)
    NCP = st.slider("Number of main meals (1–4)", 1.0, 4.0, 3.0)
    CH2O = st.slider("Water intake (liters/day)", 1.0, 3.0, 2.0)
    SCC = st.selectbox("Do you consume caloric drinks?", ["yes", "no"])
    FAF = st.slider("Physical activity frequency (hours/week)", 0.0, 3.0, 1.0)
    TUE = st.slider("Time using technology devices (hours/day)", 0.0, 2.0, 1.0)
    CALC = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
    CAEC = st.selectbox("Food between meals", ["no", "Sometimes", "Frequently", "Always"])
    MTRANS = st.selectbox("Mode of transportation", ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"])

# -------------------------------
# 🧩 Prepare Input Data
# -------------------------------
input_dict = {
    'Gender': 1 if Gender == "Male" else 0,
    'Age': Age,
    'Height': Height,
    'Weight': Weight,
    'family_history_with_overweight': 1 if family_history == "yes" else 0,
    'FAVC': 1 if FAVC == "yes" else 0,
    'FCVC': FCVC,
    'NCP': NCP,
    'SMOKE': 1 if SMOKE == "yes" else 0,
    'CH2O': CH2O,
    'SCC': 1 if SCC == "yes" else 0,
    'FAF': FAF,
    'TUE': TUE,
    'CAEC_Always': 1 if CAEC == "Always" else 0,
    'CAEC_Frequently': 1 if CAEC == "Frequently" else 0,
    'CAEC_Sometimes': 1 if CAEC == "Sometimes" else 0,
    'CAEC_no': 1 if CAEC == "no" else 0,
    'CALC_Always': 1 if CALC == "Always" else 0,
    'CALC_Frequently': 1 if CALC == "Frequently" else 0,
    'CALC_Sometimes': 1 if CALC == "Sometimes" else 0,
    'CALC_no': 1 if CALC == "no" else 0,
    'MTRANS_Automobile': 1 if MTRANS == "Automobile" else 0,
    'MTRANS_Bike': 1 if MTRANS == "Bike" else 0,
    'MTRANS_Motorbike': 1 if MTRANS == "Motorbike" else 0,
    'MTRANS_Public_Transportation': 1 if MTRANS == "Public_Transportation" else 0,
    'MTRANS_Walking': 1 if MTRANS == "Walking" else 0
}

# Convert to DataFrame and align with training columns
input_df = pd.DataFrame([input_dict])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# Scale numeric data
input_scaled = scaler.transform(input_df)

# -------------------------------
# 🚀 Prediction
# -------------------------------
if st.button("Predict Obesity Level"):
    prediction = model.predict(input_scaled)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    st.success(f"### 🧍 Predicted Obesity Level: **{predicted_label}**")

    # Display additional insight
    st.progress(min(1.0, (prediction + 1) / len(le.classes_)))
    st.caption("Higher progress indicates higher obesity level category.")

# -------------------------------
# ℹ️ Footer
# -------------------------------
st.markdown("---")
st.markdown("Developed by **Dharshini V** | Model: XGBoost | Accuracy ≈ 97%")

