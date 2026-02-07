import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Page setup
st.set_page_config(page_title="Waste Management Prediction", layout="wide")

st.title("♻️ Waste Management Prediction App")
st.write("Predict Waste Generated (Tons/Day) using Machine Learning")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("Waste_Management_and_Recycling_India.xlsx")

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Target column
target_column = "Waste Generated (Tons/Day)"

# Preprocessing
df_model = df.copy()
label_encoders = {}

for col in df_model.columns:
    if df_model[col].dtype == "object":
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

X = df_model.drop(target_column, axis=1)
y = df_model[target_column]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

st.success("Model trained successfully")

# User input
st.subheader("Enter Input Values")
user_input = {}

for col in X.columns:
    if col in label_encoders:
        user_input[col] = st.selectbox(col, label_encoders[col].classes_)
    else:
        user_input[col] = st.number_input(col, value=float(X[col].mean()))

# Prediction
if st.button("Predict Waste Generated"):
    input_df = pd.DataFrame([user_input])

    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Waste Generated: {prediction:.2f} Tons/Day")
