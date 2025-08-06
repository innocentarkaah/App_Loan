import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set Streamlit page configuration
st.set_page_config(page_title="Loan Default Predictor", layout="centered")

# =======================
# Load Model and Columns
# =======================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_columns():
    cat_cols = joblib.load("cat_cols.pkl")
    num_cols = joblib.load("num_cols.pkl")
    return cat_cols, num_cols

model = load_model()
cat_cols, num_cols = load_columns()

# ===================================
# Input Sidebar for User Prediction
# ===================================

st.sidebar.header("Input Loan Application Details")

def user_input_features():
    inputs = {}
    for col in num_cols:
        inputs[col] = st.sidebar.number_input(f"{col}", min_value=0.0, step=1.0)

    for col in cat_cols:
        inputs[col] = st.sidebar.selectbox(f"{col}", options=["A", "B", "C", "D", "E"])  # Customize options as needed

    return pd.DataFrame([inputs])

input_df = user_input_features()

# ===========================
# Prediction and Explanation
# ===========================

st.title("Loan Default Prediction App")
st.write("Predict whether a customer will default on their loan.")

if st.button("Predict"):
    if input_df is not None:
        # Preprocessing
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(f"### 🔴 Prediction: This applicant is **likely to default**.")
        else:
            st.markdown(f"### 🟢 Prediction: This applicant is **unlikely to default**.")
        st.write(f"Probability of default: **{prediction_proba:.2f}**")

# ===============================
# Sample Data for Visualizations
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")  # Replace with your actual dataset used for EDA
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str)
    return df

df = load_data()

# ===============================
# Visualization Dropdown Section
# ===============================

with st.expander("📊 Show Exploratory Charts and Data"):
    st.subheader("Target Variable Distribution")

    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Default', data=df, palette=['#4c72b0', '#dd8452'], ax=ax1)
    st.pyplot(fig1)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Boxplot of Numeric Features")

    melted = df[num_cols].melt(var_name='Feature', value_name='Value')
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Feature', y='Value', data=melted, palette='Set2', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")

    corr = df[num_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
    st.pyplot(fig3)



