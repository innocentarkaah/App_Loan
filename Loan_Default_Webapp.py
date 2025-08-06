import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# ----------------------------
# Page Configuration
# ----------------------------
sns.set_theme(style="whitegrid", palette="muted")
st.set_page_config(page_title="Loan Default Prediction App", layout="wide")
st.title("💰 Loan Default Prediction App")
st.markdown("This app predicts loan default risk using XGBoost, with cached, beautifully rendered charts.")

# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data
def load_data():
    """
    Load dataset from CSV.
    """
    return pd.read_csv("Loan_default.csv")

# ----------------------------
# Cached Plot Functions
# ----------------------------
@st.cache_data
def plot_correlation(df):
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", linewidths=0.4, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    return fig

@st.cache_data
def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(x='Default', data=df, palette=['#4c72b0', '#dd8452'], ax=ax)
    ax.set_title("Loan Default Distribution", fontsize=14)
    ax.set_xlabel("Default (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom')
    plt.tight_layout()
    return fig

@st.cache_data
def plot_histograms(df, cols):
    figures = []
    for col in cols:
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.histplot(df[col], kde=True, stat='density', edgecolor='white', ax=ax)
        ax.set_title(f"Distribution of {col}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        plt.tight_layout()
        figures.append(fig)
    return figures

@st.cache_data
def plot_combined_boxplot(df, cols):
    melt_df = df[cols].melt(var_name='Feature', value_name='Value')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Feature', y='Value', data=melt_df, palette='Set2', ax=ax)
    ax.set_title("Box Plot of Key Numeric Features", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ----------------------------
# Exploratory Data Analysis
# ----------------------------
def show_eda(df):
    with st.expander("🔍 Exploratory Data Analysis (EDA)"):
        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))

        st.subheader("Correlation Matrix")
        fig = plot_correlation(df)
        st.pyplot(fig)

        st.subheader("Target Distribution")
        fig = plot_target_distribution(df)
        st.pyplot(fig)

        st.subheader("Histograms of Numeric Features")
        num_cols = df.select_dtypes(include=[np.number]).columns.drop('Default')[:6]
        for fig in plot_histograms(df, list(num_cols)):
            st.pyplot(fig)

        st.subheader("Box Plot of Numeric Features")
        fig = plot_combined_boxplot(df, list(num_cols))
        st.pyplot(fig)

# ----------------------------
# Model Pipeline
# ----------------------------
def build_pipeline(params=None):
    num_feats = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    cat_feats = ['Education', 'EmploymentType', 'MaritalStatus',
                 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_feats),
        ('cat', cat_pipe, cat_feats)
    ])

    xgb_params = params or {'use_label_encoder': False, 'eval_metric': 'logloss'}
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**xgb_params))
    ])
    return pipeline

# ----------------------------
# Model Training
# ----------------------------
def train_model(X, y):
    pipeline = build_pipeline()
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.8, 1.0]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    st.sidebar.success(f"Best Params: {grid.best_params_}")
    joblib.dump(grid.best_estimator_, 'xgb_model.joblib')
    return grid.best_estimator_

# ----------------------------
# Model Evaluation
# ----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.subheader('Classification Report')
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df)

    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------
# Prediction Interface
# ----------------------------
def prediction_interface(model, df):
    """
    Sidebar for prediction inputs without SHAP explanation.
    """
    st.sidebar.header('🔮 Make a Prediction')
    numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    categorical_features = ['Education', 'EmploymentType', 'MaritalStatus',
                            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

    user_input = {}
    for feat in numeric_features:
        user_input[feat] = st.sidebar.number_input(feat, value=float(df[feat].median()))
    for feat in categorical_features:
        user_input[feat] = st.sidebar.selectbox(feat, df[feat].unique())

    if st.sidebar.button('Predict Default'):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader('Prediction Result')
        st.write(f"**Will Default?** {'Yes' if pred else 'No'}")
        st.write(f"**Default Probability:** {prob:.2%}")

# ----------------------------
def prediction_interface(model, df):
    st.sidebar.header('🔮 Make a Prediction')
    numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    categorical_features = ['Education', 'EmploymentType', 'MaritalStatus',
                            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

    user_input = {}
    for feat in numeric_features:
        user_input[feat] = st.sidebar.number_input(feat, value=float(df[feat].median()))
    for feat in categorical_features:
        user_input[feat] = st.sidebar.selectbox(feat, df[feat].unique())

    if st.sidebar.button('Predict Default'):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader('Prediction Result')
        st.write(f"**Will Default?** {'Yes' if pred else 'No'}")
        st.write(f"**Default Probability:** {prob:.2%}")

        booster = model.named_steps['classifier'].get_booster()
        explainer = shap.TreeExplainer(booster)
        transformed = model.named_steps['preprocessor'].transform(input_df)
        shap_values = explainer.shap_values(transformed)
        fig = shap.force_plot(explainer.expected_value, shap_values[0], transformed[0], matplotlib=True)
        st.subheader('Explanation (SHAP Force Plot)')
        st.pyplot(fig)

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == '__main__':
    data = load_data()

    with st.expander('🗄️ Data Overview', expanded=False):
        st.subheader('Data Head & Types')
        st.dataframe(data.head())
        st.dataframe(pd.DataFrame(data.dtypes, columns=['Type']))
        st.subheader('Total Missing Values')
        st.write(data.isnull().sum().sum())

    show_eda(data)

    X = data.drop(columns=['LoanID', 'Default'])
    y = data['Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = joblib.load('xgb_model.joblib')
        st.sidebar.success('Loaded model from disk')
    except FileNotFoundError:
        model = train_model(X_train, y_train)
        st.sidebar.success('Model trained and saved')

    with st.expander('📈 Model Evaluation', expanded=False):
        evaluate_model(model, X_test, y_test)

    prediction_interface(model, data)
