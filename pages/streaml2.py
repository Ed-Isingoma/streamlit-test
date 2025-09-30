import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Automated Model Trainer", layout="wide")

st.title("Interactive Machine Learning Model Trainer")

st.write("""
Upload your CSV file, choose your target variable, and this app will automatically
preprocess the data, train a Random Forest model, and evaluate its performance.
""")

uploaded_file = st.file_uploader("Upload your data in CSV format", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.header("Uploaded Data Preview")
        st.dataframe(df.head())

        # --- User Configuration ---
        st.sidebar.header("Model Configuration")
        
        # 1. Select the target column
        target_column = st.sidebar.selectbox(
            "Select the Target Variable (what you want to predict)",
            df.columns
        )

        # 2. Select the problem type
        problem_type = st.sidebar.selectbox(
            "Select the Problem Type",
            ["Classification", "Regression"]
        )
        
        # --- Start Training Button ---
        if st.sidebar.button("Train Model"):
            with st.spinner('Preprocessing data and training model...'):

                # --- Data Preprocessing ---
                st.header("Model Training and Evaluation")

                # Drop rows where the target column has missing values
                df.dropna(subset=[target_column], inplace=True)
                
                # Separate features (X) and target (y)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Identify categorical and numerical columns in features
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                numerical_cols = X.select_dtypes(include=['number']).columns

                # Handle categorical features using one-hot encoding
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

                # For classification, encode the target variable if it's not numeric
                if problem_type == "Classification" and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # --- Model Training ---
                if problem_type == "Classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    performance = accuracy_score(y_test, predictions)
                    metric_name = "Accuracy"
                else: # Regression
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    performance = r2_score(y_test, predictions)
                    metric_name = "R-squared"
                
                # --- Display Results ---
                st.subheader(f"Model Performance ({metric_name})")
                st.success(f"{metric_name}: {performance:.4f}")

                # Display feature importances
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importances")
                    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['Value','Feature'])
                    st.bar_chart(feature_imp.set_index('Feature'))

    except Exception as e:
        st.error(f"An error occurred: {e}")