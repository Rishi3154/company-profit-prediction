import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
# Page config
st.set_page_config(page_title="Company Profit Prediction", layout="wide")
st.title("📈 Company Profit Prediction")

# Create tabs
tabs = st.tabs(["Single Prediction", "Batch Prediction", "Analysis"])


with tabs[0]:
    st.header("Predict Profit for a Single Input")

    col1, col2, col3 = st.columns(3)
    with col1:
        rnd_spend = st.number_input("R&D Spend", min_value=0.0, format="%.2f")
    with col2:
        admin_cost = st.number_input("Administration", min_value=0.0, format="%.2f")
    with col3:
        marketing_spend = st.number_input("Marketing Spend", min_value=0.0, format="%.2f")

    if st.button("Predict Profit"):
        features = np.array([[rnd_spend, admin_cost, marketing_spend]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        st.success(f"Predicted Profit: ${prediction:,.2f}")


with tabs[1]:
    st.header("Batch Prediction using CSV Upload")
    uploaded_file = st.file_uploader("Upload a CSV file with columns: R&D Spend, Administration, Marketing Spend", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview:")
        st.dataframe(df.head())

        if st.button("Predict for All Rows"):
            try:
                features = df[['R&D Spend', 'Administration', 'Marketing Spend']]
                scaled_features = scaler.transform(features)
                predictions = model.predict(scaled_features)
                df['Predicted Profit'] = predictions

                st.subheader("Predictions:")
                st.dataframe(df)

                # Download link for CSV
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">📥 Download Predictions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[2]:
    st.header("Feature Importance & Outlier Analysis")

    col1, col2 = st.columns(2)

    # Feature Importance
    with col1:
        st.subheader("Feature Importance")
        try:
            estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model

            if hasattr(estimator, 'coef_'):
                importance = estimator.coef_[0]
            elif hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
            else:
                st.warning("Feature importance is not available for this model.")
                importance = None


            features = ['R&D Spend', 'Administration', 'Marketing Spend']
            fig, ax = plt.subplots()
            sns.barplot(x=importance, y=features, palette="viridis", ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error displaying feature importance: {e}")

    # Outlier Detection
    with col2:
        st.subheader("Outlier Detection")
        try:
            # Load dataset for outlier detection
            df = pd.read_csv('dataset.csv')
            fig, ax = plt.subplots()
            sns.boxplot(data=df[['R&D Spend', 'Administration', 'Marketing Spend']], ax=ax)
            ax.set_title("Outlier Detection")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error displaying outlier chart: {e}")
