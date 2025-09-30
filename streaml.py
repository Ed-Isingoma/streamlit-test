import streamlit as st
import pandas as pd

st.title("Simple CSV Data Viewer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.header("Data from the uploaded CSV file:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error: {e}") 