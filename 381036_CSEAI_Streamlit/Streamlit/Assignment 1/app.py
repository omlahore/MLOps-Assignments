import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("CSV Data Visualizer")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())
    
    chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Histogram"])
    
    if chart_type == "Line":
        st.line_chart(data)
    elif chart_type == "Bar":
        st.bar_chart(data)
    elif chart_type == "Histogram":
        numeric_columns = data.select_dtypes(include='number')
        for col in numeric_columns.columns:
            fig, ax = plt.subplots()
            ax.hist(data[col].dropna(), bins=20)
            st.write(f"Histogram for {col}")
            st.pyplot(fig)
