import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math

# Title of the Streamlit app
st.title("Store Sales Time Series Forecasting")

# Step 1: Load Data
st.subheader("Step 1: Load Sales Data")
uploaded_file = st.file_uploader("Upload your sales data (CSV file)", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    # Read the CSV file with a different encoding
    sales_data = pd.read_csv(uploaded_file, encoding='ISO-8859-1', parse_dates=['date'], index_col='date')


    # Show first few rows of the dataset
    st.write("Data preview:")
    st.write(sales_data.head())

    # Step 2: Data Preprocessing
    st.subheader("Step 2: Preprocess Sales Data")
    sales_data_daily = sales_data.resample('D').sum()

    st.write("Daily aggregated sales data:")
    st.line_chart(sales_data_daily['sales'])

    # Step 3: Train/Test Split
    st.subheader("Step 3: Train/Test Split and Model Training")
    train_size = st.slider("Select training data percentage", 50, 90, 80)

    split_point = int(len(sales_data_daily) * (train_size / 100))
    train, test = sales_data_daily[:split_point], sales_data_daily[split_point:]

    # Display Train/Test Split
    st.write(f"Training data from {train.index.min()} to {train.index.max()}")
    st.write(f"Test data from {test.index.min()} to {test.index.max()}")

    # Step 4: Build and Train ARIMA Model
    st.subheader("Step 4: ARIMA Model Training and Forecasting")

 # Separate sliders for each ARIMA parameter
    p = st.slider("Select ARIMA parameter (p)", 0, 5, 1)
    d = st.slider("Select ARIMA parameter (d)", 0, 2, 1)
    q = st.slider("Select ARIMA parameter (q)", 0, 5, 1)


    model = ARIMA(train['sales'], order=(p, d, q))
    model_fit = model.fit()

    # Forecasting
    forecast = model_fit.forecast(steps=len(test))

    # Step 5: Model Evaluation
    st.subheader("Step 5: Model Evaluation")
    rmse = math.sqrt(mean_squared_error(test['sales'], forecast))
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot Actual vs. Forecasted Sales
    st.write("Actual vs. Predicted Sales:")
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train['sales'], label='Train')
    plt.plot(test.index, test['sales'], label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.legend()
    st.pyplot(plt)

    # Show Residuals
    st.write("Residuals Analysis:")
    residuals = test['sales'] - forecast
    st.line_chart(residuals)
