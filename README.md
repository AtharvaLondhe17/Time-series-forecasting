Overview
This Streamlit app is designed for forecasting store sales using the ARIMA (AutoRegressive Integrated Moving Average) model. Users can upload their sales data and visualize predictions alongside actual sales figures, making it easier to understand trends and make data-driven decisions.

 Features
- Data Upload: Users can upload sales data in CSV format.
- Data Visualization: Displays daily aggregated sales data and actual vs. predicted sales.
- ARIMA Model: Users can select ARIMA parameters (p, d, q) and train the model on their data.
- Model Evaluation: Provides Root Mean Squared Error (RMSE) as a measure of model performance.
- Residual Analysis: Visualizes the residuals of the predictions.

Requirements
To run this app, ensure you have the following Python packages installed:

pandas
numpy
streamlit
matplotlib
statsmodels
scikit-learn
