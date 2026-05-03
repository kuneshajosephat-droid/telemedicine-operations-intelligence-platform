
📊  Architecture

This project implements an end-to-end data engineering and machine learning pipeline for analyzing telehealth adoption across regions.

1. Data Ingestion
Medicare Telehealth dataset (CSV)
Data loaded using Python
Uploaded to AWS S3 (raw data)
2. Data Storage
AWS S3
Raw Zone (original data)
Processed Zone (cleaned data)
3. Data Processing
AWS Glue (ETL)
Data cleaning
Data transformation
Aggregation
Schema mapping
Databricks (PySpark & Python)
Feature engineering
Data processing
Preparation for modeling
4. Machine Learning
Databricks Notebook (EDA & modeling)
Python script: scripts/train_model_api.py
Models used:
Linear Regression
Random Forest Regressor
Output:
telehealth_model.pkl
5. Data Consumption (Application Layer)
AWS Athena (SQL querying)
Streamlit dashboard
KPI metrics
Trend analysis
Regional insights
Forecast visualization
Data explorer
Local CSV fallback used when Athena is unavailable
6. Model Deployment
FastAPI (REST API)
Uvicorn server
Endpoint: /predict
Model loaded from .pkl file
7. Data Visualization
Streamlit (interactive dashboard)
Plotly (charts and graphs)
8. Deployment Flow

Databricks → Model (.pkl) → FastAPI → Streamlit → End Users
