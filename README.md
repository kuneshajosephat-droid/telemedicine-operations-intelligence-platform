
# 📊 Telemedicine Operations Intelligence Platform

End-to-end data engineering and machine learning platform for analyzing and forecasting telehealth adoption using AWS, Databricks, and Python.

##  Overview

This project builds a complete data pipeline and analytics platform that:

- Ingests telehealth data (CSV)
- Stores and processes data in AWS S3
- Transforms data using AWS Glue and Databricks
- Trains machine learning models (Random Forest, Linear Regression)
- Serves predictions via FastAPI
- Visualizes insights through an interactive Streamlit dashboard

The platform supports *data-driven healthcare decision-making*  by identifying trends, regional disparities, and future telehealth adoption patterns.


## 🏗️ Architecture
[Architecture](docs/architecture.png)

## 📊 Interactive Dashboard
[Dashboard](docs/dashboard.png)

## 🔌 API Integration (FastAPI)
[API](docs/api.png)

The API exposes a real-time prediction endpoint:
GET /predict

### Example parameters:
- `year`
- `region`
- `total_enrollment`
- `telehealth_users`

## 🧠 Machine Learning

Models implemented:
- Linear Regression
- Random Forest Regressor (best performing)

Output:
- Forecasted telehealth adoption rate (%)
Model saved as: models/telehealth_model.pkl


## 🛠️ Tech Stack

### Data Engineering
- AWS S3 (data storage)
- AWS Glue (ETL)
- Amazon Athena (SQL analytics)

### Data Processing & ML
- Databricks (PySpark, notebooks)
- Python (Pandas, NumPy, Scikit-learn)

### Application Layer
- Streamlit (dashboard)
- FastAPI (model API)
- Uvicorn (server)

### Visualization
- Streamlit charts
- Plotly (interactive graphs)

## 📂 Project Structure
telemedicine-operations-intelligence-platform/
│
├── api/ # FastAPI application
├── dashboards/ # Streamlit dashboard
├── data/ # Raw & processed data
├── docs/ # Architecture & visuals
├── models/ # Trained ML model (.pkl)
├── notebooks/ # Databricks/Jupyter notebook
├── scripts/ # Model training scripts
├── README.md

## ▶️ How to Run

### 1. Run Streamlit Dashboard
```bash
cd dashboards
streamlit run app.py

2. Run FastAPI
uvicorn api.main:app --reload
Then open:
http://127.0.0.1:8000/docs

📌 Key Features
End-to-end data engineering pipeline
Machine learning forecasting
Real-time API predictions
Interactive dashboard for insights
Regional analysis of telehealth adoption

📈 Insights Delivered
Telehealth adoption trends over time
Regional disparities in access
Forecasted adoption rates
Identification of low-performing regions
Data-driven policy recommendations

⚠️ Notes
AWS Athena access may require permissions; local CSV fallback is used for demo
Forecasting is based on available historical data and should be interpreted as indicative

👤 Author

Josephat Kunesha
Data Scientist & Data Engineer

GitHub: https://github.com/kuneshajosephat-droid

LinkedIn: https://www.linkedin.com/in/josephat-kunesha

 ⭐A REAL WORLD PIPELINE : Data Engineering + Machine Learning + Deployment + Visualization


