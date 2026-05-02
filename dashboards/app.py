
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyathena import connect
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import requests

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Telemedicine Operations Intelligence Platform",
    layout="wide"
)

st.title("📊 Telemedicine Operations Intelligence Platform")
st.caption("Cloud-based healthcare analytics platform powered by AWS S3, Glue, Athena, Python, Streamlit, and FastAPI")

# -----------------------------
# LOAD DATA FROM ATHENA
# -----------------------------
@st.cache_data(ttl=600)
def load_data():
    conn = connect(
        s3_staging_dir="s3://telemedicine-operations-intelligence-kunesha/athena-results/",
        region_name="us-east-1"
    )

    query = """
    SELECT year, quarter, region, total_enrollment, telehealth_users, telehealth_rate
    FROM telehealth_db.processed
    """

    df = pd.read_sql(query, conn)

    df.columns = df.columns.str.lower()
    df = df[df["quarter"] == "Overall"].copy()

    df["year"] = df["year"].astype(int)
    df["telehealth_rate"] = df["telehealth_rate"] * 100

    return df


df = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

years = sorted(df["year"].unique())
regions = sorted(df["region"].unique())

selected_years = st.sidebar.multiselect("Select Year", years, default=years)
selected_regions = st.sidebar.multiselect("Select Region", regions, default=[])

filtered_df = df[df["year"].isin(selected_years)].copy()

if selected_regions:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# -----------------------------
# KPI METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Enrollment", f"{int(filtered_df['total_enrollment'].sum()):,}")
col2.metric("Total Telehealth Users", f"{int(filtered_df['telehealth_users'].sum()):,}")
col3.metric("Avg Telehealth Rate", f"{filtered_df['telehealth_rate'].mean():.2f}%")

st.markdown("""
📌 **Platform Purpose:** This dashboard helps healthcare leaders monitor telehealth adoption, identify regional disparities, and support data-driven planning for digital healthcare access.
""")

# -----------------------------
# STRATEGIC INSIGHTS
# -----------------------------
st.subheader("🧠 Strategic Insights for Healthcare Decision-Makers")

latest_year = int(filtered_df["year"].max())
latest_df = filtered_df[filtered_df["year"] == latest_year]

high_regions = latest_df[latest_df["telehealth_rate"] > 50]["region"].nunique()
low_regions = latest_df[latest_df["telehealth_rate"] < 25]["region"].nunique()

low_trend = filtered_df.groupby("region")["telehealth_rate"].mean()
at_risk = low_trend[low_trend < 25].count()

st.write(f"""
- In **{latest_year}**, **{high_regions} regions** show high telehealth adoption above 50%.
- **{low_regions} regions** remain below 25% adoption in the latest selected year.
- Across the selected period, **{at_risk} regions** have an average adoption rate below 25%.
- These gaps may indicate differences in digital access, provider availability, broadband infrastructure, or regional healthcare readiness.
""")

# -----------------------------
# POLICY RECOMMENDATION
# -----------------------------
st.subheader("📌 Policy and Business Recommendation")

st.info("""
Healthcare organizations and policymakers should prioritize regions with persistently low telehealth adoption.

Recommended actions:
- Expand broadband and digital infrastructure
- Improve patient digital literacy
- Incentivize telehealth provider participation
- Monitor adoption trends continuously
- Use regional analytics to guide resource allocation

This supports healthcare access equity and helps institutions make evidence-based operational decisions.
""")

# -----------------------------
# NATIONAL TREND
# -----------------------------
st.subheader("📈 Telehealth Adoption Trend")

trend = (
    filtered_df.groupby("year", as_index=False)["telehealth_rate"]
    .mean()
    .sort_values("year")
)

trend["growth"] = trend["telehealth_rate"].pct_change() * 100

trend_display = trend.copy()
trend_display["year"] = trend_display["year"].astype(str)

st.line_chart(trend_display, x="year", y="telehealth_rate")

if len(trend) > 1:
    first_year = int(trend.iloc[0]["year"])
    last_year = int(trend.iloc[-1]["year"])
    first_rate = trend.iloc[0]["telehealth_rate"]
    last_rate = trend.iloc[-1]["telehealth_rate"]
    change = last_rate - first_rate

    st.write(f"""
    From **{first_year}** to **{last_year}**, the average telehealth adoption rate changed from 
    **{first_rate:.2f}%** to **{last_rate:.2f}%**, representing a change of **{change:.2f} percentage points**.
    """)

# -----------------------------
# REGION-LEVEL ML FORECASTING
# -----------------------------
st.subheader("🤖 Region-Level Machine Learning Forecast")

forecast_data = (
    filtered_df.groupby(["year", "region"], as_index=False)["telehealth_rate"]
    .mean()
    .sort_values(["region", "year"])
)

if forecast_data["year"].nunique() >= 3 and forecast_data["region"].nunique() >= 2:

    model_df = forecast_data.copy()
    model_df["region_code"] = model_df["region"].astype("category").cat.codes

    region_mapping = (
        model_df[["region", "region_code"]]
        .drop_duplicates()
        .set_index("region")["region_code"]
        .to_dict()
    )

    X = model_df[["year", "region_code"]]
    y = model_df["telehealth_rate"]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    lr_preds = lr.predict(X)
    lr_mae = mean_absolute_error(y, lr_preds)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=5
    )
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    rf_mae = mean_absolute_error(y, rf_preds)

    st.write("### Model Performance Comparison")
    st.write(f"""
    - Linear Regression MAE: **{lr_mae:.2f} percentage points**
    - Random Forest MAE: **{rf_mae:.2f} percentage points**
    """)

    best_model_name = "Random Forest Regressor" if rf_mae < lr_mae else "Linear Regression"
    st.success(f"Best performing model: **{best_model_name}**")

    # Forecast next 3 years
    last_year = int(model_df["year"].max())
    future_years = [last_year + 1, last_year + 2, last_year + 3]

    future_rows = []

    for region, code in region_mapping.items():
        for year in future_years:
            future_rows.append({
                "year": year,
                "region": region,
                "region_code": code
            })

    future_df = pd.DataFrame(future_rows)

    forecast_lr = future_df.copy()
    forecast_lr["forecast"] = lr.predict(forecast_lr[["year", "region_code"]])

    forecast_rf = future_df.copy()
    forecast_rf["forecast"] = rf.predict(forecast_rf[["year", "region_code"]])

    forecast_lr["forecast"] = forecast_lr["forecast"].clip(0, 100)
    forecast_rf["forecast"] = forecast_rf["forecast"].clip(0, 100)

    lr_avg = (
        forecast_lr.groupby("year", as_index=False)["forecast"]
        .mean()
        .rename(columns={"forecast": "Linear Regression Forecast"})
    )

    rf_avg = (
        forecast_rf.groupby("year", as_index=False)["forecast"]
        .mean()
        .rename(columns={"forecast": "Random Forest Forecast"})
    )

    actual_avg = (
        trend[["year", "telehealth_rate"]]
        .rename(columns={"telehealth_rate": "Actual Adoption Rate"})
    )

    if best_model_name == "Random Forest Regressor":
        final_forecast_year = int(rf_avg.iloc[-1]["year"])
        final_forecast_value = rf_avg.iloc[-1]["Random Forest Forecast"]
    else:
        final_forecast_year = int(lr_avg.iloc[-1]["year"])
        final_forecast_value = lr_avg.iloc[-1]["Linear Regression Forecast"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_avg["year"],
        y=actual_avg["Actual Adoption Rate"],
        mode="lines+markers",
        name="Actual Adoption Rate",
        line=dict(width=4, color="#0B3C5D"),
        marker=dict(size=9)
    ))

    fig.add_trace(go.Scatter(
        x=lr_avg["year"],
        y=lr_avg["Linear Regression Forecast"],
        mode="lines+markers",
        name="Linear Regression Forecast",
        line=dict(width=3, dash="dash", color="#F4A261"),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=rf_avg["year"],
        y=rf_avg["Random Forest Forecast"],
        mode="lines+markers",
        name="Random Forest Forecast",
        line=dict(width=3, dash="dot", color="#2A9D8F"),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title={
            "text": "Telehealth Adoption Forecast: Actual vs Model-Based Projections",
            "x": 0.02,
            "xanchor": "left"
        },
        xaxis_title="Year",
        yaxis_title="Telehealth Adoption Rate (%)",
        template="plotly_white",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=80, b=90)
    )

    fig.update_yaxes(range=[0, 60], ticksuffix="%")
    fig.update_xaxes(type="category")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Executive Forecast Insight:**  
    The model comparison indicates that **{best_model_name}** performs better based on MAE.  
    Under current regional patterns, average telehealth adoption is projected to reach approximately 
    **{final_forecast_value:.2f}%** by **{final_forecast_year}**.
    """)

    with st.expander("View regional forecast table"):
        table_df = forecast_rf if best_model_name == "Random Forest Regressor" else forecast_lr
        table_df = table_df.rename(columns={"forecast": "forecasted_telehealth_rate"})

        st.dataframe(
            table_df[["year", "region", "forecasted_telehealth_rate"]]
            .sort_values(["year", "forecasted_telehealth_rate"], ascending=[True, False]),
            use_container_width=True
        )

    st.caption("""
    ⚠️ Note: This forecast is a prototype decision-support feature. 
    Although the dataset contains many observations, predictions are based on limited historical years and regional patterns. 
    Future work should incorporate higher-frequency data, additional healthcare access variables, and advanced time-series models before operational use.
    """)

else:
    st.warning("""
    Not enough time or regional variation is available for model-based forecasting. 
    Select at least three years and multiple regions.
    """)

# -----------------------------
# LATEST YEAR REGIONAL RANKING
# -----------------------------
st.subheader(f"🏆 Top 10 Regions by Telehealth Adoption — {latest_year}")

top_latest = (
    latest_df.groupby("region")["telehealth_rate"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_latest)

st.subheader(f"⚠️ Lowest 10 Regions by Telehealth Adoption — {latest_year}")

low_latest = (
    latest_df.groupby("region")["telehealth_rate"]
    .mean()
    .sort_values()
    .head(10)
)

st.bar_chart(low_latest)

# -----------------------------
# ADOPTION GAP ANALYSIS
# -----------------------------
st.subheader("📊 Adoption Gap Analysis")

top_avg = top_latest.mean()
low_avg = low_latest.mean()
gap = top_avg - low_avg

st.metric(
    "Adoption Gap: Top 10 vs Lowest 10 Regions",
    f"{gap:.2f} percentage points"
)

st.write(f"""
The average adoption rate among the top 10 regions is **{top_avg:.2f}%**, while the average among the lowest 10 regions is **{low_avg:.2f}%**.  
This gap highlights regional inequality in telehealth adoption and helps institutions identify where access improvement may be needed.
""")

# -----------------------------
# DISTRIBUTION / EQUITY VIEW
# -----------------------------
st.subheader("⚖️ Telehealth Adoption Distribution (Equity View)")

dist_df = latest_df[["region", "telehealth_rate"]].copy()

st.bar_chart(dist_df.set_index("region")["telehealth_rate"])

st.write("""
This equity view shows how telehealth adoption is distributed across regions in the latest selected year.
It helps identify whether adoption is broadly shared or concentrated among a smaller group of higher-performing regions.
""")

# -----------------------------
# AT-RISK REGIONS
# -----------------------------
st.subheader("🚨 At-Risk Regions with Low Average Adoption")

at_risk_regions = low_trend[low_trend < 25].sort_values()

if not at_risk_regions.empty:
    st.bar_chart(at_risk_regions)

    st.write(f"""
    **{len(at_risk_regions)} regions** have an average telehealth adoption rate below 25% across the selected period.
    These regions may require targeted support to improve digital healthcare access.
    """)
else:
    st.success("No regions fall below the 25% average adoption threshold for the selected period.")

# -----------------------------
# REGION TREND COMPARISON
# -----------------------------
st.subheader("🗺️ Regional Telehealth Trend Comparison")

if selected_regions:
    region_trend = (
        filtered_df.groupby(["year", "region"], as_index=False)["telehealth_rate"]
        .mean()
        .sort_values("year")
    )

    region_trend["year"] = region_trend["year"].astype(str)

    region_pivot = region_trend.pivot(
        index="year",
        columns="region",
        values="telehealth_rate"
    )

    st.line_chart(region_pivot)
else:
    st.info("Select one or more regions from the sidebar to compare regional trends.")

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# LIVE MODEL API PREDICTION
# -----------------------------
st.subheader("🔌 Live Model Prediction via API")

st.markdown("""
Generate real-time telehealth forecasts using the deployed FastAPI model.
""")

col1, col2 = st.columns(2)

with col1:
    api_year = st.number_input("Forecast Year", min_value=2025, max_value=2035, value=2027)
    api_region = st.selectbox("Region", sorted(df["region"].unique()))

with col2:
    api_total_enrollment = st.number_input(
        "Total Enrollment",
        min_value=0.0,
        value=1000000.0,
        step=10000.0
    )

    api_telehealth_users = st.number_input(
        "Telehealth Users",
        min_value=0.0,
        value=250000.0,
        step=10000.0
    )

if st.button("🚀 Generate API Forecast"):
    try:
        api_url = "http://127.0.0.1:8000/predict"

        params = {
            "year": int(api_year),
            "region": api_region,
            "total_enrollment": float(api_total_enrollment),
            "telehealth_users": float(api_telehealth_users)
        }

        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            result = response.json()

            if "forecasted_telehealth_rate" in result and "calculated_input_rate" in result:
                st.success("Prediction generated successfully")

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Forecasted Telehealth Rate",
                        f"{result['forecasted_telehealth_rate']:.2f}%"
                    )

                with metric_col2:
                    st.metric(
                        "Calculated Current Rate",
                        f"{result['calculated_input_rate']:.2f}%"
                    )

                st.json(result)

            else:
                st.error("API returned a response, but the expected prediction fields were missing.")
                st.json(result)

        else:
            st.error(f"API error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("API connection failed. Make sure FastAPI is running.")
        st.write(str(e))
