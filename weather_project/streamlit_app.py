import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #e0f2fe 0%,
            #f8fafc 100%
        );
    }

    /* Remove Streamlit default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 20px;
        color: #64748b;
        margin-bottom: 45px;
    }

    /* Cards */
    .custom-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
    }

    /* Section title */
    .section-title {
        text-align: center;
        font-size: 30px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .section-subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px;
        font-size: 18px;
        font-weight: 700;
        height: 55px;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* Input label */
    label {
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Temperature result */
    .temperature-result {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #2563eb;
        margin: 15px 0;
    }

    /* Responsive */
    @media (max-width: 700px) {

        .main-title {
            font-size: 36px;
        }

        .main-subtitle {
            font-size: 16px;
        }

        .section-title {
            font-size: 24px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "predictor" / "final_weather_model.pkl"

DATA_PATH = BASE_DIR / "weather_cleaned.csv"


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: {MODEL_PATH}"
        )
        st.stop()

    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# Load Weather Data
# =========================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found: {DATA_PATH}"
        )
        st.stop()

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        index_col="date"
    )

    df = df.asfreq("D")

    return df


df = load_data()


# =========================================================
# Fourier Features
# =========================================================

def create_fourier_features(
    index,
    period=365.25,
    K=5
):

    t = np.arange(len(index))

    features = {}

    for k in range(1, K + 1):

        features[f"sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )

        features[f"cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    return pd.DataFrame(
        features,
        index=index
    )


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="main-title">🌤️ Weather Forecast</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
        Predict future temperature using a SARIMAX
        time-series forecasting model.
    </div>
    """,
    unsafe_allow_html=True
)
# =========================================================
# INPUT
# =========================================================

st.markdown(
    """
    <div class="input-label">
        Forecast Horizon (Days)
    </div>
    """,
    unsafe_allow_html=True
)

forecast_days = st.number_input(
    "Forecast Horizon",
    min_value=1,
    max_value=90,
    value=30,
    step=1,
    label_visibility="collapsed"
)

st.markdown(
    """
    <div style="
        color:#64748b;
        font-size:16px;
        margin-top:4px;
        margin-bottom:20px;
    ">
        Enter a value between 1 and 90 days.
    </div>
    """,
    unsafe_allow_html=True
)

generate = st.button("Generate Forecast")

# =========================================================
# Generate Forecast
# =========================================================

if generate:

    try:

        forecast_days = int(forecast_days)

        # -------------------------------------------------
        # Future Dates
        # -------------------------------------------------

        future_index = pd.date_range(
            start=df.index.max() + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        # -------------------------------------------------
        # Fourier Features
        # -------------------------------------------------

        fourier_future = create_fourier_features(
            future_index,
            period=365.25,
            K=5
        )

        # -------------------------------------------------
        # Forecast
        # -------------------------------------------------

        future_forecast = model.get_forecast(
            steps=forecast_days,
            exog=fourier_future
        )

        future_pred = future_forecast.predicted_mean

        future_ci = future_forecast.conf_int()

        # =================================================
        # Results DataFrame
        # =================================================

        results = pd.DataFrame({

            "Date": future_index,

            "Forecast Temperature": future_pred.values,

            "Lower 95% CI": future_ci.iloc[:, 0].values,

            "Upper 95% CI": future_ci.iloc[:, 1].values

        })

        # Round values

        results["Forecast Temperature"] = (
            results["Forecast Temperature"]
            .round(2)
        )

        results["Lower 95% CI"] = (
            results["Lower 95% CI"]
            .round(2)
        )

        results["Upper 95% CI"] = (
            results["Upper 95% CI"]
            .round(2)
        )

        # =================================================
        # Table Section
        # =================================================

        st.markdown(
            '<div class="custom-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="section-title">
                📊 {forecast_days}-Day Temperature Forecast
            </div>

            <div class="section-subtitle">
                Forecast with 95% confidence intervals
            </div>
            """,
            unsafe_allow_html=True
        )

        # Format date

        display_table = results.copy()

        display_table["Date"] = (
            display_table["Date"]
            .dt.strftime("%Y-%m-%d")
        )

        display_table["Forecast Temperature"] = (
            display_table["Forecast Temperature"]
            .astype(str)
            + " °C"
        )

        display_table["Lower 95% CI"] = (
            display_table["Lower 95% CI"]
            .astype(str)
            + " °C"
        )

        display_table["Upper 95% CI"] = (
            display_table["Upper 95% CI"]
            .astype(str)
            + " °C"
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # =================================================
        # Chart Section
        # =================================================

        st.markdown(
            '<div class="custom-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-title">
                📈 Temperature Forecast Trend
            </div>

            <div class="section-subtitle">
                Forecast temperature with 95% confidence interval
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # Plotly Chart
        # -------------------------------------------------

        fig = go.Figure()

        # Lower CI

        fig.add_trace(
            go.Scatter(
                x=results["Date"],
                y=results["Lower 95% CI"],
                mode="lines",
                line=dict(
                    color="rgba(37, 99, 235, 0.35)",
                    dash="dash",
                    width=1
                ),
                name="Lower 95% CI"
            )
        )

        # Upper CI with fill

        fig.add_trace(
            go.Scatter(
                x=results["Date"],
                y=results["Upper 95% CI"],
                mode="lines",
                line=dict(
                    color="rgba(37, 99, 235, 0.35)",
                    dash="dash",
                    width=1
                ),
                fill="tonexty",
                fillcolor="rgba(37, 99, 235, 0.12)",
                name="Upper 95% CI"
            )
        )

        # Forecast

        fig.add_trace(
            go.Scatter(
                x=results["Date"],
                y=results["Forecast Temperature"],
                mode="lines+markers",
                line=dict(
                    color="#2563eb",
                    width=3
                ),
                marker=dict(
                    size=7,
                    color="#2563eb"
                ),
                name="Forecast Temperature"
            )
        )

        # Chart layout

        fig.update_layout(

            height=500,

            plot_bgcolor="white",

            paper_bgcolor="white",

            hovermode="x unified",

            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="#e2e8f0"
            ),

            yaxis=dict(
                title="Temperature (°C)",
                showgrid=True,
                gridcolor="#e2e8f0"
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),

            margin=dict(
                l=50,
                r=30,
                t=50,
                b=50
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"Unable to generate forecast: {str(e)}"
        )
