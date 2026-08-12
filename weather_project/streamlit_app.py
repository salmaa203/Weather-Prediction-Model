import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ==============================
       BACKGROUND
    ============================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #e0f2fe 0%,
            #f8fafc 100%
        );
    }


    /* ==============================
       MAIN CONTENT WIDTH
    ============================== */

    .block-container {
        max-width: 1400px;
        padding-top: 35px;
        padding-bottom: 50px;
    }


    /* ==============================
       HEADER
    ============================== */

    .main-title {
        text-align: center;
        color: #0f172a;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 20px;
        margin-bottom: 45px;
    }


    /* ==============================
       CARD
    ============================== */

    .card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
    }


    /* ==============================
       SECTION TITLE
    ============================== */

    .section-title {
        text-align: center;
        color: #0f172a;
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .section-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 18px;
        margin-bottom: 25px;
    }


    /* ==============================
       INPUT LABEL
    ============================== */

    .input-label {
        color: #1e293b;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }


    /* ==============================
       NUMBER INPUT
    ============================== */

    div[data-testid="stNumberInput"] {
        margin-bottom: 5px;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
        font-size: 18px;
    }


    /* ==============================
       BUTTON
    ============================== */

    div[data-testid="stButton"] {
        width: 100%;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        height: 58px;
        border-radius: 12px;
        background-color: #2563eb;
        color: white;
        border: none;
        font-size: 19px;
        font-weight: 700;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #1d4ed8;
        color: white;
    }


    /* ==============================
       TABLE
    ============================== */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* ==============================
       MOBILE
    ============================== */

    @media (max-width: 600px) {

        .main-title {
            font-size: 32px;
        }

        .main-subtitle {
            font-size: 16px;
        }

        .section-title {
            font-size: 24px;
        }

        .section-subtitle {
            font-size: 16px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🌤️ Weather Forecast
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
        Predict future temperature using a SARIMAX time-series forecasting model.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "predictor"
    / "final_weather_model.pkl"
)

DATA_PATH = (
    BASE_DIR
    / "weather_cleaned.csv"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    df = df.set_index("date")

    df = df.asfreq("D")

    return df


# =========================================================
# FOURIER FEATURES
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
# LOAD MODEL + DATA
# =========================================================

try:

    model = load_model()

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load model or dataset: {e}"
    )

    st.stop()


# =========================================================
# INPUT CARD
# =========================================================

st.markdown(
    """
    <div class="card">
    """,
    unsafe_allow_html=True
)

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

generate = st.button(
    "Generate Forecast"
)

st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FORECAST
# =========================================================

if generate:

    try:

        # -------------------------------------------------
        # FUTURE DATES
        # -------------------------------------------------

        future_index = pd.date_range(
            start=df.index.max() + pd.Timedelta(days=1),
            periods=int(forecast_days),
            freq="D"
        )


        # -------------------------------------------------
        # FOURIER FEATURES
        # -------------------------------------------------

        fourier_future = create_fourier_features(
            future_index,
            period=365.25,
            K=5
        )


        # -------------------------------------------------
        # MODEL FORECAST
        # -------------------------------------------------

        future_forecast = model.get_forecast(
            steps=int(forecast_days),
            exog=fourier_future
        )


        future_pred = (
            future_forecast.predicted_mean
        )

        future_ci = (
            future_forecast.conf_int()
        )


        # -------------------------------------------------
        # RESULTS DATAFRAME
        # -------------------------------------------------

        forecast_df = pd.DataFrame({

            "Date": future_index,

            "Forecast Temperature":
                future_pred.values,

            "Lower 95% CI":
                future_ci.iloc[:, 0].values,

            "Upper 95% CI":
                future_ci.iloc[:, 1].values

        })


        # -------------------------------------------------
        # ROUND VALUES
        # -------------------------------------------------

        forecast_df[
            "Forecast Temperature"
        ] = forecast_df[
            "Forecast Temperature"
        ].round(2)

        forecast_df[
            "Lower 95% CI"
        ] = forecast_df[
            "Lower 95% CI"
        ].round(2)

        forecast_df[
            "Upper 95% CI"
        ] = forecast_df[
            "Upper 95% CI"
        ].round(2)


        # =================================================
        # RESULTS
        # =================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-title">
                📊 Temperature Forecast
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-subtitle">
                Forecast with 95% confidence intervals
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------

        display_df = forecast_df.copy()

        display_df["Date"] = (
            display_df["Date"]
            .dt.strftime("%Y-%m-%d")
        )

        display_df[
            "Forecast Temperature"
        ] = display_df[
            "Forecast Temperature"
        ].map(
            lambda x: f"{x:.2f} °C"
        )

        display_df[
            "Lower 95% CI"
        ] = display_df[
            "Lower 95% CI"
        ].map(
            lambda x: f"{x:.2f} °C"
        )

        display_df[
            "Upper 95% CI"
        ] = display_df[
            "Upper 95% CI"
        ].map(
            lambda x: f"{x:.2f} °C"
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # CHART
        # =================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-title">
                Temperature Forecast Trend
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-subtitle">
                Forecast temperature with 95% confidence interval
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # PLOTLY
        # -------------------------------------------------

        chart = go.Figure()


        # Lower CI
        chart.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Lower 95% CI"],
                mode="lines",
                name="Lower 95% CI",
                line=dict(
                    color="rgba(100,116,139,0.5)",
                    dash="dash",
                    width=1
                )
            )
        )


        # Upper CI
        chart.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Upper 95% CI"],
                mode="lines",
                name="Upper 95% CI",
                line=dict(
                    color="rgba(100,116,139,0.5)",
                    dash="dash",
                    width=1
                ),
                fill="tonexty",
                fillcolor="rgba(37,99,235,0.12)"
            )
        )


        # Forecast
        chart.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast Temperature"],
                mode="lines+markers",
                name="Forecast Temperature",
                line=dict(
                    color="#2563eb",
                    width=3
                ),
                marker=dict(
                    color="#2563eb",
                    size=7
                )
            )
        )


        # -------------------------------------------------
        # CHART STYLE
        # -------------------------------------------------

        chart.update_layout(

            height=520,

            template="plotly_white",

            paper_bgcolor="white",

            plot_bgcolor="white",

            margin=dict(
                l=60,
                r=30,
                t=20,
                b=80
            ),

            xaxis=dict(
                title="Date",
                tickangle=-45,
                showgrid=True,
                gridcolor="rgba(148,163,184,0.25)"
            ),

            yaxis=dict(
                title="Temperature (°C)",
                showgrid=True,
                gridcolor="rgba(148,163,184,0.25)"
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),

            hovermode="x unified"
        )


        st.plotly_chart(
            chart,
            use_container_width=True
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"Error while generating forecast: {e}"
        )
