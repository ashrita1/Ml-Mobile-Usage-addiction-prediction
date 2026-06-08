# ============================================================
#   app.py  —  Mobile Addiction Prediction
#   Run : streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Mobile Addiction Predictor",
    page_icon  = "📱",
    layout     = "centered"
)

# ── Load Model (single pipeline — preprocessor + GB model combined) ──
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("model.pkl")
    return pipeline

pipeline = load_artifacts()

# ── Header ───────────────────────────────────────────────────
st.title("📱 Mobile Addiction Level Predictor")
st.markdown("Fill in the details below to predict the **mobile addiction level** of a user.")
st.divider()

# ── Input Form ───────────────────────────────────────────────
st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)

with col1:
    age    = st.number_input("Age",    min_value=10, max_value=80, value=22, step=1)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

phone_usage_purpose = st.selectbox(
    "Phone Usage Purpose",
    ["Social Media", "Gaming", "Education", "Browsing", "Communication", "Entertainment"]
)

st.divider()
st.subheader("📊 Usage Behaviour")

col3, col4 = st.columns(2)
with col3:
    daily_usage_hours      = st.slider("Daily Usage Hours",                0.0, 16.0, 6.0, 0.5)
    phone_checks_per_day   = st.slider("Phone Checks Per Day",             0,   200,  60,  1)
    apps_used_daily        = st.slider("Apps Used Daily",                  1,   30,   10,  1)
    screen_time_before_bed = st.slider("Screen Time Before Bed (hrs)",     0.0, 5.0,  1.0, 0.5)
with col4:
    time_on_social_media   = st.slider("Time on Social Media (hrs)",       0.0, 10.0, 2.5, 0.5)
    time_on_gaming         = st.slider("Time on Gaming (hrs)",             0.0, 10.0, 1.0, 0.5)
    time_on_education      = st.slider("Time on Education (hrs)",          0.0, 10.0, 0.5, 0.5)
    weekend_usage_hours    = st.slider("Weekend Usage Hours",              0.0, 16.0, 7.0, 0.5)

st.divider()
st.subheader("🧠 Mental Health & Lifestyle")

col5, col6 = st.columns(2)
with col5:
    anxiety_level             = st.slider("Anxiety Level (1–10)",             1, 10, 5)
    depression_level          = st.slider("Depression Level (1–10)",          1, 10, 5)
    self_esteem               = st.slider("Self Esteem (1–10)",               1, 10, 5)
    interllectual_performance = st.slider("Intellectual Performance (1–10)",  1, 10, 5)
with col6:
    sleep_hours               = st.slider("Sleep Hours",                      2.0, 12.0, 7.0, 0.5)
    exercise_hours            = st.slider("Exercise Hours Per Day",           0.0,  5.0, 1.0, 0.5)
    social_interactions       = st.slider("Social Interactions (daily)",      0,   20,   5,   1)
    family_communication      = st.slider("Family Communication (1–10)",      1,   10,   5,   1)

st.divider()

# ── Predict Button ────────────────────────────────────────────
if st.button("🔍 Predict Addiction Level", use_container_width=True):

    # Build input dataframe — columns must match training data exactly
    input_data = pd.DataFrame([{
        'age'                       : age,
        'gender'                    : gender,
        'daily_usage_hours'         : daily_usage_hours,
        'phone_checks_per_day'      : phone_checks_per_day,
        'apps_used_daily'           : apps_used_daily,
        'screen_time_before_bed'    : screen_time_before_bed,
        'time_on_social_media'      : time_on_social_media,
        'time_on_gaming'            : time_on_gaming,
        'time_on_education'         : time_on_education,
        'anxiety_level'             : anxiety_level,
        'depression_level'          : depression_level,
        'self_esteem'               : self_esteem,
        'sleep_hours'               : sleep_hours,
        'exercise_hours'            : exercise_hours,
        'social_interactions'       : social_interactions,
        'family_communication'      : family_communication,
        'interllectual_performance' : interllectual_performance,
        'weekend_usage_hours'       : weekend_usage_hours,
        'phone_usage_purpose'       : phone_usage_purpose
    }])

    # ── Predict (pipeline handles preprocessing internally) ───
    prediction = pipeline.predict(input_data)[0]
    prediction = round(float(prediction), 2)

    # ── Result Display ────────────────────────────────────────
    st.subheader("📈 Prediction Result")

    if prediction <= 3:
        level  = "Low Addiction"
        color  = "green"
        emoji  = "✅"
        advice = "Great! Your mobile usage appears healthy. Keep maintaining a balanced lifestyle."
    elif prediction <= 6:
        level  = "Moderate Addiction"
        color  = "orange"
        emoji  = "⚠️"
        advice = "You show moderate addiction signs. Try reducing screen time, especially before bed."
    else:
        level  = "High Addiction"
        color  = "red"
        emoji  = "🚨"
        advice = "High addiction level detected. Consider digital detox and seeking professional advice."

    st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6;
            border-left: 6px solid {color};
            padding: 20px;
            border-radius: 8px;
            margin-top: 10px;">
            <h2 style="color:{color}; margin:0;">{emoji} {level}</h2>
            <h1 style="margin:8px 0; color:#333;">Addiction Score: {prediction} / 10</h1>
            <p style="color:#555; margin:0;">{advice}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Progress bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Addiction Level Scale**")
    st.progress(min(prediction / 10.0, 1.0))
    st.caption(f"Score {prediction} out of 10")

    # ── Input Summary Table ───────────────────────────────────
    with st.expander("📋 View Input Summary"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Mobile Addiction Prediction  |  Gradient Boosting Model  |  Sprint 3 Final Build")
