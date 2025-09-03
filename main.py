# =============================
# Futuristic Sci-Fi Breast Cancer Prediction App
# =============================

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px

# -----------------------------
# 1️⃣ Load trained pipeline
# -----------------------------
pipeline = joblib.load("breast_cancer_pipeline.pkl")

# -----------------------------
# 2️⃣ Page Config
# -----------------------------
st.set_page_config(page_title="Futuristic Breast Cancer Prediction", page_icon="🩺", layout="wide")

# -----------------------------
# 3️⃣ Custom CSS for Sci-Fi Cards & Particles
# -----------------------------
st.markdown("""
<style>
/* Background particle animation */
.stApp {
    background: #0f0c29;
    background: radial-gradient(circle at 50% 50%, #0f0c29, #302b63, #24243e);
    font-family: 'Segoe UI', sans-serif;
    color: #fff;
    overflow-x: hidden;
}

/* Sidebar style */
[data-testid="stSidebar"]{
    background: linear-gradient(to bottom, #ff6a00, #ee0979);
    color: #fff;
    border-radius: 15px;
    padding: 15px;
}

/* 3D card container */
.card {
    perspective: 1500px;
    margin: 20px 0;
}
.card-inner {
    position: relative;
    width: 100%;
    height: 250px; /* Fixed height for all cards */
    transition: transform 0.8s;
    transform-style: preserve-3d;
}
.card:hover .card-inner {
    transform: rotateY(180deg) scale(1.05);
}
.card-front, .card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: 20px;
    backface-visibility: hidden;
    padding: 20px;
    box-shadow: 0 0 15px rgba(255,255,255,0.3);
    border: 2px solid transparent;
    animation: glow 2s infinite alternate;
}
.card-front {
    background: rgba(20, 20, 40, 0.9);
}
.card-back {
    background: rgba(60, 10, 30, 0.95);
    transform: rotateY(180deg);
}

/* Glow animation */
@keyframes glow {
    0% { box-shadow: 0 0 15px #ff0080, 0 0 30px #ff0080; }
    50% { box-shadow: 0 0 25px #ff00ff, 0 0 50px #ff00ff; }
    100% { box-shadow: 0 0 15px #ff0080, 0 0 30px #ff0080; }
}

/* Headings and text */
h2, h3 {
    color: #ffffff;
    font-weight: bold;
}
p {
    color: #ffffff;
    font-weight: bold;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    background: linear-gradient(to right, #00f260, #0575e6);
    color: #fff;
    font-weight: bold;
    padding: 0.6em 1.2em;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    box-shadow: 0 0 15px #00f260, 0 0 30px #0575e6;
    transform: scale(1.08);
}

/* Links */
a {
    color:#00ffcc;
    font-weight:bold;
    text-decoration:none;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 4️⃣ Sidebar: Patient Input
# -----------------------------
with st.sidebar:
    st.title("🚀 Enter Patient Features")
    feature_names = pipeline.named_steps['scaler'].feature_names_in_
    user_input = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, format="%.5f")
        user_input.append(val)
    predict_clicked = st.button("Predict Now")

# -----------------------------
# 5️⃣ Prediction Logic
# -----------------------------
if predict_clicked:
    input_array = np.array(user_input).reshape(1, -1)
    prediction = pipeline.predict(input_array)[0]
    prediction_proba = pipeline.predict_proba(input_array)[0]

    col1, col2 = st.columns([1,1])  # Equal column widths for equal card size

    # ----- 3D Prediction Card -----
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-inner">
                <div class="card-front">
                    <h2>Prediction Result</h2>
                    <p>{"Malignant ⚠️" if prediction==1 else "Benign ✅"}</p>
                </div>
                <div class="card-back">
                    <img src='{"https://static.vecteezy.com/system/resources/previews/010/400/390/non_2x/information-risk-icon-healthcare-and-medical-icon-vector.jpg" if prediction==1 else "https://img.freepik.com/premium-vector/medical-health-icon-medical-danger-save-plus-icon_786080-27.jpg"}' width='150'>
                    <p>Click to flip</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----- 3D Risk Alert Card -----
    with col2:
        risk_text = ("High Risk ⚠️" if prediction_proba[1]>=0.8 else
                     "Moderate Risk ⚠️" if prediction_proba[1]>=0.5 else
                     "Low Risk ✅")
        st.markdown(f"""
        <div class="card">
            <div class="card-inner">
                <div class="card-front">
                    <h2>Risk Alert</h2>
                    <p>{risk_text}</p>
                </div>
                <div class="card-back">
                    <p>Malignant Probability: {prediction_proba[1]*100:.2f}%</p>
                    <p>Click to flip</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----- Probability Bar Chart -----
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Type": ["Benign", "Malignant"],
        "Probability": [prediction_proba[0]*100, prediction_proba[1]*100]
    })
    fig_prob = px.bar(prob_df, x='Type', y='Probability', text='Probability', color='Type',
                      color_discrete_map={'Benign':'green', 'Malignant':'red'})
    fig_prob.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#ffffff'))
    st.plotly_chart(fig_prob, use_container_width=True)

    # ----- Feature Importance -----
    st.subheader("Top Feature Importance")
    if hasattr(pipeline.named_steps['classifier'], "coef_"):
        coefs = pipeline.named_steps['classifier'].coef_[0]
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": coefs * input_array[0]
        })
        importance_df['AbsImportance'] = importance_df['Importance'].abs()
        importance_df = importance_df.sort_values(by='AbsImportance', ascending=False).head(10)
        fig_imp = px.bar(importance_df, x='Feature', y='Importance', text='Importance',
                         color='Importance', color_continuous_scale='Turbo')
        fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#ffffff'))
        st.plotly_chart(fig_imp, use_container_width=True)

# ----- Awareness Section -----
st.markdown("---")
st.markdown("<h2 style='color:#ffffff;'>Breast Cancer Awareness</h2>", unsafe_allow_html=True)
st.image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41392-024-02108-4/MediaObjects/41392_2024_2108_Fig1_HTML.png?as=webp", width=600)
st.markdown("<p style='color:#ffffff; font-weight:bold;'>Learn More About Breast Cancer</p>", unsafe_allow_html=True)
st.markdown("[Click Here](https://www.cancer.org/cancer/breast-cancer.html)", unsafe_allow_html=True)
