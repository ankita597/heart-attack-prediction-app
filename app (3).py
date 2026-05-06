import pickle
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="centered"
)

# Load model
model = pickle.load(open(os.path.join(os.path.dirname(__file__), 'rf_model.pkl'), 'rb'))

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0f1e;
    color: #e8eaf0;
}

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1a2e 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem; max-width: 760px; margin: auto; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 24px rgba(255,90,90,0.6));
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 24px rgba(255,90,90,0.5)); }
    50% { transform: scale(1.06); filter: drop-shadow(0 0 36px rgba(255,90,90,0.9)); }
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 400;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero h1 span { color: #ff5a5a; font-style: italic; }
.hero p {
    font-size: 1rem;
    color: #8892a4;
    font-weight: 300;
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #ff5a5a;
    margin: 2.2rem 0 1rem;
    padding-left: 2px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,90,90,0.3), transparent);
}

/* ── Card wrapper ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(255,90,90,0.25); }

/* ── Streamlit widget overrides ── */
label, .stSelectbox label, .stNumberInput label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #a0aab8 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 4px !important;
}

input[type="number"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.5rem 0.8rem !important;
    transition: border-color 0.2s !important;
}
input[type="number"]:focus {
    border-color: #ff5a5a !important;
    box-shadow: 0 0 0 3px rgba(255,90,90,0.12) !important;
    outline: none !important;
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(255,90,90,0.4) !important;
}

/* Dropdown menu */
ul[data-baseweb="menu"] {
    background: #131c2e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}
li[role="option"]:hover {
    background: rgba(255,90,90,0.1) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff3d3d, #ff6b6b) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(255,61,61,0.35) !important;
    margin-top: 1.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,61,61,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result boxes ── */
.result-high {
    background: linear-gradient(135deg, rgba(255,50,50,0.12), rgba(255,100,100,0.06));
    border: 1px solid rgba(255,80,80,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeUp 0.5s ease;
}
.result-low {
    background: linear-gradient(135deg, rgba(50,210,120,0.1), rgba(50,210,120,0.04));
    border: 1px solid rgba(50,210,120,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeUp 0.5s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; display: block; }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 400;
    margin-bottom: 0.4rem;
}
.result-sub { font-size: 0.9rem; color: #8892a4; font-weight: 300; }

/* ── Divider ── */
hr.custom {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 2rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #3d4a5c;
    margin-top: 3rem;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🫀</span>
    <h1>Cardio<span>Risk</span> AI</h1>
    <p>Enter your clinical parameters below to assess your cardiovascular risk using our trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Vitals ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Vitals</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    Age = st.number_input('Age', min_value=20, max_value=100, value=45)
with col2:
    gender = st.selectbox('Sex', ('M', 'F'))
with col3:
    FastingBS = st.selectbox('Fasting Blood Sugar > 120', (0, 1))

col4, col5, col6 = st.columns(3)
with col4:
    RestingBP = st.number_input('Resting BP (mmHg)', min_value=0, max_value=300, value=120)
with col5:
    Cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=0, max_value=600, value=200)
with col6:
    MaxHR = st.number_input('Max Heart Rate', min_value=60, max_value=600, value=150)

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Clinical Findings ─────────────────────────────────────────────
st.markdown('<div class="section-label">Clinical Findings</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    ChestPainType = st.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
with col8:
    RestingECG = st.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))

col9, col10, col11 = st.columns(3)
with col9:
    ExerciseAngina = st.selectbox('Exercise Angina', ('N', 'Y'))
with col10:
    ST_Slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
with col11:
    Oldpeak = st.number_input('Oldpeak', min_value=-3, max_value=10, value=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Encoding ─────────────────────────────────────────────────────────────────
Exercise_Angina = 1 if ExerciseAngina == 'Y' else 0
Sex_F = 1 if gender == 'F' else 0
Sex_M = 1 if gender == 'M' else 0
ChestPainType_dict = {'ASY': 3, 'NAP': 2, 'ATA': 1, 'TA': 0}
Chest_PainType = ChestPainType_dict[ChestPainType]
Resting_ECG_dict = {'Normal': 0, 'LVH': 1, 'ST': 2}
RestingECG_enc = Resting_ECG_dict[RestingECG]
ST_Slope_dict = {'Down': 0, 'Up': 1, 'Flat': 2}
st_Slope = ST_Slope_dict[ST_Slope]

input_features = pd.DataFrame({
    'Age': [Age], 'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS], 'MaxHR': [MaxHR], 'Oldpeak': [Oldpeak],
    'Exercise_Angina': [Exercise_Angina],
    'Sex_F': [Sex_F], 'Sex_M': [Sex_M],
    'Chest_PainType': [Chest_PainType],
    'Resting_ECG': [RestingECG_enc],
    'st_Slope': [st_Slope]
})

scaler = StandardScaler()
input_features[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']] = scaler.fit_transform(
    input_features[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
)

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button('Analyse Risk →'):
    predictions = model.predict(input_features)
    if predictions == 1:
        st.markdown("""
        <div class="result-high">
            <span class="result-icon">⚠️</span>
            <div class="result-title" style="color:#ff6b6b;">High Cardiac Risk Detected</div>
            <div class="result-sub">Please consult a cardiologist for a comprehensive evaluation.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-low">
            <span class="result-icon">✅</span>
            <div class="result-title" style="color:#32d278;">Low Cardiac Risk</div>
            <div class="result-sub">Your parameters indicate a lower cardiovascular risk. Keep up the healthy habits!</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="custom">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    CardioRisk AI · For informational purposes only · Not a substitute for medical advice
</div>
""", unsafe_allow_html=True)
