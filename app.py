# -*- coding: utf-8 -*-
"""
Car Insurance Fraud Prediction System
Production-Quality Streamlit Frontend
Author: Antigravity AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Car Insurance Fraud Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark SaaS Interface (Minimal, Modern, Clean)
st.markdown("""
<style>
/* Base Styles & Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* App Background Gradient */
div[data-testid="stAppViewContainer"] {
    background-color: #080c14;
    background-image: 
        radial-gradient(at 0% 0%, rgba(17, 24, 39, 0.95) 0, transparent 50%), 
        radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.05) 0, transparent 40%);
    background-attachment: fixed;
    color: #f8fafc;
}

/* Container Padding */
div[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* Sidebar Custom Styling */
div[data-testid="stSidebar"] {
    background-color: rgba(9, 13, 22, 0.98) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
}

/* Hide Default Header/Footer elements */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}
div[data-testid="stHeader"] {
    background: rgba(0,0,0,0); 
    height: 0rem;
    display: none;
}
footer {
    visibility: hidden;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
    transition: all 0.2s ease-in-out;
}
.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.2);
    box-shadow: 0 6px 24px 0 rgba(99, 102, 241, 0.04);
    transform: translateY(-2px);
}

/* Hero Typography */
.hero-title {
    font-size: 62px;
    font-weight: 800;
    margin: 0 auto 18px auto;
    color: #ffffff;
    letter-spacing: -1.5px;
    line-height: 1.1;
    text-align: center;
    max-width: 800px;
}
.fraud-gradient {
    background: linear-gradient(135deg, #3b82f6 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.hero-subtitle {
    font-size: 20px;
    color: #B8BDC7;
    max-width: 650px;
    margin: 0 auto 14px auto;
    text-align: center;
    font-weight: 400;
    line-height: 1.5;
}
.pill-badge {
    background: rgba(99, 102, 241, 0.06);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 9999px;
    margin-bottom: 20px;
    display: inline-block;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.08);
}

/* Primary Button (Blue Gradient) */
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.2s ease !important;
    padding: 10px 20px !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

/* Secondary Button (Outlined) */
button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    padding: 10px 20px !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #ffffff !important;
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.02) !important;
    transform: translateY(-1px) !important;
}

/* Custom KPI Metric Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 30px;
}
.kpi-card {
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 16px;
    text-align: left;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}
.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.25);
    background: rgba(255, 255, 255, 0.03);
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 4px;
}
.kpi-label {
    color: #64748b;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

/* Styled Form Section Headers */
.form-section-header {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    margin-top: 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    padding-bottom: 6px;
}
.form-section-number {
    font-size: 9px;
    font-weight: 700;
    color: #6366f1;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 4px;
    padding: 1px 5px;
}

/* Result Cards */
.result-card {
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-low {
    background: rgba(16, 185, 129, 0.08);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.badge-medium {
    background: rgba(245, 158, 11, 0.08);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.2);
}
.badge-high {
    background: rgba(239, 68, 68, 0.08);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

/* Custom Table Theme */
div[data-testid="stTable"] table {
    background-color: rgba(255, 255, 255, 0.01) !important;
    color: #cbd5e1 !important;
    border-collapse: collapse !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
}
div[data-testid="stTable"] th {
    background-color: rgba(255, 255, 255, 0.02) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 10px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}
div[data-testid="stTable"] td {
    padding: 10px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
    margin-top: 24px;
    margin-bottom: 14px;
    letter-spacing: -0.3px;
}

.tech-badge {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    color: #cbd5e1;
    display: inline-block;
}

/* Background glows and blur circles */
.glow-bg {
    position: absolute;
    width: 450px;
    height: 450px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, rgba(99, 102, 241, 0) 70%);
    top: -10%;
    left: 25%;
    pointer-events: none;
    z-index: -1;
    filter: blur(60px);
}
.blur-circle-1 {
    position: absolute;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.06) 0%, rgba(59, 130, 246, 0) 70%);
    top: 20%;
    left: 10%;
    pointer-events: none;
    z-index: -1;
    filter: blur(50px);
    animation: float-slow 12s ease-in-out infinite alternate;
}
.blur-circle-2 {
    position: absolute;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.06) 0%, rgba(99, 102, 241, 0) 70%);
    bottom: 15%;
    right: 15%;
    pointer-events: none;
    z-index: -1;
    filter: blur(50px);
    animation: float-slow 18s ease-in-out infinite alternate-reverse;
}
@keyframes float-slow {
    0% { transform: translateY(0px) translateX(0px); }
    100% { transform: translateY(15px) translateX(10px); }
}
/* Center and Round Illustration */
div[data-testid="stImage"] img {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHE UTILITIES
# ==========================================

@st.cache_resource
def load_model():
    """Load the model pipeline."""
    model_path = "insurance_fraud_model.pkl"
    if not os.path.exists(model_path):
        return f"Model file '{model_path}' not found."
    try:
        return joblib.load(model_path)
    except Exception as e:
        return f"Error loading model: {str(e)}"


@st.cache_data
def load_claims_data():
    """Load claims dataset."""
    data_path = "insurance_claims.csv"
    if not os.path.exists(data_path):
        return None
    try:
        df = pd.read_csv(data_path)
        if '_c39' in df.columns:
            df = df.drop(columns=['_c39'])
        return df
    except Exception:
        return None


# Load model and dataset
model = load_model()
df_claims = load_claims_data()


# ==========================================
# PLOTLY THEME APPLYER
# ==========================================

def customize_plotly_layout(fig):
    """Applies a clean dark theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#94a3b8", 'family': "Plus Jakarta Sans, sans-serif"},
        title={'font': {'color': '#ffffff', 'size': 14, 'family': 'Plus Jakarta Sans'}},
        legend={'font': {'color': '#cbd5e1', 'size': 10}},
        xaxis={'gridcolor': 'rgba(255, 255, 255, 0.02)', 'linecolor': 'rgba(255, 255, 255, 0.04)', 'zerolinecolor': 'rgba(255, 255, 255, 0.04)'},
        yaxis={'gridcolor': 'rgba(255, 255, 255, 0.02)', 'linecolor': 'rgba(255, 255, 255, 0.04)', 'zerolinecolor': 'rgba(255, 255, 255, 0.04)'},
        margin=dict(l=15, r=15, t=40, b=15)
    )
    return fig


# ==========================================
# PRESETS FOR PAGE 2
# ==========================================

CATEGORIES = {
    'policy_state': ['IL', 'IN', 'OH'],
    'policy_csl': ['100/300', '250/500', '500/1000'],
    'insured_sex': ['FEMALE', 'MALE'],
    'insured_education_level': ['Associate', 'College', 'High School', 'JD', 'MD', 'Masters', 'PhD'],
    'insured_occupation': ['adm-clerical', 'armed-forces', 'craft-repair', 'exec-managerial', 'farming-fishing', 'handlers-cleaners', 'machine-op-inspct', 'other-service', 'priv-house-serv', 'prof-specialty', 'protective-serv', 'sales', 'tech-support', 'transport-moving'],
    'insured_hobbies': ['base-jumping', 'basketball', 'board-games', 'bungie-jumping', 'camping', 'chess', 'cross-fit', 'dancing', 'exercise', 'golf', 'hiking', 'kayaking', 'movies', 'paintball', 'polo', 'reading', 'skydiving', 'sleeping', 'video-games', 'yachting'],
    'insured_relationship': ['husband', 'not-in-family', 'other-relative', 'own-child', 'unmarried', 'wife'],
    'incident_type': ['Multi-vehicle Collision', 'Parked Car', 'Single Vehicle Collision', 'Vehicle Theft'],
    'collision_type': ['?', 'Front Collision', 'Rear Collision', 'Side Collision'],
    'incident_severity': ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'],
    'authorities_contacted': ['Ambulance', 'Fire', 'Other', 'Police'],
    'incident_state': ['NC', 'NY', 'OH', 'PA', 'SC', 'VA', 'WV'],
    'incident_city': ['Arlington', 'Columbus', 'Hillsdale', 'Northbend', 'Northbrook', 'Riverwood', 'Springfield'],
    'property_damage': ['?', 'NO', 'YES'],
    'police_report_available': ['?', 'NO', 'YES'],
    'auto_make': ['Accura', 'Audi', 'BMW', 'Chevrolet', 'Dodge', 'Ford', 'Honda', 'Jeep', 'Mercedes', 'Nissan', 'Saab', 'Suburu', 'Toyota', 'Volkswagen'],
    'auto_model': ['3 Series', '92x', '93', '95', 'A3', 'A5', 'Accord', 'C300', 'CRV', 'Camry', 'Civic', 'Corolla', 'E400', 'Escape', 'F150', 'Forrestor', 'Fusion', 'Grand Cherokee', 'Highlander', 'Impreza', 'Jetta', 'Legacy', 'M5', 'MDX', 'ML350', 'Malibu', 'Maxima', 'Neon', 'Passat', 'Pathfinder', 'RAM', 'RSX', 'Silverado', 'TL', 'Tahoe', 'Ultima', 'Wrangler', 'X5', 'X6']
}

BLANK_DEFAULTS = {
    'months_as_customer': 0, 'age': 18, 'policy_state': 'IL', 'policy_csl': '100/300',
    'policy_deductable': 500, 'policy_annual_premium': 100.0, 'umbrella_limit': 0,
    'insured_sex': 'FEMALE', 'insured_education_level': 'Associate', 'insured_occupation': 'adm-clerical',
    'insured_hobbies': 'base-jumping', 'insured_relationship': 'husband', 'capital-gains': 0,
    'capital-loss': 0, 'policy_bind_date': datetime.date(2015, 1, 1),
    'incident_date': datetime.date(2015, 1, 1), 'incident_type': 'Multi-vehicle Collision',
    'collision_type': '?', 'incident_severity': 'Major Damage',
    'authorities_contacted': 'Ambulance', 'incident_state': 'NC', 'incident_city': 'Arlington',
    'incident_hour_of_the_day': 12, 'number_of_vehicles_involved': 1, 'property_damage': '?',
    'bodily_injuries': 0, 'witnesses': 0, 'police_report_available': '?',
    'injury_claim': 0, 'property_claim': 0, 'vehicle_claim': 0,
    'auto_make': 'Accura', 'auto_model': '3 Series', 'auto_year': 2010
}

LEGIT_SAMPLE = {
    'months_as_customer': 134, 'age': 29, 'policy_state': 'OH', 'policy_csl': '100/300', 
    'policy_deductable': 2000, 'policy_annual_premium': 1413.14, 'umbrella_limit': 5000000, 
    'insured_sex': 'FEMALE', 'insured_education_level': 'PhD', 'insured_occupation': 'sales', 
    'insured_hobbies': 'board-games', 'insured_relationship': 'own-child', 'capital-gains': 35100, 
    'capital-loss': 0, 'policy_bind_date': datetime.date(2000, 9, 6), 
    'incident_date': datetime.date(2015, 2, 22), 'incident_type': 'Multi-vehicle Collision', 
    'collision_type': 'Rear Collision', 'incident_severity': 'Minor Damage', 
    'authorities_contacted': 'Police', 'incident_state': 'NY', 'incident_city': 'Columbus', 
    'incident_hour_of_the_day': 7, 'number_of_vehicles_involved': 3, 'property_damage': 'NO', 
    'bodily_injuries': 2, 'witnesses': 3, 'police_report_available': 'NO', 
    'injury_claim': 7700, 'property_claim': 3850, 'vehicle_claim': 23100, 
    'auto_make': 'Dodge', 'auto_model': 'RAM', 'auto_year': 2007
}


# ==========================================
# UNDER-THE-HOOD FEATURE ENGINEERING
# ==========================================

def preprocess_and_engineer_features(raw_inputs):
    """Calculate extra inputs required by the model in the background."""
    data = dict(raw_inputs)
    
    # 1. policy_duration_day
    policy_bind = pd.to_datetime(data['policy_bind_date'])
    incident_d = pd.to_datetime(data['incident_date'])
    policy_duration_day = (incident_d - policy_bind).days
    if policy_duration_day < 0:
        policy_duration_day = 0
    data['policy_duration_day'] = policy_duration_day
    
    # 2. Vehicle_age
    current_year = datetime.datetime.now().year
    vehicle_age = current_year - data['auto_year']
    if vehicle_age < 0:
        vehicle_age = 0
    data['Vehicle_age'] = vehicle_age
    
    # Calculate total claim size
    injury = data['injury_claim']
    prop = data['property_claim']
    veh = data['vehicle_claim']
    total_claim = injury + prop + veh
    data['total_claim_amount'] = total_claim
    
    # 3. claim_per_vehicle_age
    if vehicle_age == 0:
        claim_per_vehicle_age = total_claim + 1
    else:
        claim_per_vehicle_age = (total_claim / vehicle_age) + 1
    data['claim_per_vehicle_age'] = claim_per_vehicle_age
    
    # 4. Ratios
    if total_claim == 0:
        data['property_claim_ratio'] = 0.0
        data['Vehicle_claim_ratio'] = 0.0
        data['injury_claim_ratio'] = 0.0
    else:
        data['property_claim_ratio'] = prop / total_claim
        data['Vehicle_claim_ratio'] = veh / total_claim
        data['injury_claim_ratio'] = injury / total_claim
        
    # 5. claim_per_vehicle
    num_vehicles = data['number_of_vehicles_involved']
    if num_vehicles == 0:
        data['claim_per_vehicle'] = total_claim
    else:
        data['claim_per_vehicle'] = total_claim / num_vehicles

    cols = [
        'months_as_customer', 'age', 'policy_state', 'policy_csl', 'policy_deductable',
        'policy_annual_premium', 'umbrella_limit', 'insured_sex', 'insured_education_level',
        'insured_occupation', 'insured_hobbies', 'insured_relationship', 'capital-gains',
        'capital-loss', 'incident_type', 'collision_type', 'incident_severity',
        'authorities_contacted', 'incident_state', 'incident_city', 'incident_hour_of_the_day',
        'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses',
        'police_report_available', 'total_claim_amount', 'injury_claim', 'property_claim',
        'vehicle_claim', 'auto_make', 'auto_model', 'policy_duration_day', 'Vehicle_age',
        'claim_per_vehicle_age', 'property_claim_ratio', 'Vehicle_claim_ratio',
        'claim_per_vehicle', 'injury_claim_ratio'
    ]
    
    df_pred = pd.DataFrame([data])[cols]
    return df_pred


# ==========================================
# SIDEBAR LOGO & NAVIGATION
# ==========================================

st.sidebar.markdown("""
<div style="padding: 10px 4px; margin-bottom: 8px;">
    <h2 style="color: #ffffff; font-size: 15px; font-weight: 700; margin: 0; letter-spacing: -0.3px; display: flex; align-items: center; gap: 8px;">
        🛡️ Car Insurance Fraud Detection
    </h2>
</div>
<hr style="border-color: rgba(255,255,255,0.05); margin-top: 0; margin-bottom: 20px;" />
""", unsafe_allow_html=True)

nav_options = [
    "🏠 Home", 
    "🔍 Predict Fraud", 
    "📊 Analytics", 
    "📈 Model Comparison", 
    "ℹ️ About"
]

if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

nav_index = nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0

selected_page = st.sidebar.radio(
    "Navigation Menu",
    nav_options,
    index=nav_index,
    key="nav_radio",
    label_visibility="collapsed"
)

# If radio button updates page, sync state
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()

st.sidebar.markdown("""
<div style="position: fixed; bottom: 20px; left: 20px; font-size: 11px; color: #64748b;">
    Model Type: Logistic Regression
</div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: HOME (PREMIUM SAAS LANDING)
# ==========================================

if selected_page == "🏠 Home":
    # Subtle Background Glow and Blur Circles
    st.markdown("""
    <div class="glow-bg"></div>
    <div class="blur-circle-1"></div>
    <div class="blur-circle-2"></div>
    """, unsafe_allow_html=True)
    
    # 1. Hero Section
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center; padding: 20px 0 0 0; width: 100%;">
        <div class="pill-badge">✨ AI Powered • Machine Learning</div>
        <h1 class="hero-title">
            Car Insurance <span class="fraud-gradient">Fraud</span> <br/>Prediction System
        </h1>
        <p class="hero-subtitle">
            Predict whether a car insurance claim is Fraudulent or Genuine using a trained Machine Learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Modern CTA Buttons (Centered Symmetrical columns with full container width)
    col_l, col_btn1, col_btn2, col_r = st.columns([2.2, 1.8, 1.8, 2.2])
    with col_btn1:
        if st.button("Start Prediction", type="primary", use_container_width=True):
            st.session_state.current_page = "🔍 Predict Fraud"
            st.rerun()
    with col_btn2:
        if st.button("Explore Analytics", type="secondary", use_container_width=True):
            st.session_state.current_page = "📊 Analytics"
            st.rerun()
            
    st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)
    
    # 3. Centered Illustration (Reduced by ~30% in width and centered)
    if os.path.exists("hero_illustration.png"):
        col_img1, col_img2, col_img3 = st.columns([4.0, 4.0, 4.0])
        with col_img2:
            st.image("hero_illustration.png", use_container_width=True)
            
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    
    # 4. Feature Section (Three Elegant Glassmorphism Cards)
    col_ft1, col_ft2, col_ft3 = st.columns(3)
    with col_ft1:
        st.markdown("""
        <div class="glass-card" style="min-height: 110px; padding: 18px 20px;">
            <h4 style="color: #ffffff; margin: 0 0 8px 0; font-size: 14.5px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                <span>🛡️</span> Fraud Detection
            </h4>
            <p style="color: #94a3b8; margin: 0; font-size: 12.5px; line-height: 1.45;">
                Detect fraudulent insurance claims using Machine Learning.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_ft2:
        st.markdown("""
        <div class="glass-card" style="min-height: 110px; padding: 18px 20px;">
            <h4 style="color: #ffffff; margin: 0 0 8px 0; font-size: 14.5px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                <span>⚡</span> Fast Prediction
            </h4>
            <p style="color: #94a3b8; margin: 0; font-size: 12.5px; line-height: 1.45;">
                Generate predictions instantly with a trained Logistic Regression pipeline.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_ft3:
        st.markdown("""
        <div class="glass-card" style="min-height: 110px; padding: 18px 20px;">
            <h4 style="color: #ffffff; margin: 0 0 8px 0; font-size: 14.5px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                <span>📊</span> Interactive Analytics
            </h4>
            <p style="color: #94a3b8; margin: 0; font-size: 12.5px; line-height: 1.45;">
                Visualize fraud trends using interactive Plotly charts.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: PREDICT FRAUD
# ==========================================

elif selected_page == "🔍 Predict Fraud":
    st.markdown('<div class="section-title">Insurance Claim Prediction</div>', unsafe_allow_html=True)
    
    if not isinstance(model, str):
        # Initialize form data in session state if not present
        if 'form_data' not in st.session_state:
            st.session_state.form_data = dict(BLANK_DEFAULTS)
            
        # Small button to load demo sample
        if st.button("Load Sample Data"):
            st.session_state.form_data = dict(LEGIT_SAMPLE)
            st.rerun()

        # Render Form
        with st.form("simple_prediction_form"):
            
            # Policy Section
            st.markdown('<div class="form-section-header"><span class="form-section-number">01</span> Policy Details</div>', unsafe_allow_html=True)
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                months_as_customer = st.number_input(
                    "Months as Customer", min_value=0, max_value=600, 
                    value=int(st.session_state.form_data.get('months_as_customer', 120))
                )
                policy_bind_date = st.date_input(
                    "Policy Bind Date", value=st.session_state.form_data.get('policy_bind_date', datetime.date(2010, 1, 1))
                )
            with p_col2:
                policy_state = st.selectbox(
                    "Policy State", options=CATEGORIES['policy_state'], 
                    index=CATEGORIES['policy_state'].index(st.session_state.form_data.get('policy_state', 'OH'))
                )
                policy_deductable = st.selectbox(
                    "Policy Deductible ($)", options=[500, 1000, 2000],
                    index=[500, 1000, 2000].index(st.session_state.form_data.get('policy_deductable', 1000))
                )
            with p_col3:
                policy_csl = st.selectbox(
                    "CSL Limit", options=CATEGORIES['policy_csl'], 
                    index=CATEGORIES['policy_csl'].index(st.session_state.form_data.get('policy_csl', '100/300'))
                )
                policy_annual_premium = st.number_input(
                    "Annual Premium ($)", min_value=100.0, max_value=4000.0, step=10.0,
                    value=float(st.session_state.form_data.get('policy_annual_premium', 1000.0))
                )
                umbrella_limit = st.number_input(
                    "Umbrella Limit ($)", min_value=0, max_value=20000000, step=100000,
                    value=int(st.session_state.form_data.get('umbrella_limit', 0))
                )
                
            # Customer Section
            st.markdown('<div class="form-section-header"><span class="form-section-number">02</span> Customer Details</div>', unsafe_allow_html=True)
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                age = st.number_input(
                    "Age", min_value=18, max_value=100, 
                    value=int(st.session_state.form_data.get('age', 30))
                )
                insured_sex = st.selectbox(
                    "Gender", options=CATEGORIES['insured_sex'], 
                    index=CATEGORIES['insured_sex'].index(st.session_state.form_data.get('insured_sex', 'MALE'))
                )
                insured_relationship = st.selectbox(
                    "Relationship", options=CATEGORIES['insured_relationship'], 
                    index=CATEGORIES['insured_relationship'].index(st.session_state.form_data.get('insured_relationship', 'own-child'))
                )
            with c_col2:
                insured_education_level = st.selectbox(
                    "Education Level", options=CATEGORIES['insured_education_level'], 
                    index=CATEGORIES['insured_education_level'].index(st.session_state.form_data.get('insured_education_level', 'College'))
                )
                insured_occupation = st.selectbox(
                    "Occupation", options=CATEGORIES['insured_occupation'], 
                    index=CATEGORIES['insured_occupation'].index(st.session_state.form_data.get('insured_occupation', 'adm-clerical'))
                )
                capital_gains = st.number_input(
                    "Capital Gains ($)", min_value=0, max_value=100000, step=1000,
                    value=int(st.session_state.form_data.get('capital-gains', 0))
                )
            with c_col3:
                insured_hobbies = st.selectbox(
                    "Hobbies", options=CATEGORIES['insured_hobbies'], 
                    index=CATEGORIES['insured_hobbies'].index(st.session_state.form_data.get('insured_hobbies', 'reading'))
                )
                capital_loss = st.number_input(
                    "Capital Loss ($)", min_value=0, max_value=100000, step=1000,
                    value=int(st.session_state.form_data.get('capital-loss', 0))
                )

            # Incident Section
            st.markdown('<div class="form-section-header"><span class="form-section-number">03</span> Incident Details</div>', unsafe_allow_html=True)
            i_col1, i_col2, i_col3 = st.columns(3)
            with i_col1:
                incident_date = st.date_input(
                    "Incident Date", value=st.session_state.form_data.get('incident_date', datetime.date(2015, 1, 1))
                )
                incident_type = st.selectbox(
                    "Incident Type", options=CATEGORIES['incident_type'], 
                    index=CATEGORIES['incident_type'].index(st.session_state.form_data.get('incident_type', 'Single Vehicle Collision'))
                )
                collision_type = st.selectbox(
                    "Collision Type", options=CATEGORIES['collision_type'], 
                    index=CATEGORIES['collision_type'].index(st.session_state.form_data.get('collision_type', 'Front Collision'))
                )
                incident_severity = st.selectbox(
                    "Incident Severity", options=CATEGORIES['incident_severity'], 
                    index=CATEGORIES['incident_severity'].index(st.session_state.form_data.get('incident_severity', 'Minor Damage'))
                )
            with i_col2:
                authorities_contacted = st.selectbox(
                    "Authorities Contacted", options=CATEGORIES['authorities_contacted'], 
                    index=CATEGORIES['authorities_contacted'].index(st.session_state.form_data.get('authorities_contacted', 'Police'))
                )
                incident_state = st.selectbox(
                    "Incident State", options=CATEGORIES['incident_state'], 
                    index=CATEGORIES['incident_state'].index(st.session_state.form_data.get('incident_state', 'OH'))
                )
                incident_city = st.selectbox(
                    "Incident City", options=CATEGORIES['incident_city'], 
                    index=CATEGORIES['incident_city'].index(st.session_state.form_data.get('incident_city', 'Columbus'))
                )
                incident_hour_of_the_day = st.slider(
                    "Incident Hour (0-23)", min_value=0, max_value=23, 
                    value=int(st.session_state.form_data.get('incident_hour_of_the_day', 12))
                )
            with i_col3:
                number_of_vehicles_involved = st.selectbox(
                    "Vehicles Involved", options=[1, 2, 3, 4], 
                    index=[1, 2, 3, 4].index(st.session_state.form_data.get('number_of_vehicles_involved', 1))
                )
                property_damage = st.selectbox(
                    "Property Damage?", options=CATEGORIES['property_damage'], 
                    index=CATEGORIES['property_damage'].index(st.session_state.form_data.get('property_damage', 'NO'))
                )
                police_report_available = st.selectbox(
                    "Police Report Available?", options=CATEGORIES['police_report_available'], 
                    index=CATEGORIES['police_report_available'].index(st.session_state.form_data.get('police_report_available', 'NO'))
                )
                bodily_injuries = st.selectbox(
                    "Bodily Injuries", options=[0, 1, 2], 
                    index=[0, 1, 2].index(st.session_state.form_data.get('bodily_injuries', 0))
                )
                witnesses = st.selectbox(
                    "Witnesses Count", options=[0, 1, 2, 3], 
                    index=[0, 1, 2, 3].index(st.session_state.form_data.get('witnesses', 0))
                )

            # Vehicle Section
            st.markdown('<div class="form-section-header"><span class="form-section-number">04</span> Vehicle Details</div>', unsafe_allow_html=True)
            v_col1, v_col2, v_col3 = st.columns(3)
            with v_col1:
                auto_make = st.selectbox(
                    "Vehicle Make", options=CATEGORIES['auto_make'], 
                    index=CATEGORIES['auto_make'].index(st.session_state.form_data.get('auto_make', 'Toyota'))
                )
            with v_col2:
                auto_model = st.selectbox(
                    "Vehicle Model", options=CATEGORIES['auto_model'], 
                    index=CATEGORIES['auto_model'].index(st.session_state.form_data.get('auto_model', 'Corolla'))
                )
            with v_col3:
                auto_year = st.number_input(
                    "Vehicle Year", min_value=1995, max_value=2026, 
                    value=int(st.session_state.form_data.get('auto_year', 2010))
                )

            # Claims Section
            st.markdown('<div class="form-section-header"><span class="form-section-number">05</span> Claims Breakdown</div>', unsafe_allow_html=True)
            cl_col1, cl_col2, cl_col3 = st.columns(3)
            with cl_col1:
                injury_claim = st.number_input(
                    "Injury Claim ($)", min_value=0, max_value=100000, step=100, 
                    value=int(st.session_state.form_data.get('injury_claim', 5000))
                )
            with cl_col2:
                property_claim = st.number_input(
                    "Property Claim ($)", min_value=0, max_value=100000, step=100, 
                    value=int(st.session_state.form_data.get('property_claim', 5000))
                )
            with cl_col3:
                vehicle_claim = st.number_input(
                    "Vehicle Claim ($)", min_value=0, max_value=150000, step=100, 
                    value=int(st.session_state.form_data.get('vehicle_claim', 20000))
                )
            
            calc_total = injury_claim + property_claim + vehicle_claim
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.015); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 8px; padding: 12px; margin-top: 8px;">
                <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight:600; letter-spacing:0.05em;">Total Calculated Claim</span><br/>
                <span style="color: #ffffff; font-size: 16px; font-weight: 700;">${calc_total:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Predict Fraud Likelihood")

        # Form Submit Predict
        if submit_btn:
            compiled_inputs = {
                'months_as_customer': months_as_customer, 'age': age, 'policy_state': policy_state,
                'policy_csl': policy_csl, 'policy_deductable': policy_deductable,
                'policy_annual_premium': policy_annual_premium, 'umbrella_limit': umbrella_limit,
                'insured_sex': insured_sex, 'insured_education_level': insured_education_level,
                'insured_occupation': insured_occupation, 'insured_hobbies': insured_hobbies,
                'insured_relationship': insured_relationship, 'capital-gains': capital_gains,
                'capital-loss': capital_loss, 'policy_bind_date': policy_bind_date,
                'incident_date': incident_date, 'incident_type': incident_type,
                'collision_type': collision_type, 'incident_severity': incident_severity,
                'authorities_contacted': authorities_contacted, 'incident_state': incident_state,
                'incident_city': incident_city, 'incident_hour_of_the_day': incident_hour_of_the_day,
                'number_of_vehicles_involved': number_of_vehicles_involved, 'property_damage': property_damage,
                'bodily_injuries': bodily_injuries, 'witnesses': witnesses, 'police_report_available': police_report_available,
                'injury_claim': injury_claim, 'property_claim': property_claim, 'vehicle_claim': vehicle_claim,
                'auto_make': auto_make, 'auto_model': auto_model, 'auto_year': auto_year
            }
            
            with st.spinner("Calculating variables and running prediction..."):
                try:
                    df_engineered = preprocess_and_engineer_features(compiled_inputs)
                    
                    pred_class = model.predict(df_engineered)[0]
                    prob_array = model.predict_proba(df_engineered)[0]
                    
                    fraud_prob = float(prob_array[1])
                    confidence = float(prob_array[pred_class])
                    
                    pred_label = "Fraud Suspicion Flagged" if pred_class == 1 else "Legitimate Claim Approved"
                    
                    if fraud_prob < 0.30:
                        risk_level = "Low Risk Profile"
                        risk_badge_class = "badge-low"
                        risk_color = "#10b981"
                        risk_desc = "This claim looks typical and is likely legitimate."
                    elif fraud_prob < 0.70:
                        risk_level = "Medium Suspicion"
                        risk_badge_class = "badge-medium"
                        risk_color = "#f59e0b"
                        risk_desc = "This claim has some unusual details. Manual review is suggested."
                    else:
                        risk_level = "High Fraud Risk"
                        risk_badge_class = "badge-high"
                        risk_color = "#ef4444"
                        risk_desc = "This claim is highly suspicious and should be investigated."

                    st.markdown("<br/><hr style='border-color: rgba(255,255,255,0.04);'/><br/>", unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        # Premium Result Card
                        st.markdown(f"""
                        <div class="result-card" style="border-top: 4px solid {risk_color};">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
                                <div>
                                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Prediction</div>
                                    <div style="font-size: 20px; font-weight: 700; color: #ffffff; margin-top: 4px;">{pred_label}</div>
                                </div>
                                <span class="badge {risk_badge_class}">{risk_level}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; padding: 16px 0; border-top: 1px solid rgba(255,255,255,0.04); border-bottom: 1px solid rgba(255,255,255,0.04);">
                                <div>
                                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Fraud Probability</div>
                                    <div style="font-size: 24px; font-weight: 700; color: {risk_color}; margin-top: 4px;">{(fraud_prob*100):.1f}%</div>
                                </div>
                                <div>
                                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Confidence Score</div>
                                    <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-top: 4px;">{(confidence*100):.1f}%</div>
                                </div>
                            </div>
                            <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6; margin: 0;">
                                <strong>Recommended Action:</strong> {risk_desc}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col2:
                        # Clean Gauge Chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = float(fraud_prob * 100),
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk Score (%)", 'font': {'size': 13, 'color': '#ffffff', 'family': 'Plus Jakarta Sans'}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#64748b', 'size': 10}},
                                'bar': {'color': "#6366f1"},
                                'bgcolor': "rgba(255, 255, 255, 0.015)",
                                'borderwidth': 1,
                                'bordercolor': "rgba(255, 255, 255, 0.05)",
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.08)'},
                                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.08)'},
                                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.08)'}
                                ],
                                'threshold': {
                                    'line': {'color': "#6366f1", 'width': 2},
                                    'thickness': 0.75,
                                    'value': float(fraud_prob * 100)
                                }
                            }
                        ))
                        fig_gauge.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
                            height=190,
                            margin=dict(l=30, r=30, t=35, b=10)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                except Exception as ex:
                    st.error(f"Prediction Error: {str(ex)}")
    else:
        st.error(model)


# ==========================================
# PAGE 3: HISTORICAL CLAIMS ANALYTICS
# ==========================================

elif selected_page == "📊 Analytics":
    st.markdown('<div class="section-title">Analytics (1,000 Historical Claims)</div>', unsafe_allow_html=True)
    
    if df_claims is not None:
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Claims</div>
                <div class="kpi-value">1,000</div>
            </div>
            """, unsafe_allow_html=True)
        with a_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Fraud Rate</div>
                <div class="kpi-value">24.7%</div>
            </div>
            """, unsafe_allow_html=True)
        with a_col3:
            avg_claim = df_claims['total_claim_amount'].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Claim Amount</div>
                <div class="kpi-value">${avg_claim:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Grid of Charts - EXACTLY 5 Charts
        row1_col1, row1_col2 = st.columns([5, 7])
        
        # 1. Fraud Distribution Donut
        with row1_col1:
            fraud_counts = df_claims['fraud_reported'].value_counts().reset_index()
            fraud_counts.columns = ['Fraud Status', 'Count']
            fraud_counts['Fraud Status'] = fraud_counts['Fraud Status'].map({'Y': 'Fraudulent', 'N': 'Legitimate'})
            fig1 = px.pie(
                fraud_counts, values='Count', names='Fraud Status', hole=0.55,
                color='Fraud Status', color_discrete_map={'Fraudulent': '#ef4444', 'Legitimate': '#10b981'},
                title="Fraud Distribution"
            )
            customize_plotly_layout(fig1)
            st.plotly_chart(fig1, use_container_width=True)
            
        # 2. Claim Amount Distribution Histogram
        with row1_col2:
            fig2 = px.histogram(
                df_claims, x='total_claim_amount', color='fraud_reported',
                color_discrete_map={'Y': '#ef4444', 'N': '#10b981'}, barmode='overlay',
                labels={'total_claim_amount': 'Total Claim Amount ($)', 'count': 'Number of Claims'},
                title="Claim Amount Distribution"
            )
            customize_plotly_layout(fig2)
            st.plotly_chart(fig2, use_container_width=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        row2_col1, row2_col2 = st.columns(2)
        
        # 3. Fraud by Incident Severity
        with row2_col1:
            severity_counts = df_claims.groupby(['incident_severity', 'fraud_reported']).size().reset_index(name='Count')
            severity_counts['fraud_reported'] = severity_counts['fraud_reported'].map({'Y': 'Fraudulent', 'N': 'Legitimate'})
            fig3 = px.bar(
                severity_counts, x='incident_severity', y='Count', color='fraud_reported',
                color_discrete_map={'Fraudulent': '#ef4444', 'Legitimate': '#10b981'}, barmode='group',
                category_orders={'incident_severity': ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']},
                labels={'incident_severity': 'Severity', 'Count': 'Claims Count'},
                title="Fraud by Incident Severity"
            )
            customize_plotly_layout(fig3)
            st.plotly_chart(fig3, use_container_width=True)
            
        # 4. Fraud by Incident Type
        with row2_col2:
            incident_type_counts = df_claims.groupby(['incident_type', 'fraud_reported']).size().reset_index(name='Count')
            incident_type_counts['fraud_reported'] = incident_type_counts['fraud_reported'].map({'Y': 'Fraudulent', 'N': 'Legitimate'})
            fig4 = px.bar(
                incident_type_counts, x='incident_type', y='Count', color='fraud_reported',
                color_discrete_map={'Fraudulent': '#ef4444', 'Legitimate': '#10b981'}, barmode='group',
                labels={'incident_type': 'Incident Type', 'Count': 'Claims Count'},
                title="Fraud by Incident Type"
            )
            customize_plotly_layout(fig4)
            st.plotly_chart(fig4, use_container_width=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # 5. Fraud by Property Damage (Full Width)
        prop_counts = df_claims.groupby(['property_damage', 'fraud_reported']).size().reset_index(name='Count')
        prop_counts['fraud_reported'] = prop_counts['fraud_reported'].map({'Y': 'Fraudulent', 'N': 'Legitimate'})
        fig5 = px.bar(
            prop_counts, x='property_damage', y='Count', color='fraud_reported',
            color_discrete_map={'Fraudulent': '#ef4444', 'Legitimate': '#10b981'}, barmode='stack',
            labels={'property_damage': 'Property Damage Flag', 'Count': 'Claims Count'},
            title="Fraud by Property Damage"
        )
        customize_plotly_layout(fig5)
        st.plotly_chart(fig5, use_container_width=True)
        
    else:
        st.error("Historical dataset `insurance_claims.csv` not found in current directory.")


# ==========================================
# PAGE 4: MODEL COMPARISON
# ==========================================

elif selected_page == "📈 Model Comparison":
    st.markdown('<div class="section-title">Model Evaluation Metrics (Logistic Regression)</div>', unsafe_allow_html=True)
    
    # Model evaluation metrics
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value">84%</div>
            <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Correct predictions</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision</div>
            <div class="kpi-value">77%</div>
            <div style="color: #64748b; font-size: 11px; margin-top: 4px;">True positive rate</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Recall</div>
            <div class="kpi-value">49%</div>
            <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Fraud detection rate</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">F1 Score</div>
            <div class="kpi-value">60%</div>
            <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Balanced metric</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ROC AUC</div>
            <div class="kpi-value">84%</div>
            <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Separation ability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Selection Banner
    st.markdown("""
    <div class="glass-card" style="border-left: 3px solid #6366f1; background: rgba(99,102,241,0.015);">
        <h3 style="color:#ffffff; margin-top:0; font-size:15px; font-weight: 600;">🏆 Selected Model: Logistic Regression</h3>
        <p style="color:#94a3b8; font-size:13px; line-height:1.5; margin:0;">
            We selected <strong>Logistic Regression</strong> for this project. It is simple, fast, and achieved a solid accuracy of <strong>84.5%</strong>. It also allows us to understand which factors contribute most to a fraud prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Benchmarks DataFrame
    comparison_data = {
        'Model Pipeline': ['Logistic Regression (Selected)', 'Linear SVM', 'PCA + SVM', 'RBF SVM', 'KNN', 'Naive Bayes'],
        'Accuracy (%)': [84.5, 83.0, 77.5, 76.0, 74.5, 62.0],
        'Precision (%)': [78.1, 71.4, 83.3, 60.0, 45.0, 33.3],
        'Recall (%)': [51.0, 51.0, 10.2, 6.1, 18.4, 55.1],
        'F1 Score (%)': [61.7, 59.5, 18.2, 11.1, 26.1, 41.5],
        'ROC AUC (%)': [83.6, 79.9, 82.1, 85.0, 69.1, 64.4]
    }
    df_compare = pd.DataFrame(comparison_data)
    
    st.subheader("Model Performance Comparison")
    st.table(df_compare)
    
    # Benchmarking Chart
    fig_compare = px.bar(
        df_compare, x='Model Pipeline', y=['Accuracy (%)', 'F1 Score (%)', 'ROC AUC (%)'],
        barmode='group', color_discrete_sequence=['#3b82f6', '#818cf8', '#10b981'],
        title="Metrics Comparison"
    )
    customize_plotly_layout(fig_compare)
    st.plotly_chart(fig_compare, use_container_width=True)


# ==========================================
# PAGE 5: ABOUT (CONSOLIDATED WORKFLOW & STATS)
# ==========================================

elif selected_page == "ℹ️ About":
    st.markdown('<div class="section-title">About the Project</div>', unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#818cf8; margin-top:0; font-size:15px; font-weight:600;">📋 Project Details</h3>
        <p style="color:#cbd5e1; font-size:13.5px; line-height:1.6; margin:0 0 12px 0;">
            This system helps identify suspicious insurance claims using a trained machine learning model.
        </p>
        <ul style="color:#cbd5e1; font-size:13px; line-height:1.8; margin:0; padding-left:18px;">
            <li><strong>Easy to Use</strong>: Enter standard claim details and get instant predictions.</li>
            <li><strong>Hidden Preprocessing</strong>: Calculates advanced variables under the hood automatically.</li>
            <li><strong>Quick Templates</strong>: Test the tool instantly using pre-filled templates.</li>
            <li><strong>Visual Insights</strong>: View simple graphs explaining the claim patterns.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Project Workflow
    st.markdown('<div class="section-title" style="font-size:14px; margin-top:10px;">Project Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.04); padding: 20px; border-radius: 12px; margin-bottom: 24px; text-align: center;">
        <div style="flex: 1; min-width: 80px; padding: 5px;">
            <div style="font-size: 18px; margin-bottom: 4px;">📝</div>
            <div style="font-size: 11px; font-weight: 600; color: #ffffff;">Insurance Claim</div>
        </div>
        <div style="color: #64748b; font-size: 14px; padding: 0 5px; font-weight: bold;">↓</div>
        <div style="flex: 1; min-width: 80px; padding: 5px;">
            <div style="font-size: 18px; margin-bottom: 4px;">⚙️</div>
            <div style="font-size: 11px; font-weight: 600; color: #ffffff;">Data Preprocessing</div>
        </div>
        <div style="color: #64748b; font-size: 14px; padding: 0 5px; font-weight: bold;">↓</div>
        <div style="flex: 1; min-width: 80px; padding: 5px;">
            <div style="font-size: 18px; margin-bottom: 4px;">🛠️</div>
            <div style="font-size: 11px; font-weight: 600; color: #ffffff;">Feature Engineering</div>
        </div>
        <div style="color: #64748b; font-size: 14px; padding: 0 5px; font-weight: bold;">↓</div>
        <div style="flex: 1; min-width: 80px; padding: 5px;">
            <div style="font-size: 18px; margin-bottom: 4px;">🧠</div>
            <div style="font-size: 11px; font-weight: 600; color: #ffffff;">ML Model</div>
        </div>
        <div style="color: #64748b; font-size: 14px; padding: 0 5px; font-weight: bold;">↓</div>
        <div style="flex: 1; min-width: 80px; padding: 5px;">
            <div style="font-size: 18px; margin-bottom: 4px;">🎯</div>
            <div style="font-size: 11px; font-weight: 600; color: #ffffff;">Fraud Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Summary
    st.markdown('<div class="section-title" style="font-size:14px; margin-top:10px;">Dataset Summary</div>', unsafe_allow_html=True)
    d_col1, d_col2, d_col3, d_col4, d_col5, d_col6 = st.columns(6)
    with d_col1:
        st.markdown("""
        <div class="kpi-card" style="height: 90px; padding: 12px; border-color: rgba(255,255,255,0.03);">
            <div class="kpi-label" style="font-size: 9px;">Dataset</div>
            <div style="font-size: 12px; font-weight: 700; color: #ffffff; margin-top: 4px; line-height: 1.2;">Auto Insurance</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col2:
        st.markdown("""
        <div class="kpi-card" style="height: 90px; padding: 12px; border-color: rgba(255,255,255,0.03);">
            <div class="kpi-label" style="font-size: 9px;">Records</div>
            <div class="kpi-value" style="font-size: 18px; margin-top: 2px;">1,000</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col3:
        st.markdown("""
        <div class="kpi-card" style="height: 90px; padding: 12px; border-color: rgba(255,255,255,0.03);">
            <div class="kpi-label" style="font-size: 9px;">Features</div>
            <div class="kpi-value" style="font-size: 18px; margin-top: 2px;">40</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col4:
        st.markdown("""
        <div class="kpi-card" style="height: 90px; padding: 12px; border-color: rgba(255,255,255,0.03);">
            <div class="kpi-label" style="font-size: 9px;">Target</div>
            <div style="font-size: 12px; font-weight: 700; color: #ffffff; margin-top: 4px; line-height: 1.2;">Fraud Reported</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col5:
        st.markdown("""
        <div class="kpi-card" style="height: 90px; padding: 12px; border-color: rgba(255,255,255,0.03);">
            <div class="kpi-label" style="font-size: 9px;">Fraud Cases</div>
            <div class="kpi-value" style="font-size: 18px; color: #ef4444; margin-top: 2px;">247</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col6:
        st.markdown("""
        <div class="kpi-card" style="height: 90px; padding: 12px; border-color: rgba(255,255,255,0.03);">
            <div class="kpi-label" style="font-size: 9px;">Genuine Cases</div>
            <div class="kpi-value" style="font-size: 18px; color: #10b981; margin-top: 2px;">753</div>
        </div>
        """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown('<div class="section-title" style="font-size:14px; margin-top:10px;">Technology Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px;">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">Joblib</span>
        <span class="tech-badge">GitHub</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Why This Project
    st.markdown('<div class="section-title" style="font-size:14px; margin-top:10px;">Why This Project?</div>', unsafe_allow_html=True)
    wp1, wp2, wp3, wp4 = st.columns(4)
    with wp1:
        st.markdown("""
        <div class="kpi-card" style="text-align: center; padding: 16px; border-color: rgba(255,255,255,0.03);">
            <div style="font-size: 13px; font-weight: 600; color: #ffffff;">Business Problem</div>
        </div>
        """, unsafe_allow_html=True)
    with wp2:
        st.markdown("""
        <div class="kpi-card" style="text-align: center; padding: 16px; border-color: rgba(255,255,255,0.03);">
            <div style="font-size: 13px; font-weight: 600; color: #ffffff;">Model Development</div>
        </div>
        """, unsafe_allow_html=True)
    with wp3:
        st.markdown("""
        <div class="kpi-card" style="text-align: center; padding: 16px; border-color: rgba(255,255,255,0.03);">
            <div style="font-size: 13px; font-weight: 600; color: #ffffff;">Interactive Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    with wp4:
        st.markdown("""
        <div class="kpi-card" style="text-align: center; padding: 16px; border-color: rgba(255,255,255,0.03);">
            <div style="font-size: 13px; font-weight: 600; color: #ffffff;">Real-world Portfolio Project</div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Background Challenge
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#818cf8; margin-top:0; font-size:15px; font-weight:600;">💼 1. The Challenge</h3>
        <ul style="color:#cbd5e1; font-size:13.5px; line-height:1.7; margin:0; padding-left:18px;">
            <li>Insurance fraud costs companies billions of dollars every year.</li>
            <li>Manually checking every claim is slow, tedious, and expensive.</li>
            <li>Automating initial checks flags suspicious claims early to speed up investigations.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset specs
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#818cf8; margin-top:0; font-size:15px; font-weight:600;">📋 2. The Dataset Description</h3>
        <p style="color:#cbd5e1; font-size:13.5px; line-height:1.6; margin:0 0 10px 0;">
            The model was trained on 1,000 historical auto insurance claims containing variables such as:
        </p>
        <ul style="color:#cbd5e1; font-size:13px; line-height:1.8; margin:0; padding-left:18px;">
            <li><strong>Policy details</strong>: Start date, deductible choices, and annual premiums.</li>
            <li><strong>Customer profiles</strong>: Age, education, occupation, and hobbies.</li>
            <li><strong>Incident variables</strong>: Severity, collision type, location, weather details, and police reports.</li>
            <li><strong>Claim values</strong>: Breakdown of property, vehicle, and injury claims.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Engineering
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#818cf8; margin-top:0; font-size:15px; font-weight:600;">🛠️ 3. Feature Calculations</h3>
        <p style="color:#cbd5e1; font-size:13.5px; line-height:1.6; margin-bottom:12px;">
            The app automatically calculates 7 custom variables from the raw inputs before making predictions:
        </p>
        <table style="width:100%; border-collapse:collapse; color:#cbd5e1; font-size:12.5px;">
            <tr style="border-bottom:1px solid rgba(255,255,255,0.05); text-align:left;">
                <th style="padding:8px 0; color:#ffffff;">Calculated Column</th>
                <th style="padding:8px 0; color:#ffffff;">Description</th>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:8px 0; font-weight:600; color:#818cf8;">policy_duration_day</td>
                <td style="padding:8px 0;">Days between Policy Bind Date and Incident Date.</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:8px 0; font-weight:600; color:#818cf8;">Vehicle_age</td>
                <td style="padding:8px 0;">Age of the vehicle since its manufacture year.</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:8px 0; font-weight:600; color:#818cf8;">claim_per_vehicle_age</td>
                <td style="padding:8px 0;">Total claim amount relative to the vehicle's age.</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:8px 0; font-weight:600; color:#818cf8;">property_claim_ratio</td>
                <td style="padding:8px 0;">Property claim portion relative to the total claim amount.</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:8px 0; font-weight:600; color:#818cf8;">Vehicle_claim_ratio</td>
                <td style="padding:8px 0;">Vehicle claim portion relative to the total claim amount.</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:8px 0; font-weight:600; color:#818cf8;">claim_per_vehicle</td>
                <td style="padding:8px 0;">Total claim amount divided by vehicles involved.</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                <td style="padding:8px 0; font-weight:600; color:#818cf8;">injury_claim_ratio</td>
                <td style="padding:8px 0;">Injury claim portion relative to the total claim amount.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Pipeline specs
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#818cf8; margin-top:0; font-size:15px; font-weight:600;">🧠 4. Machine Learning Model Pipeline</h3>
        <p style="color:#cbd5e1; font-size:13.5px; line-height:1.6; margin:0 0 10px 0;">
            The model pipeline is composed of the following steps:
        </p>
        <ul style="color:#cbd5e1; font-size:13px; line-height:1.8; margin:0; padding-left:18px;">
            <li><strong>Data Routing</strong>: Splits categorical and numerical values.</li>
            <li><strong>Encoding</strong>: Converts text variables into numerical inputs.</li>
            <li><strong>Scaling</strong>: Normalizes numeric ranges to improve predictive accuracy.</li>
            <li><strong>Inference</strong>: Predicts the final probability of fraud using Logistic Regression.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
