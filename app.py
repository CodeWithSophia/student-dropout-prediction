import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('dropout_model.pkl')
scaler = joblib.load('dropout_scaler.pkl')

# Page config
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #f0f4f8;
    }
    
    .main-container {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #0f3460;
    }
    
    .section-header {
        color: #1a1a2e;
        font-size: 1.3rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0f3460;
        margin-bottom: 1.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        color: white;
        border: none;
        padding: 0.75rem 3rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 52, 96, 0.4);
    }

    .stSelectbox label, .stNumberInput label {
        color: #2d3748;
        font-weight: 500;
    }

    .result-high {
        background: #fff5f5;
        border: 2px solid #fc8181;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .result-low {
        background: #f0fff4;
        border: 2px solid #68d391;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>🎓 Student Dropout Risk Predictor</h1>
        <p>Early identification of at-risk students using machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Model stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color:#0f3460; margin:0">92%</h3>
            <p style="color:#718096; margin:0">Model Recall</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color:#0f3460; margin:0">0.972</h3>
            <p style="color:#718096; margin:0">AUC Score</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color:#0f3460; margin:0">3,630</h3>
            <p style="color:#718096; margin:0">Students Trained On</p>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
        <div class="metric-card">
            <h3 style="color:#0f3460; margin:0">34</h3>
            <p style="color:#718096; margin:0">Features Used</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input section
st.markdown('<p class="section-header">📋 Student Information</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📚 Academic Details**")
    cu1_enrolled = st.number_input("1st Sem Units Enrolled", min_value=0, max_value=20, value=6,
        help="Number of subjects the student registered for in semester 1.")
    cu1_approved = st.number_input("1st Sem Units Approved", min_value=0, max_value=20, value=6,
        help="Number of subjects passed in semester 1. If much lower than enrolled, high dropout risk.")
    cu1_grade = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=12.0,
        help="Average grade in semester 1 on a scale of 0 to 20.")
    cu2_enrolled = st.number_input("2nd Sem Units Enrolled", min_value=0, max_value=20, value=6,
        help="Number of subjects the student registered for in semester 2.")
    cu2_approved = st.number_input("2nd Sem Units Approved", min_value=0, max_value=20, value=6,
        help="Number of subjects passed in semester 2. Key dropout predictor.")
    cu2_grade = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=12.0,
        help="Average grade in semester 2 on a scale of 0 to 20.")

with col2:
    st.markdown("**💰 Financial Details**")
    tuition = st.selectbox("Tuition Fees Up to Date", options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Is the student current with tuition payments? Students not up to date have a 94% dropout rate.")
    scholarship = st.selectbox("Scholarship Holder", options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Scholarship holders are 3.5x less likely to drop out.")
    debtor = st.selectbox("Debtor", options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Does the student owe money to the institution?")
    unemployment = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=25.0, value=10.0,
        help="National unemployment rate at time of enrollment. Higher rates increase dropout risk.")

with col3:
    st.markdown("**👤 Personal Details**")
    age = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20,
        help="Age of the student when they first enrolled.")
    gender = st.selectbox("Gender", options=[1, 0],
        format_func=lambda x: "Male" if x == 1 else "Female",
        help="Student's gender.")
    international = st.selectbox("International Student", options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="International students may face additional pressures including visa issues and homesickness.")
    displaced = st.selectbox("Displaced", options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether the student relocated from their home region to study.")

st.markdown("<br>", unsafe_allow_html=True)

# Build input array
input_data = np.array([[
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    displaced, 0, debtor, tuition, gender, scholarship, age, international,
    cu1_enrolled, cu1_enrolled, cu1_enrolled, cu1_approved, cu1_grade, 0,
    0, cu2_enrolled, cu2_enrolled, cu2_approved, cu2_grade, 0,
    unemployment, 1.0, 1.0
]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict Dropout Risk")

if predict_btn:
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Prediction Result</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if prediction == 1:
            st.markdown(f"""
                <div class="result-high">
                    <h2 style="color:#e53e3e">⚠️ High Dropout Risk</h2>
                    <h3 style="color:#e53e3e">Probability: {probability:.1%}</h3>
                    <p style="color:#718096">This student may need academic or financial intervention.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-low">
                    <h2 style="color:#38a169">✅ Low Dropout Risk</h2>
                    <h3 style="color:#38a169">Probability: {probability:.1%}</h3>
                    <p style="color:#718096">This student is on track to graduate.</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⚠️ This tool is for educational purposes only and should be used alongside professional academic advising.")

# Footer
st.markdown("---")
st.markdown("""
    <p style="text-align:center; color:#718096">
        Built by <strong>Sophia</strong> | Healthcare & Education ML Specialist | 
        <a href="https://github.com/CodeWithSophia" style="color:#0f3460">GitHub</a>
    </p>
""", unsafe_allow_html=True)