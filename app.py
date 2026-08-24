# app.py - FINAL VERSION: Loan Prediction + Risk Band + DDL + Gamification + Community Insights + Multilingual + ITR + Mock KYC + SHAP Suggestions + Mandatory Fields

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import numpy as np
import os

# --- Import Refactored Modules ---
from translations import TRANSLATIONS
from auth import login_page, logout
from utils import extract_bill_amount, extract_itr_info
from risk_analysis import (
    load_artifacts,
    preprocess_input,
    compute_composite_score,
    risk_band_classification,
    sanctionable_flag
)
from visualization import visualize_credit_data, community_insights

# --- Mock Database for Prototype Verification ---
MOCK_PAN_DB = {
    "ABCDE1234F": {"name": "John Doe", "status": "Valid"},
    "FGHIJ5678K": {"name": "Jane Smith", "status": "Valid"},
}

MOCK_AADHAAR_DB = {
    "123456789012": {"status": "Valid"},
    "987654321098": {"status": "Valid"},
}

# --- Session State Defaults ---
if 'language' not in st.session_state:
    st.session_state['language'] = "English"
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'chat_support_loaded' not in st.session_state:
    st.session_state['chat_support_loaded'] = False
if 'pan_status' not in st.session_state:
    st.session_state['pan_status'] = None
if 'aadhaar_status' not in st.session_state:
    st.session_state['aadhaar_status'] = None

# --- Sidebar Language Selector ---
st.sidebar.selectbox(
    "🌐 Select Language",
    list(TRANSLATIONS.keys()),
    key="language"
)
t = TRANSLATIONS[st.session_state['language']]

# --- Load Model & Features ---
MODEL, MODEL_FEATURES = load_artifacts(t)

# --- KYC Verification Logic ---
def verify_pan_mock():
    pan_number_input = st.session_state.pan_input.upper()
    if pan_number_input in MOCK_PAN_DB:
        applicant_info = MOCK_PAN_DB[pan_number_input]
        st.session_state['pan_status'] = f"✅ PAN is valid! Name on record: {applicant_info['name']}"
    else:
        st.session_state['pan_status'] = "❌ PAN not found. Prototype only verifies the predefined list."

def verify_aadhaar_mock():
    aadhaar_number_input = st.session_state.aadhaar_input
    if aadhaar_number_input in MOCK_AADHAAR_DB:
        st.session_state['aadhaar_status'] = "✅ Aadhaar is valid!"
    else:
        st.session_state['aadhaar_status'] = "❌ Aadhaar not registered. Prototype only verifies the predefined list."

# --- Main App ---
def main_app():
    t = TRANSLATIONS[st.session_state['language']]
    st.set_page_config(page_title=t["loan_prediction"], layout="wide")
    st.title(t["loan_prediction"])

    # --- Sidebar ---
    with st.sidebar:
        st.title(t["menu_title"])
        app_option = st.radio(t.get("select_page","Select Page:"), [t["loan_prediction"], t["data_viz"], t["community"]])
        st.button(t["logout"], on_click=logout, type="secondary")
        st.markdown("---")
        st.title(t.get("chat_support","💬 Chat Support"))

        st.markdown(
            """
            <style>
                .chat-container {
                    height: 60vh;
                    max-height: 600px;
                    overflow-y: auto;
                    padding: 10px;
                }
                .chat-container iframe {
                    width: 100%;
                    height: 100%;
                    border: none;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if not st.session_state['chat_support_loaded']:
            st.session_state['chat_support_loaded'] = True

        components.html(
            """
            <script>
                window.chtlConfig = { chatbotId: "6317658313" }
            </script>
            <script async data-id="6317658313" id="chtl-script" type="text/javascript"
                src="https://chatling.ai/js/embed.js"></script>
            """,
            height=600,
            scrolling=False
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Loan Prediction Page ---
    if app_option == t["loan_prediction"]:
        with st.form("applicant_form"):
            st.header(t.get("applicant_data","1. Applicant Data Input"))
            col1, col2 = st.columns(2)

            with col1:
                full_name = st.text_input(t.get("full_name","Applicant's Full Name"))
                city = st.text_input(t.get("city_residence","City of Residence"))

                # --- Current Bill (Mandatory) ---
                current_bill_file = st.file_uploader(t.get("upload_current_bill","Upload Monthly Current Bill (PDF) *"), type=["csv", "xlsx"])
                monthly_current_bill = 0
                if current_bill_file:
                    monthly_current_bill = extract_bill_amount(current_bill_file, t)
                    st.success(f"{t.get('bill_extracted','Monthly Current Bill extracted')}: {monthly_current_bill}")
                else:
                    monthly_current_bill = st.number_input(t.get("current_bill","Monthly Current Bill (INR) *"), 0, 30_000, 0, step=10)

                previous_loan_paid = st.selectbox(t.get("previous_loan","Previous Loan Paid?"), ["Yes", "No"])

            with col2:
                # --- Family Recharge Bill (Mandatory) ---
                recharge_bill_file = st.file_uploader(t.get("upload_recharge_bill","Upload Family Recharge Bill (PDF) *"), type=["csv", "xlsx"])
                family_recharge_bill = 0
                if recharge_bill_file:
                    family_recharge_bill = extract_bill_amount(recharge_bill_file, t)
                    st.success(f"{t.get('bill_extracted','Monthly Current Bill extracted')}: {family_recharge_bill}")
                else:
                    family_recharge_bill = st.number_input(t.get("recharge_bill","Family Recharge Bill (INR) *"), 0, 30_000, 0, step=10)

                # --- Annual Income (Mandatory) ---
                income = st.number_input(t.get("annual_income","Annual Income (USD) *"), 0, 1_000_0000, 0, step=1_000)
                credit_score = st.number_input(t.get("credit_score","Credit Score"), 300, 850, 650, step=1)
                loan_amount = st.number_input(t.get("loan_amount","Loan Amount Requested (INR)"), 1_000, 500_000, 50_000, step=1_000)
                years_employed = st.number_input(t.get("years_employed","Number of Years Employed"), 0, 50, 5, step=1)

                # --- ITR Upload ---
                itr_file = st.file_uploader(t.get("upload_itr","Upload Income Tax Return (PDF)"), type=["csv","xlsx"])
                itr_income, itr_tax, itr_years = 0, 0, []
                if itr_file:
                    itr_income, itr_tax, itr_years = extract_itr_info(itr_file, t)
                    st.success(f"{t.get('itr_info','ITR extracted')}: Income={itr_income}, Tax Paid={itr_tax}, Years Filed={len(itr_years)}")
                else:
                    st.info(t.get('itr_missing','No ITR uploaded'))

            st.markdown("---")
            predict_button = st.form_submit_button(t.get("predict","Predict Loan Approval"))

        # --- Check mandatory fields before prediction ---
        if predict_button:
            if monthly_current_bill <= 0 or family_recharge_bill <= 0 or income <= 0:
                st.error("⚠ Please fill all mandatory fields marked with * before prediction (Monthly Current Bill, Family Recharge Bill, Annual Income).")
            else:
                # --- MOCK KYC Verification ---
                st.markdown("---")
                st.header(t.get("document_verification", "2. Document Verification (Prototype)"))
                kyc_col1, kyc_col2 = st.columns(2)

                with kyc_col1:
                    with st.form("pan_verification_form"):
                        st.subheader("Verify PAN Card")
                        st.info(f"💡 For this demo, use: {', '.join(MOCK_PAN_DB.keys())}")
                        pan_number_input = st.text_input("Enter PAN Number", key="pan_input", placeholder="e.g., ABCDE1234F")
                        st.form_submit_button("Verify PAN (Mock)", on_click=verify_pan_mock)

                if st.session_state['pan_status']:
                    st.success(st.session_state['pan_status']) if st.session_state['pan_status'].startswith("✅") else st.error(st.session_state['pan_status'])

                with kyc_col2:
                    with st.form("aadhaar_verification_form"):
                        st.subheader("Verify Aadhaar Card")
                        st.info(f"💡 For this demo, use: {', '.join(MOCK_AADHAAR_DB.keys())}")
                        aadhaar_number_input = st.text_input("Enter Aadhaar Number", key="aadhaar_input", placeholder="e.g., 123456789012")
                        st.form_submit_button("Verify Aadhaar (Mock)", on_click=verify_aadhaar_mock)

                if st.session_state['aadhaar_status']:
                    st.success(st.session_state['aadhaar_status']) if st.session_state['aadhaar_status'].startswith("✅") else st.error(st.session_state['aadhaar_status'])

                st.markdown("---")

                # --- Prediction Logic ---
                user_input_dict = {
                    'monthly_current_bill': monthly_current_bill,
                    'family_recharge_bill': family_recharge_bill,
                    'previous_loan_paid': previous_loan_paid,
                    'income': income,
                    'credit_score': credit_score,
                    'loan_amount': loan_amount,
                    'years_employed': years_employed,
                    'user_itr_income': itr_income,
                    'user_itr_tax': itr_tax,
                    'user_itr_years': len(itr_years)
                }

                X_processed = preprocess_input(user_input_dict, MODEL_FEATURES)
                prob_approval = MODEL.predict_proba(X_processed)[:, 1][0]
                approval_score = int(prob_approval * 100)
                status = t["approved"] if approval_score >= 50 else t["declined"]
                points = int((1 - prob_approval) * 100)

                completeness = 0.9
                income_score = income / 1_000_000
                composite_score = compute_composite_score(prob_approval, income_score, completeness)
                risk_band = risk_band_classification(composite_score, income_score, t)
                auto_sanction = sanctionable_flag(composite_score, completeness)

                st.markdown(f"## {t['result_title']} {full_name}")
                st.metric(t.get("approval_prob","Loan Approval Probability (%)"), approval_score)
                st.success(f"{t.get('loan_status','Loan Status')}: {status}")
                st.info(f"{t.get('risk_points','Risk Points')}: {points}")
                st.markdown(f"### {t.get('risk_band','🧮 Risk Band')}: {risk_band}")
                st.markdown(f"### {t.get('auto_sanction','💵 Automatic Loan Sanction')}: {'✅ '+t.get('approved','Approved') if auto_sanction else '⚠ '+t.get('manual_review','Requires Manual Review')}")

                # --- Gauge Chart ---
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=approval_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': t.get("approval_prob","Approval Probability"), 'font': {'size': 20}},
                    delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if approval_score>=80 else "gold" if approval_score>=50 else "red"},
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(255,0,0,0.3)'},
                            {'range': [50, 80], 'color': 'rgba(255,215,0,0.3)'},
                            {'range': [80, 100], 'color': 'rgba(0,255,0,0.3)'}
                        ],
                        'threshold': {'line': {'color': "green", 'width': 6}, 'thickness': 0.8, 'value': approval_score}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                # --- SHAP Plot + Improvement Suggestions ---
                st.subheader(t.get("shap_plot","Model Explanation (SHAP Waterfall Plot)"))
                try:
                    explainer = shap.TreeExplainer(MODEL)
                    shap_values = explainer(X_processed)
                    plt.figure(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(plt)
                    plt.close('all')

                    # --- Generate improvement tips ---
                    tips = []
                    feature_map = {
                        'monthly_current_bill': 'Monthly Current Bill',
                        'family_recharge_bill': 'Family Recharge Bill',
                        'income': 'Annual Income',
                        'credit_score': 'Credit Score',
                        'loan_amount': 'Requested Loan Amount',
                        'years_employed': 'Years Employed'
                    }
                    for i, feature in enumerate(MODEL_FEATURES):
                        shap_value = shap_values.values[0][i]
                        readable_name = feature_map.get(feature, feature)
                        advice = ""
                        if feature in ['monthly_current_bill','family_recharge_bill','income']:
                            if shap_value < 0:
                                advice = f"Increase {readable_name} to improve approval chances."
                        elif feature == 'credit_score':
                            if shap_value < 0:
                                advice = "Try to improve credit score (≥700 recommended)."
                        elif feature == 'loan_amount':
                            if shap_value < 0:
                                advice = "Request lower loan amount to improve approval likelihood."
                        elif feature == 'years_employed':
                            if shap_value < 0:
                                advice = "Longer employment history improves approval."
                        if advice:
                            formula = f"SHAP Contribution ≈ {shap_value:.3f} impact on log-odds"
                            tips.append(f"- {readable_name}: {advice}\n  Formula: {formula}")
                    if tips:
                        st.info("\n\n".join(tips))
                    else:
                        st.success("✅ All key features positively contribute to approval. Keep it up!")
                except Exception as e:
                    st.warning(f"Could not generate SHAP improvement suggestions: {e}")

    # --- Data Visualization ---
    elif app_option == t["data_viz"]:
        st.header(t.get("data_viz","📊 Credit Data Visualization"))
        data_file = "indian_data_set.csv"
        try:
            df = pd.read_csv(data_file)
            visualize_credit_data(df, t)
        except FileNotFoundError:
            st.error(t.get("data_not_found", f"Data file '{data_file}' not found!"))

    # --- Community Insights ---
    elif app_option == t["community"]:
        st.header(t.get("community","🌐 Community / Policy Insights"))
        data_file = "indian_data_set.csv"
        try:
            df = pd.read_csv(data_file)
            community_insights(df, t)
        except FileNotFoundError:
            st.error(t.get("data_not_found", f"Data file '{data_file}' not found!"))

# --- Entry Point ---
if st.session_state['logged_in']:
    main_app()
else:
    login_page(TRANSLATIONS[st.session_state['language']])