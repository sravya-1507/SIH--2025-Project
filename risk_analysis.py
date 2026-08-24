# risk_analysis.py - Handles model loading, prediction, preprocessing, and DDL logic

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- DDL Constants ---
W_REPAY = 0.6
W_INCOME = 0.3
W_COMPLETENESS = 0.1
T_LOWRISK = 0.65
T_LOWINCOME = 0.35
SANCTIONABLE_SCORE = 0.8

# --- 1. Load Model & Artifacts ---
@st.cache_resource
def load_artifacts(t):
    """Loads the ML model and feature list."""
    try:
        model = joblib.load('loan_approval_model_xgboost.pkl')
        model_features = joblib.load('model_features.pkl')
        return model, model_features
    except FileNotFoundError:
        st.error(t.get("model_not_found", "⚠ Model files not found! Please run 'train_model.py' first."))
        st.stop()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

# --- 2. Preprocessing ---
def preprocess_input(input_dict, model_features):
    """Transforms raw input dictionary into the required format for the model."""
    input_df = pd.DataFrame([input_dict])
    input_df.columns = [col.strip().lower().replace(' ', '_') for col in input_df.columns]

    if 'previous_loan_paid' in input_df.columns:
        input_df['previous_loan_paid'] = input_df['previous_loan_paid'].apply(lambda x: 1 if x=="Yes" else 0)

    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    return input_df[model_features]

# --- 3. DDL Feature Functions ---
def compute_composite_score(repayment_score, income_score, completeness, itr_income_factor=1):
    """
    Calculates the composite DDL score.
    itr_income_factor: normalized ITR income / declared income (default=1 if no ITR)
    """
    repayment_score = repayment_score if repayment_score is not None else 0.5
    income_score = income_score if income_score is not None else 0.5
    completeness = completeness if completeness is not None else 0.8

    composite = W_REPAY*repayment_score + W_INCOME*income_score + W_COMPLETENESS*completeness
    composite *= itr_income_factor  # Adjust score based on ITR reliability
    return composite

def risk_band_classification(composite, income_score, t):
    """Classifies the beneficiary into a risk band."""
    lowrisk = composite >= T_LOWRISK
    lowincome = income_score <= T_LOWINCOME
    
    if lowrisk and lowincome:
        return t.get("lowrisk_highneed", "LowRisk-HighNeed 🟢")
    elif lowrisk and not lowincome:
        return t.get("lowrisk_lowneed", "LowRisk-LowNeed 🟡")
    elif not lowrisk and lowincome:
        return t.get("highrisk_highneed", "HighRisk-HighNeed 🟠")
    else:
        return t.get("highrisk_lowneed", "HighRisk-LowNeed 🔴")

def sanctionable_flag(composite, completeness, last_default_months=36):
    """Determines if the loan is eligible for automatic sanction."""
    if composite >= SANCTIONABLE_SCORE and completeness >= 0.8 and last_default_months > 24:
        return True
    return False