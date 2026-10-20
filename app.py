# app.py - Credit Scoring System
# Loan Prediction + Risk Band + Community Insights + Multilingual + Mock KYC + SHAP Explanations

import os
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import numpy as np

from translations import TRANSLATIONS
from auth import login_page, logout
from utils import extract_bill_amount, extract_itr_info
from risk_analysis import (
    load_artifacts,
    preprocess_input,
    compute_composite_score,
    risk_band_classification,
    sanctionable_flag,
)
from visualization import visualize_credit_data, community_insights


# Configure Streamlit exactly once, before other Streamlit commands.
st.set_page_config(page_title="Credit Scoring System", layout="wide")


# ---------------------------------------------------------------------
# Mock KYC database. These are demo-only values, not real verification.
# ---------------------------------------------------------------------
MOCK_PAN_DB = {
    "ABCDE1234F": {"name": "John Doe", "status": "Valid"},
    "FGHIJ5678K": {"name": "Jane Smith", "status": "Valid"},
}

MOCK_AADHAAR_DB = {
    "123456789012": {"status": "Valid"},
    "987654321098": {"status": "Valid"},
}

# Human-friendly feature names used in SHAP explanations.
FEATURE_LABELS = {
    "monthly_current_bill": "Monthly Current Bill",
    "family_recharge_bill": "Family Recharge Bill",
    "income": "Annual Income",
    "credit_score": "Credit Score",
    "loan_amount": "Requested Loan Amount",
    "years_employed": "Years Employed",
    "previous_loan_paid": "Previous Loan Paid",
    "user_itr_income": "ITR Income",
    "user_itr_tax": "ITR Tax Paid",
    "user_itr_years": "ITR Years Filed",
}


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
DEFAULTS = {
    "language": "English",
    "logged_in": False,
    "chat_support_loaded": False,
    "pan_status": None,
    "aadhaar_status": None,
    # Holds the last submitted prediction inputs so results survive reruns
    # (for example when the PAN/Aadhaar verify buttons are clicked).
    "prediction_inputs": None,
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ---------------------------------------------------------------------
# Language selector and model loading
# ---------------------------------------------------------------------
st.sidebar.selectbox(
    "🌐 Select Language",
    list(TRANSLATIONS.keys()),
    key="language",
)
t = TRANSLATIONS[st.session_state["language"]]

# Load the model once per Streamlit cache key. This also avoids repeatedly
# loading model files on every rerun if load_artifacts is cache-compatible.
MODEL, MODEL_FEATURES = load_artifacts(t)


# ---------------------------------------------------------------------
# Mock KYC callbacks
# ---------------------------------------------------------------------
def verify_pan_mock():
    pan_number = str(st.session_state.get("pan_input", "")).strip().upper()
    if pan_number in MOCK_PAN_DB:
        applicant = MOCK_PAN_DB[pan_number]
        st.session_state["pan_status"] = (
            f"✅ PAN is valid! Name on record: {applicant['name']}"
        )
    else:
        st.session_state["pan_status"] = (
            "❌ PAN not found. Prototype only verifies the predefined list."
        )


def verify_aadhaar_mock():
    aadhaar_number = str(st.session_state.get("aadhaar_input", "")).strip()
    if aadhaar_number in MOCK_AADHAAR_DB:
        st.session_state["aadhaar_status"] = "✅ Aadhaar is valid!"
    else:
        st.session_state["aadhaar_status"] = (
            "❌ Aadhaar not registered. Prototype only verifies the predefined list."
        )


def render_status(status):
    """Render a saved mock-verification status safely."""
    if not status:
        return
    if str(status).startswith("✅"):
        st.success(status)
    else:
        st.error(status)


# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------
def _to_float(value):
    """Convert a SHAP value (number, string like '[4.34E-1]', or 1-element array) to float."""
    if isinstance(value, (list, tuple, np.ndarray)):
        flattened = np.asarray(value, dtype=object).ravel()
        if len(flattened) != 1:
            raise ValueError(f"Unexpected nested SHAP value: {value!r}")
        value = flattened[0]
    if isinstance(value, str):
        value = value.strip().strip("[]").strip()
    return float(value)


def get_shap_values_for_one_prediction(model, X):
    """
    Return a single-row shap.Explanation (1-D values, 1-D data, scalar base value)
    plus a 1-D numpy array of the SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)

    # Some model/SHAP combinations return [samples, features, classes].
    if values.ndim == 3:
        class_index = 1 if values.shape[-1] > 1 else 0
        values = values[:, :, class_index]
        if base_values.ndim >= 2:
            base_values = base_values[:, class_index]

    if values.ndim == 2:
        values = values[0]
    elif values.ndim != 1:
        raise ValueError(
            f"Unexpected SHAP values shape: {np.asarray(explanation.values).shape}"
        )

    numeric_values = np.asarray(
        [_to_float(v) for v in np.asarray(values, dtype=object).ravel()],
        dtype=float,
    )

    if len(numeric_values) != len(MODEL_FEATURES):
        raise ValueError(
            "SHAP feature count does not match model feature count: "
            f"{len(numeric_values)} vs {len(MODEL_FEATURES)}."
        )

    base_value = _to_float(np.asarray(base_values, dtype=object).ravel()[0])

    if hasattr(X, "iloc"):
        data_row = X.iloc[0].to_numpy()
    else:
        data_row = np.asarray(X)[0]
    try:
        data_row = np.asarray(data_row, dtype=float)
    except (TypeError, ValueError):
        data_row = np.asarray(data_row, dtype=object)

    single_explanation = shap.Explanation(
        values=numeric_values,
        base_values=base_value,
        data=data_row,
        feature_names=list(MODEL_FEATURES),
    )
    return single_explanation, numeric_values


def friendly_name(feature):
    return FEATURE_LABELS.get(feature, str(feature).replace("_", " ").title())


def format_value(value):
    """Format a feature value for display."""
    try:
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def get_display_value(feature, X, user_input_dict):
    """Prefer the applicant's raw entered value; fall back to the processed value."""
    if feature in user_input_dict:
        return format_value(user_input_dict[feature])
    if hasattr(X, "columns") and feature in X.columns:
        return format_value(X.iloc[0][feature])
    return "-"


def predict_probability(user_input_dict):
    """Approval probability for a set of raw inputs (used for what-if scenarios)."""
    X = preprocess_input(user_input_dict, MODEL_FEATURES)
    return float(MODEL.predict_proba(X)[0, 1])


def build_what_if_table(user_input_dict, current_prob):
    """Re-run the model with changed inputs to show real probability changes."""
    scenarios = []
    credit_score = int(user_input_dict["credit_score"])
    loan_amount = int(user_input_dict["loan_amount"])

    if credit_score < 850:
        scenarios.append(
            (
                "Credit score +50",
                {"credit_score": min(850, credit_score + 50)},
            )
        )
        scenarios.append(
            (
                "Credit score +100",
                {"credit_score": min(850, credit_score + 100)},
            )
        )
    if loan_amount > 1_000:
        scenarios.append(
            (
                "Loan amount -25%",
                {"loan_amount": max(1_000, int(loan_amount * 0.75))},
            )
        )
        scenarios.append(
            (
                "Loan amount -50%",
                {"loan_amount": max(1_000, int(loan_amount * 0.50))},
            )
        )
    if credit_score < 850 and loan_amount > 1_000:
        scenarios.append(
            (
                "Credit score +50 and loan amount -25%",
                {
                    "credit_score": min(850, credit_score + 50),
                    "loan_amount": max(1_000, int(loan_amount * 0.75)),
                },
            )
        )

    rows = []
    for label, changes in scenarios:
        modified = {**user_input_dict, **changes}
        try:
            new_prob = predict_probability(modified)
        except Exception:
            continue
        rows.append(
            {
                "Scenario": label,
                "New approval probability (%)": round(new_prob * 100, 1),
                "Change (percentage points)": round((new_prob - current_prob) * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def render_shap_section(t, X_processed, user_input_dict, prob_approval):
    """Explain the prediction: contribution chart, plain-language summary, what-if."""
    st.subheader(t.get("shap_plot", "Why did the model decide this?"))

    try:
        explanation, shap_values = get_shap_values_for_one_prediction(
            MODEL, X_processed
        )
    except Exception as shap_error:
        st.warning(
            "The loan prediction succeeded, but the explanation could not be "
            f"generated: {shap_error}"
        )
        with st.expander("Show SHAP technical details"):
            st.exception(shap_error)
        return

    base_value = float(np.asarray(explanation.base_values).ravel()[0])

    # Tree models often explain in log-odds; random forests explain in probability.
    raw_output = base_value + float(shap_values.sum())
    in_probability_space = abs(raw_output - prob_approval) < 1e-3
    unit = "probability" if in_probability_space else "log-odds"

    rows = []
    for feature, shap_value in zip(MODEL_FEATURES, shap_values):
        rows.append(
            {
                "feature": feature,
                "name": friendly_name(feature),
                "value": get_display_value(feature, X_processed, user_input_dict),
                "impact": float(shap_value),
            }
        )
    df = pd.DataFrame(rows)
    df["abs_impact"] = df["impact"].abs()
    df = df.sort_values("abs_impact", ascending=False).reset_index(drop=True)

    # ---- Plain-language summary ----
    helping = df[df["impact"] > 0].head(3)
    hurting = df[df["impact"] < 0].head(3)

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown("#### ✅ Factors helping approval")
        if helping.empty:
            st.write("No factors increased the approval score.")
        for _, row in helping.iterrows():
            st.markdown(f"- **{row['name']}** ({row['value']}) raised the score")
    with summary_col2:
        st.markdown("#### ⚠ Factors hurting approval")
        if hurting.empty:
            st.write("No factors reduced the approval score.")
        for _, row in hurting.iterrows():
            st.markdown(f"- **{row['name']}** ({row['value']}) lowered the score")

    # ---- Contribution bar chart ----
    chart_df = df.sort_values("impact", ascending=True)
    bar_fig = go.Figure(
        go.Bar(
            x=chart_df["impact"],
            y=[f"{n} = {v}" for n, v in zip(chart_df["name"], chart_df["value"])],
            orientation="h",
            marker_color=[
                "#2e9e5b" if v > 0 else "#d64545" for v in chart_df["impact"]
            ],
            text=[f"{v:+.3f}" for v in chart_df["impact"]],
            textposition="outside",
            hovertemplate="%{y}<br>Impact: %{x:+.3f}<extra></extra>",
        )
    )
    bar_fig.update_layout(
        title=f"How each input moved the result (impact in {unit})",
        xaxis_title=f"Impact on approval score ({unit}). Right = helps, left = hurts",
        yaxis_title="",
        height=max(320, 60 * len(chart_df)),
        margin=dict(l=10, r=40, t=60, b=40),
    )
    bar_fig.add_vline(x=0, line_width=1, line_color="gray")
    st.plotly_chart(bar_fig, use_container_width=True)

    if not in_probability_space:
        st.caption(
            "Impacts are in log-odds: positive values push toward approval, "
            "negative values push toward decline. Larger bars mean a bigger effect."
        )

    # ---- What-if scenarios ----
    st.markdown("#### 🔮 What could improve the result?")
    what_if_df = build_what_if_table(user_input_dict, prob_approval)
    if what_if_df.empty:
        st.info("No what-if scenarios are available for these inputs.")
    else:
        st.dataframe(what_if_df, use_container_width=True, hide_index=True)
        best = what_if_df.sort_values(
            "Change (percentage points)", ascending=False
        ).iloc[0]
        if best["Change (percentage points)"] > 0:
            st.success(
                f"Biggest simulated improvement: **{best['Scenario']}** "
                f"({best['Change (percentage points)']:+.1f} percentage points)."
            )
        else:
            st.info(
                "None of these simple changes improves the approval probability."
            )
        st.caption(
            "These are model simulations, not guarantees. They change only the "
            "listed input and keep everything else the same."
        )

    # ---- Detailed table ----
    with st.expander("Show detailed contribution table"):
        table = df[["name", "value", "impact"]].rename(
            columns={
                "name": "Feature",
                "value": "Your value",
                "impact": f"Impact ({unit})",
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(f"Model baseline ({unit}): {base_value:.3f}")

    # ---- Original SHAP waterfall (optional) ----
    with st.expander("Show SHAP waterfall plot"):
        try:
            plt.close("all")
            # Do not call plt.figure() first: waterfall manages its own figure.
            shap.plots.waterfall(explanation, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close("all")
        except Exception as plot_error:
            plt.close("all")
            st.warning(f"Waterfall plot could not be drawn: {plot_error}")


# ---------------------------------------------------------------------
# Section renderers (called from session state so they survive reruns)
# ---------------------------------------------------------------------
def render_kyc_section(t):
    """Mock PAN / Aadhaar verification forms."""
    st.markdown("---")
    st.header(t.get("document_verification", "2. Document Verification (Prototype)"))
    kyc_col1, kyc_col2 = st.columns(2)

    with kyc_col1:
        with st.form("pan_verification_form"):
            st.subheader("Verify PAN Card")
            st.info("💡 For this demo, use: " + ", ".join(MOCK_PAN_DB.keys()))
            st.text_input(
                "Enter PAN Number",
                key="pan_input",
                placeholder="e.g., ABCDE1234F",
            )
            st.form_submit_button("Verify PAN (Mock)", on_click=verify_pan_mock)
        render_status(st.session_state.get("pan_status"))

    with kyc_col2:
        with st.form("aadhaar_verification_form"):
            st.subheader("Verify Aadhaar Card")
            st.info("💡 For this demo, use: " + ", ".join(MOCK_AADHAAR_DB.keys()))
            st.text_input(
                "Enter Aadhaar Number",
                key="aadhaar_input",
                placeholder="e.g., 123456789012",
            )
            st.form_submit_button(
                "Verify Aadhaar (Mock)", on_click=verify_aadhaar_mock
            )
        render_status(st.session_state.get("aadhaar_status"))

    st.markdown("---")


def render_prediction_results(t, saved):
    """Run the model on the saved inputs and render all result widgets."""
    full_name = saved["full_name"]
    user_input_dict = saved["user_input_dict"]
    income = user_input_dict["income"]

    try:
        X_processed = preprocess_input(user_input_dict, MODEL_FEATURES)
        probabilities = MODEL.predict_proba(X_processed)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                "The loaded model does not return two class probabilities."
            )

        prob_approval = float(probabilities[0, 1])
        approval_score = int(round(prob_approval * 100))
        status = t["approved"] if approval_score >= 50 else t["declined"]
        points = int(round((1 - prob_approval) * 100))

        completeness = 0.9
        income_score = income / 1_000_000
        composite_score = compute_composite_score(
            prob_approval, income_score, completeness
        )
        risk_band = risk_band_classification(composite_score, income_score, t)
        auto_sanction = sanctionable_flag(composite_score, completeness)

        st.markdown(f"## {t['result_title']} {full_name}")
        st.metric(
            t.get("approval_prob", "Loan Approval Probability (%)"),
            approval_score,
        )
        st.success(f"{t.get('loan_status', 'Loan Status')}: {status}")
        st.info(f"{t.get('risk_points', 'Risk Points')}: {points}")
        st.markdown(f"### {t.get('risk_band', '🧮 Risk Band')}: {risk_band}")
        sanction_text = (
            "✅ " + t.get("approved", "Approved")
            if auto_sanction
            else "⚠ " + t.get("manual_review", "Requires Manual Review")
        )
        st.markdown(
            f"### {t.get('auto_sanction', '💵 Automatic Loan Sanction')}: "
            f"{sanction_text}"
        )

        # Gauge chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=approval_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": t.get("approval_prob", "Loan Approval Probability (%)"),
                    "font": {"size": 20},
                },
                delta={
                    "reference": 50,
                    "increasing": {"color": "green"},
                    "decreasing": {"color": "red"},
                },
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": (
                            "green"
                            if approval_score >= 80
                            else "gold"
                            if approval_score >= 50
                            else "red"
                        )
                    },
                    "steps": [
                        {"range": [0, 50], "color": "rgba(255,0,0,0.3)"},
                        {"range": [50, 80], "color": "rgba(255,215,0,0.3)"},
                        {"range": [80, 100], "color": "rgba(0,255,0,0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "green", "width": 6},
                        "thickness": 0.8,
                        "value": approval_score,
                    },
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # SHAP explanation, summary and what-if scenarios
        render_shap_section(t, X_processed, user_input_dict, prob_approval)

    except Exception as prediction_error:
        st.error(f"Could not generate loan prediction: {prediction_error}")
        with st.expander("Show prediction technical details"):
            st.exception(prediction_error)


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------
def main_app():
    t = TRANSLATIONS[st.session_state["language"]]
    st.title(t["loan_prediction"])

    with st.sidebar:
        st.title(t["menu_title"])
        app_option = st.radio(
            t.get("select_page", "Select Page:"),
            [t["loan_prediction"], t["data_viz"], t["community"]],
        )
        st.button(t["logout"], on_click=logout, type="secondary")
        st.markdown("---")
        st.title(t.get("chat_support", "💬 Chat Support"))

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
            unsafe_allow_html=True,
        )

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if not st.session_state["chat_support_loaded"]:
            st.session_state["chat_support_loaded"] = True

        components.html(
            """
            <script>
                window.chtlConfig = { chatbotId: "6317658313" }
            </script>
            <script async data-id="6317658313" id="chtl-script" type="text/javascript"
                src="https://chatling.ai/js/embed.js"></script>
            """,
            height=600,
            scrolling=False,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Loan prediction
    # -----------------------------------------------------------------
    if app_option == t["loan_prediction"]:
        with st.form("applicant_form"):
            st.header(t.get("applicant_data", "1. Applicant Data Input"))
            col1, col2 = st.columns(2)

            with col1:
                full_name = st.text_input(
                    t.get("full_name", "Applicant's Full Name")
                )
                city = st.text_input(t.get("city_residence", "City of Residence"))

                current_bill_file = st.file_uploader(
                    t.get(
                        "upload_current_bill",
                        "Upload Monthly Current Bill (CSV/XLSX)",
                    ),
                    type=["csv", "xlsx"],
                    key="current_bill_file",
                )
                if current_bill_file is not None:
                    monthly_current_bill = extract_bill_amount(current_bill_file, t)
                    st.success(
                        f"{t.get('bill_extracted', 'Monthly Current Bill extracted')}: "
                        f"{monthly_current_bill}"
                    )
                else:
                    monthly_current_bill = st.number_input(
                        t.get("current_bill", "Monthly Current Bill (INR) *"),
                        min_value=0,
                        max_value=30_000,
                        value=0,
                        step=10,
                    )

                previous_loan_paid = st.selectbox(
                    t.get("previous_loan", "Previous Loan Paid?"),
                    ["Yes", "No"],
                )

            with col2:
                recharge_bill_file = st.file_uploader(
                    t.get(
                        "upload_recharge_bill",
                        "Upload Family Recharge Bill (CSV/XLSX)",
                    ),
                    type=["csv", "xlsx"],
                    key="recharge_bill_file",
                )
                if recharge_bill_file is not None:
                    family_recharge_bill = extract_bill_amount(
                        recharge_bill_file, t
                    )
                    st.success(
                        f"{t.get('bill_extracted', 'Family Recharge Bill extracted')}: "
                        f"{family_recharge_bill}"
                    )
                else:
                    family_recharge_bill = st.number_input(
                        t.get("recharge_bill", "Family Recharge Bill (INR) *"),
                        min_value=0,
                        max_value=30_000,
                        value=0,
                        step=10,
                    )

                income = st.number_input(
                    t.get("annual_income", "Annual Income (INR) *"),
                    min_value=0,
                    max_value=10_000_000,
                    value=0,
                    step=1_000,
                )
                credit_score = st.number_input(
                    t.get("credit_score", "Credit Score"),
                    min_value=300,
                    max_value=850,
                    value=650,
                    step=1,
                )
                loan_amount = st.number_input(
                    t.get("loan_amount", "Loan Amount Requested (INR)"),
                    min_value=1_000,
                    max_value=500_000,
                    value=50_000,
                    step=1_000,
                )
                years_employed = st.number_input(
                    t.get("years_employed", "Number of Years Employed"),
                    min_value=0,
                    max_value=50,
                    value=5,
                    step=1,
                )

                itr_file = st.file_uploader(
                    t.get("upload_itr", "Upload Income Tax Return (CSV/XLSX)"),
                    type=["csv", "xlsx"],
                    key="itr_file",
                )
                itr_income, itr_tax, itr_years = 0, 0, []
                if itr_file is not None:
                    itr_income, itr_tax, itr_years = extract_itr_info(itr_file, t)
                    st.success(
                        f"{t.get('itr_info', 'ITR extracted')}: "
                        f"Income={itr_income}, Tax Paid={itr_tax}, "
                        f"Years Filed={len(itr_years)}"
                    )
                else:
                    st.info(t.get("itr_missing", "No ITR uploaded"))

            st.markdown("---")
            predict_button = st.form_submit_button(
                t.get("predict", "Predict Loan Approval")
            )

        # Only the button click validates and saves the inputs. The results
        # below are drawn from session state, so they persist across reruns.
        if predict_button:
            if (
                monthly_current_bill <= 0
                or family_recharge_bill <= 0
                or income <= 0
            ):
                st.session_state["prediction_inputs"] = None
                st.error(
                    "⚠ Please fill all mandatory fields marked with * before "
                    "prediction (Monthly Current Bill, Family Recharge Bill, "
                    "Annual Income)."
                )
            else:
                st.session_state["prediction_inputs"] = {
                    "full_name": full_name,
                    "user_input_dict": {
                        "monthly_current_bill": monthly_current_bill,
                        "family_recharge_bill": family_recharge_bill,
                        "previous_loan_paid": previous_loan_paid,
                        "income": income,
                        "credit_score": credit_score,
                        "loan_amount": loan_amount,
                        "years_employed": years_employed,
                        "user_itr_income": itr_income,
                        "user_itr_tax": itr_tax,
                        "user_itr_years": len(itr_years),
                    },
                }

        saved_inputs = st.session_state.get("prediction_inputs")
        if saved_inputs:
            render_kyc_section(t)
            render_prediction_results(t, saved_inputs)

    # -----------------------------------------------------------------
    # Data visualization
    # -----------------------------------------------------------------
    elif app_option == t["data_viz"]:
        st.header(t.get("data_viz", "📊 Credit Data Visualization"))
        data_file = "indian_data_set.csv"
        try:
            df = pd.read_csv(data_file)
            visualize_credit_data(df, t)
        except FileNotFoundError:
            st.error(t.get("data_not_found", f"Data file '{data_file}' not found!"))
        except Exception as data_error:
            st.error(f"Could not load credit data: {data_error}")

    # -----------------------------------------------------------------
    # Community insights
    # -----------------------------------------------------------------
    elif app_option == t["community"]:
        st.header(t.get("community", "🌐 Community / Policy Insights"))
        data_file = "indian_data_set.csv"
        try:
            df = pd.read_csv(data_file)
            community_insights(df, t)
        except FileNotFoundError:
            st.error(t.get("data_not_found", f"Data file '{data_file}' not found!"))
        except Exception as data_error:
            st.error(f"Could not load community insights: {data_error}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if st.session_state["logged_in"]:
    main_app()
else:
    login_page(TRANSLATIONS[st.session_state["language"]])