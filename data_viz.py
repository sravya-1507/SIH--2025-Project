# ==========================================
# CREDIT SCORING DATA VISUALIZATION MODULE
# ==========================================
# Author: Twinkle Tita 🌸
# Works with: XGBoost / LightGBM predictions
# Libraries: pandas, matplotlib, seaborn, plotly
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def visualize_credit_data(df):
    """
    Function to visualize credit scoring data:
    - Distribution of composite scores or loan approvals
    - Risk band distribution (bar + pie)
    - Correlation heatmap
    - Interactive scatter and pie charts (Plotly)
    """
    # Ensure columns are lowercase and clean
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # --- 1. Credit Score / Loan Approval Distribution ---
    plt.figure(figsize=(8,5))
    if 'composite_score' in df.columns:
        plt.hist(df['composite_score'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Composite Score")
        plt.ylabel("Number of Beneficiaries")
        plt.title("Distribution of Composite Credit Scores")
    elif 'loan_approved' in df.columns:
        sns.countplot(x='loan_approved', data=df, palette='pastel')
        plt.title("Loan Approval Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 2. Risk Band Distribution (Bar + Pie) ---
    if 'risk_band' in df.columns and not df['risk_band'].isnull().all():
        counts = df['risk_band'].value_counts()

        # Bar Chart
        plt.figure(figsize=(7,5))
        sns.countplot(x='risk_band', data=df, palette='pastel', order=counts.index)
        plt.title("Beneficiaries by Risk Band")
        plt.xlabel("Risk Band")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Pie Chart
        plt.figure(figsize=(6,6))
        counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        plt.title("Risk Band Percentage Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    # --- 3. Correlation Heatmap ---
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10,8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # --- 4. Interactive Plotly Visualizations ---
    # Scatter Plot: Composite Score vs Repayment Rate
    if 'repayment_rate' in df.columns and 'composite_score' in df.columns:
        fig = px.scatter(df,
                         x='repayment_rate',
                         y='composite_score',
                         color='loan_approved' if 'loan_approved' in df.columns else 'risk_band',
                         size='loan_amount' if 'loan_amount' in df.columns else None,
                         hover_data=df.columns,
                         title="Composite Score vs Repayment Rate (Interactive)")
        fig.show()

    # Optional: Interactive Pie Chart using Plotly
    if 'risk_band' in df.columns:
        fig = px.pie(df, names='risk_band', title="Risk Band Percentage Distribution")
        fig.show()