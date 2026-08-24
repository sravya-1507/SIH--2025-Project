# visualization.py - Handles all data visualization and dashboard displays

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# A helper to set up plots for Streamlit
def plot_config():
    """Configures matplotlib and clears the figure."""
    plt.close('all')
    plt.figure()

# --- 1. Credit Data Visualization ---
def visualize_credit_data(df, t):
    """Displays various credit data visualizations."""
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # 1. Distribution Plot
    st.subheader(t.get("credit_distribution", "1️⃣ Credit Score / Loan Approval Distribution"))
    plot_config()
    if 'composite_score' in df.columns:
        plt.hist(df['composite_score'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel(t.get("composite_score", "Composite Score"))
        plt.ylabel(t.get("num_beneficiaries", "Number of Beneficiaries"))
        plt.title(t.get("distribution_title", "Distribution of Composite Credit Scores"))
        st.pyplot(plt)
    elif 'loan_approved' in df.columns:
        sns.countplot(x='loan_approved', data=df, palette='pastel')
        plt.title(t.get("loan_approval_distribution", "Loan Approval Distribution"))
        st.pyplot(plt)
    plt.close() # Close figure after showing

    # 2. Risk Band Distribution Plots
    if 'risk_band' in df.columns and not df['risk_band'].isnull().all():
        counts = df['risk_band'].value_counts()
        
        st.subheader(t.get("risk_band_bar", "2️⃣ Risk Band Distribution (Bar Chart)"))
        plot_config()
        plt.figure(figsize=(7,5))
        sns.countplot(x='risk_band', data=df, palette='pastel', order=counts.index)
        plt.title(t.get("beneficiaries_by_risk", "Beneficiaries by Risk Band"))
        st.pyplot(plt)
        plt.close()

        st.subheader(t.get("risk_band_pie", "Risk Band Distribution (Matplotlib Pie Chart)"))
        plot_config()
        plt.figure(figsize=(6,6))
        counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        plt.ylabel("")
        st.pyplot(plt)
        plt.close()

    # 3. Feature Correlation Heatmap
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    if len(numeric_cols) > 1:
        st.subheader(t.get("feature_correlation", "3️⃣ Feature Correlation Heatmap"))
        plot_config()
        plt.figure(figsize=(10,8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
        plt.close()

    # 4. Interactive Pie Chart
    if 'risk_band' in df.columns and not df['risk_band'].isnull().all():
        st.subheader(t.get("interactive_pie", "Interactive Pie Chart: Risk Band Percentage (Plotly)"))
        fig = px.pie(df, names='risk_band', title=t.get("risk_band_percentage", "Risk Band Percentage Distribution"))
        st.plotly_chart(fig, use_container_width=True)

# --- 2. Community/Policy Insights Dashboard ---
def community_insights(df, t):
    """Displays policy and community-level insights visualizations."""
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    st.subheader(t.get("community_insights", "🌐 Community / Policy Insights"))
    
    if 'loan_approved' not in df.columns:
         st.warning("Cannot generate policy insights as 'loan_approved' column is missing.")
         return

    # Loan Approval by City
    if 'city' in df.columns:
        st.markdown(t.get("loan_by_city", "Loan Approval by City"))
        city_counts = df.groupby('city')['loan_approved'].mean().sort_values(ascending=False).reset_index()
        city_counts['approval_rate'] = city_counts['loan_approved'] * 100 # Convert to percentage
        fig = px.bar(city_counts, x='city', y='approval_rate',
                     labels={'city':t.get("city","City"),'approval_rate':t.get("approval_rate","Approval Rate (%)")},
                     title=t.get("citywise_title","City-wise Loan Approval Rate"))
        st.plotly_chart(fig, use_container_width=True)

    # Loan Approval by Gender
    if 'gender' in df.columns:
        st.markdown(t.get("loan_by_gender","Loan Approval by Gender"))
        gender_counts = df.groupby('gender')['loan_approved'].mean().reset_index()
        gender_counts['approval_rate'] = gender_counts['loan_approved'] * 100
        fig = px.bar(gender_counts, x='gender', y='approval_rate',
                     labels={'gender':t.get("gender","Gender"),'approval_rate':t.get("approval_rate","Approval Rate (%)")},
                     title=t.get("genderwise_title","Gender-wise Loan Approval Rate"))
        st.plotly_chart(fig, use_container_width=True)

    # Beneficiaries Distribution by Risk Band
    if 'risk_band' in df.columns:
        st.markdown(t.get("beneficiaries_risk","Beneficiaries Distribution by Risk Band"))
        risk_counts = df['risk_band'].value_counts().reset_index()
        risk_counts.columns = ['risk_band', 'count']
        fig = px.pie(risk_counts, names='risk_band', values='count',
                     title=t.get("risk_band_community","Risk Band Distribution in Community"))
        st.plotly_chart(fig, use_container_width=True)