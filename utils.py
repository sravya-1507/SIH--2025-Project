import pandas as pd
import re

# --- Function to extract bill amounts ---
def extract_bill_amount(file, t):
    """
    Extract monthly bill amount from CSV or Excel file
    """
    amount = 0
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return amount

        if 'Amount' in df.columns:
            amount = df['Amount'].sum()
        else:
            # fallback: sum the first numeric column
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    amount = df[col].sum()
                    break
    except Exception as e:
        print(f"Bill Extraction Error: {e}")

    return amount

# --- Function to extract ITR information ---
def extract_itr_info(file, t):
    """
    Extract basic info from Income Tax Return file (CSV or Excel)
    Returns: income (float), tax_paid (float), years_filed (list)
    """
    income, tax_paid, years_filed = 0, 0, []

    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return income, tax_paid, years_filed

        if 'Total Income' in df.columns:
            income = df['Total Income'].sum()
        if 'Tax Paid' in df.columns:
            tax_paid = df['Tax Paid'].sum()
        if 'Assessment Year' in df.columns:
            years_filed = df['Assessment Year'].unique().tolist()

    except Exception as e:
        print(f"ITR Extraction Error: {e}")

    return income, tax_paid, years_filed