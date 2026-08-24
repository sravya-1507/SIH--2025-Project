# generate_data.py - Creates a clean CSV file for the project

import pandas as pd
import numpy as np

# Set seed for reproducibility (so the data is the same every time you run it)
np.random.seed(42)

N = 1000  # We will generate 1000 fictional beneficiaries

# --- 1. Create Raw Data (Features and Targets) ---
data = pd.DataFrame({
    'Beneficiary_ID': [f'B{i+1:04d}' for i in range(N)],
    'Age': np.random.randint(25, 65, N),
    'Loan_Amount': np.random.randint(20000, 150000, N),
    # Repayment Ratio: Average around 95%, some are lower (risky)
    'Repayment_Ratio': np.clip(np.random.normal(loc=0.95, scale=0.15, size=N), 0.1, 1.0),
    
    # Consumption Metrics (Proxies for Income/Need)
    'Energy_Usage_KWh': np.random.normal(loc=200, scale=80, size=N),
    'Mobile_Recharge_Freq': np.random.randint(2, 8, N),
    'Business_Type': np.random.choice(['Retail', 'Artisan', 'Agriculture', 'Service'], N, p=[0.3, 0.25, 0.25, 0.2]),
    'Gender': np.random.choice(['F', 'M'], N),
})

# --- 2. Engineer the Target Variable (RISK) ---
# Create the Default_Flag: 1 if high risk, 0 if low risk
# Logic: Default is likely if Repayment is poor AND Loan Amount is high.
default_logic = (data['Repayment_Ratio'] < 0.8) & (data['Loan_Amount'] > 80000)
default_noise = np.random.rand(N) < 0.05
data['Default_Flag'] = np.where(default_logic | default_noise, 1, 0) # 1 for default, 0 otherwise

# --- 3. Introduce Missing Data (Meeting the "Incomplete Data" Constraint) ---
# We intentionally set 10% of the consumption data to be missing (NaN)
data.loc[data.sample(frac=0.1).index, 'Energy_Usage_KWh'] = np.nan
data.loc[data.sample(frac=0.1).index, 'Mobile_Recharge_Freq'] = np.nan

# --- 4. Engineer the Need Category (NEED) ---
# This is simplified classification for the Low/Medium/High Need metric
conditions = [
    # High Need: Low energy use AND low recharge frequency
    (data['Energy_Usage_KWh'].fillna(data['Energy_Usage_KWh'].median()) <= 150) & (data['Mobile_Recharge_Freq'].fillna(5) <= 3),
    # Low Need: High energy use AND high recharge frequency
    (data['Energy_Usage_KWh'].fillna(data['Energy_Usage_KWh'].median()) > 300) & (data['Mobile_Recharge_Freq'].fillna(5) > 6)
]
choices = ['High_Need', 'Low_Need']
data['Need_Category'] = np.select(conditions, choices, default='Medium_Need')

# --- 5. Save the final data file ---
data.to_csv('lending_data.csv', index=False)
print("\n'lending_data.csv' created successfully with 1000 records.")