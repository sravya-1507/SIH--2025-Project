# train_model.py — CORRECTED Version

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- CONFIG ---
DATA_FILE_PATH = 'modified_loan_dataset.csv'  # ← new dataset name
TARGET_COLUMN = 'loan_approved'               # keep the same (correct column name)
MODEL_OUTPUT = 'loan_approval_model_xgboost.pkl'
FEATURES_OUTPUT = 'model_features.pkl'


# --- 1. Load Data ---
df = pd.read_csv(DATA_FILE_PATH)
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
print(f"✅ Data loaded. Shape: {df.shape}")

# --- 2. Drop non-predictive columns (as you intended) ---
for col_to_drop in ['full_name', 'city', 'points']:
    if col_to_drop in df.columns:
        df = df.drop(columns=[col_to_drop])
        print(f"🧹 Dropped column: {col_to_drop}")

# --- 3. Encode 'previous_loan_paid' Yes/No ---
if 'previous_loan_paid' in df.columns:
    df['previous_loan_paid'] = df['previous_loan_paid'].map({'Yes': 1, 'No': 0})
    print("🔄 Encoded 'previous_loan_paid' (Yes=1, No=0)")


# --- 4. Convert all columns except target to numeric ---
for col in df.columns:
    if col != TARGET_COLUMN:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaN values (from conversion errors) with 0
df = df.fillna(0)
print("🧼 Cleaned and filled NaN values.")


# ******************************************************
# --- 4.5 DEFINE FEATURES (X) AND TARGET (Y) HERE ---
# ******************************************************
if TARGET_COLUMN in df.columns:
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
else:
    print(f"❌ Error: Target column '{TARGET_COLUMN}' not found.")
    exit()

# Save feature names
joblib.dump(X.columns.tolist(), FEATURES_OUTPUT)
print(f"📝 Feature list saved: {FEATURES_OUTPUT}")
print(f"Features used in training: {X.columns.tolist()}") # SHOW ALL FEATURES


# --- 5. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🧠 Training on {X_train.shape[0]} samples.")

# --- 6. Train XGBoost ---
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42,
    'n_estimators': 200,
    'learning_rate': 0.05
}

model = xgb.XGBClassifier(**params)
print("🚀 Training XGBoost...")
model.fit(X_train, y_train)
print("✅ Model trained successfully.")

# --- 7. Evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# --- 8. Save model ---
joblib.dump(model, MODEL_OUTPUT)
print(f"💾 Model saved: {MODEL_OUTPUT}")
print("🏁 Training complete. All intended features included! ✅")