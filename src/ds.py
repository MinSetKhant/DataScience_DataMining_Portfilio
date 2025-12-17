# Healthcare Equity Explorer - Model Script
import argparse
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Define numerical columns used in the notebook
NUM_COLS = ['age', 'num_meds', 'total_med_cost', 'num_procedures', 
            'total_proc_cost', 'pain_score', 'height_cm', 'encounter_cost',
            'chronic_count', 'cost_ratio', 'proc_ratio'] 

# --- (engineer_features function remains the same) ---
def engineer_features(df):
    """Apply feature engineering consistent with the notebook pipeline."""
    chronic_feats = ['has_chronic_pain','has_hypertension','has_diabetes',
                     'has_asthma','has_depression']
    df['chronic_count'] = df[chronic_feats].sum(axis=1)

    df['cost_ratio'] = df['total_med_cost'] / (df['encounter_cost'] + 1)
    df['proc_ratio'] = df['num_procedures'] / (df['num_meds'] + 1)
    df['high_risk_flag'] = ((df['age'] > 70) | (df['chronic_count'] >= 3)).astype(int)
    
    # Handle age_group creation before OHE
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 90], 
                             labels=['Young', 'Middle-aged', 'Senior', 'Elderly'],
                             include_lowest=True)
    df['age_group'] = df['age_group'].cat.add_categories('Unknown').fillna('Unknown')
    
    cat_cols = ['gender','race','ethnicity','payer_type','age_group']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

# --------------------------------------------
# Train Function (No changes needed here, it was correct)
# --------------------------------------------
def train_healthcare(train_path, dev_path, model_path="model.pkl", scaler_path="scaler.pkl"):
    """Train XGBoost model, save model and scaler."""
    print("Loading datasets ...")
    # Assuming the clean data is the source for robustness
    train_df = pd.read_csv(train_path)

    # 1. Feature Engineering
    train_df = engineer_features(train_df)
    
    # Define X and y
    X = train_df.drop(columns=['encounter_id', 'patient_id', 'readmitted_within_30_days', 'zip'])
    y = train_df['readmitted_within_30_days']

    # 2. Standardization
    print("Fitting and saving StandardScaler...")
    scaler = StandardScaler()
    # Note: We fit only on training data (X), then transform
    X[NUM_COLS] = scaler.fit_transform(X[NUM_COLS])

    # 3. Model Training (Use your known good parameters from the notebook)
    print("Training final XGBoost model...")
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, random_state=42, 
        eval_metric='auc', use_label_encoder=False # Recommended to suppress warning
    )
    xgb.fit(X, y)

    # 4. Save Artifacts
    print(f" Saving Model to {model_path} and Scaler to {scaler_path}")
    joblib.dump(xgb, model_path)
    joblib.dump(scaler, scaler_path)

# --------------------------------------------
# Predict Function (CRITICAL FIXES APPLIED HERE)
# --------------------------------------------
def predict_healthcare(input_path, output_path="submission.csv", model_path="model.pkl", scaler_path="scaler.pkl"):
    """Generate predictions using the saved model and scaler."""
    print(f"Loading Model from {model_path} and Scaler from {scaler_path}...")
    
    try:
        # 1. Load Artifacts
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        print(f" ERROR: Could not find required model or scaler file: {e}")
        return

    # 2. Load Input Data
    df = pd.read_csv(input_path)
    
    # CRITICAL FIX 1: Capture the original IDs to enforce submission order
    original_ids = df['encounter_id'].reset_index(drop=True)

    # 3. Feature Engineering
    df = engineer_features(df)
    
    # 4. Prepare Features (Drop non-feature columns and ensure index is clean)
    cols_to_drop = ['encounter_id', 'patient_id', 'zip', 'readmitted_within_30_days'] 
    
    # We explicitly reset the index after dropping columns to guarantee X is ordered 0, 1, 2...
    X = df.drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)

    # 5. Standardization
    X[NUM_COLS] = scaler.transform(X[NUM_COLS])

    # 6. Generate Predictions
    print("Generating predictions ...")
    preds = model.predict(X)

    # CRITICAL FIX 2: Use the original_ids captured at the start
    submission = pd.DataFrame({
        "encounter_id": original_ids,
        "readmitted_within_30_days": preds
    })
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# --------------------------------------------
# CLI Entry Point
# --------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare Equity Explorer Model Trainer & Predictor")
    parser.add_argument("command", choices=["train","predict"], help="Train a model or generate predictions.")
    # Set model/scaler paths to the ROOT by default
    parser.add_argument("--model_path", default="model.pkl", help="Path to save/load model.")
    parser.add_argument("--scaler_path", default="scaler.pkl", help="Path to save/load scaler.") 
    parser.add_argument("--train_path", default="cleaned_train.csv", help="Path to training dataset CSV.")
    parser.add_argument("--input_path", help="Path to dataset for prediction (test.csv).")
    parser.add_argument("--output_path", default="submission.csv", help="Path to save predictions.")
    args = parser.parse_args()

    if args.command == "train":
        train_healthcare(args.train_path, None, args.model_path, args.scaler_path) 
    elif args.command == "predict":
        predict_healthcare(args.input_path, args.output_path, args.model_path, args.scaler_path)