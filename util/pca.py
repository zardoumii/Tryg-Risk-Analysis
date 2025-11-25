import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# ================= CONFIGURATION =================
# Use the cleaned file (before it was fully processed/scaled) to have full control
INPUT_FILE = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_train_cleaned.csv"
SAVE_DIR = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\results"
OUTPUT_DATA_FILE = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_data_hybrid_pca.csv"

# Define your column groups
NUMERICAL_COLS = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
CATEGORICAL_COLS = ['Area', 'VehBrand', 'VehGas', 'Region']
TARGET_COL = 'ClaimFrequency' # We need to create this if it doesn't exist yet
# =================================================

def run_hybrid_pca():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 0. Ensure Target Variable Exists (same logic as before)
    if 'ClaimFrequency' not in df.columns:
        df['ClaimFrequency'] = df['ClaimNb'] / df['Exposure'].clip(lower=0.001)

    # 1. Prepare Numerical Data for PCA
    print(f"\nProcessing Numerical Columns: {NUMERICAL_COLS}")
    X_num = df[NUMERICAL_COLS].copy()
    
    # Log transform skew variables (optional but recommended for these specific vars)
    for col in ['Density', 'Exposure']:
         X_num[col] = np.log1p(X_num[col])

    # Scale
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # 2. Run PCA on Numerical Only
    # We have 6 numerical features. Let's see how much 5 components cover.
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_num_scaled)
    
    # Calculate Variance
    explained_var = pca.explained_variance_ratio_ * 100
    total_var = explained_var.sum()
    
    print("\n=== PCA RESULTS (Numerical Only) ===")
    for i, var in enumerate(explained_var, 1):
        print(f"  PC{i}: {var:.2f}%")
    print(f"Total Variance Explained: {total_var:.2f}%")
    
    # Plot Scree
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Scree Plot (Numerical Features Only)\nTotal Explained: {total_var:.2f}%')
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'hybrid_pca_scree.png'))
    plt.close()
    
    # 3. Create PCA DataFrame
    pca_cols = [f'PC{i+1}' for i in range(5)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    
    # 4. Process Categorical Data (One-Hot Encode)
    print(f"\nProcessing Categorical Columns: {CATEGORICAL_COLS}")
    df_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    
    # 5. Combine: PCA Features + Encoded Categoricals + Target
    print("Combining datasets...")
    df_final = pd.concat([df_pca, df_cat, df[[TARGET_COL]]], axis=1)
    
    print(f"Final Dataset Shape: {df_final.shape}")
    print(f"  - PCA Features: {len(pca_cols)}")
    print(f"  - Categorical Features: {df_cat.shape[1]}")
    print(f"  - Target: 1")
    
    # 6. Save
    df_final.to_csv(OUTPUT_DATA_FILE, index=False)
    print(f"\nHybrid dataset saved to: {OUTPUT_DATA_FILE}")
    print("You can now use this file for your Machine Learning models.")

if __name__ == "__main__":
    run_hybrid_pca()
