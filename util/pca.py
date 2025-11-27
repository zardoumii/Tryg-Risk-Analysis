import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import sys

#final code from advanced analysis for scaled features
INPUT_FINAL = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_train_cleaned_final.csv"
#code from basic analysis for unscaled features
INPUT_CLEANED = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_train_cleaned.csv"

SAVE_DIR = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\results"
OUTPUT_DATA_FILE = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_data_pca.csv"

#numerical columns
NUMERICAL_COLS = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']

# columns to exclude from features
EXCLUDE_COLS = ['IDpol', 'ClaimNb', 'ClaimFrequency']


def run_hybrid_pca():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FINAL):
        print(f"Error: Final processed file not found at {INPUT_FINAL}")
        return
    print(f"Loading processed data (for features): {INPUT_FINAL}")
    df_final = pd.read_csv(INPUT_FINAL)
    
    if not os.path.exists(INPUT_CLEANED):
        print(f"Error: Cleaned file not found at {INPUT_CLEANED}")
        return
    print(f"Loading cleaned data (for real target values): {INPUT_CLEANED}")
    df_cleaned = pd.read_csv(INPUT_CLEANED)

    # --- CRITICAL STEP: Align Indices ---
    # Ensure we grab the targets for the exact same rows that exist in df_final
    # (In case rows were dropped during outlier removal in advanced analysis)
    # We assume the row order was preserved or indices match.
    if len(df_final) != len(df_cleaned):
        print(f"Warning: Row counts differ (Final: {len(df_final)}, Cleaned: {len(df_cleaned)}).")
        print("Aligning target based on index (assuming no index reset happened violently).")
        # Truncate to match if necessary, though ideally they match
        min_len = min(len(df_final), len(df_cleaned))
        df_final = df_final.iloc[:min_len]
        df_cleaned = df_cleaned.iloc[:min_len]

    # 3. Re-Create the Target Variable (Unscaled)
    # We calculate it fresh from the cleaned file to ensure it is POSITIVE
    print("Recreating unscaled target variable...")
    target = df_cleaned['ClaimNb'] / df_cleaned['Exposure'].clip(lower=0.001)
    target.name = 'ClaimFrequency'
    
    # Verify Target
    print(f"Target stats: Min={target.min():.4f}, Max={target.max():.4f}, Mean={target.mean():.4f}")
    if target.min() < 0:
        print("WARNING: Target still has negative values. Checking calculation...")
        target = target.abs() # Force positive if something went wrong

    # 4. Extract Numerical Data for PCA
    available_num_cols = [col for col in NUMERICAL_COLS if col in df_final.columns]
    print(f"\nProcessing Numerical Columns (from Final): {available_num_cols}")
    X_num = df_final[available_num_cols]

    # 5. Run PCA on Numerical Only
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_num)
    
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
    
    # 6. Create PCA DataFrame
    pca_cols = [f'PC{i+1}' for i in range(5)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    # 7. Extract Categorical Data (One-Hot Encoded)
    # Identify columns to keep: All columns in Final that are NOT Num, NOT ID, NOT Target
    all_cols = df_final.columns.tolist()
    # We exclude numericals we just PCA'd, and potential ID/Target columns if they exist in Final
    cols_to_drop = available_num_cols + [c for c in EXCLUDE_COLS if c in df_final.columns]
    
    cat_ohe_cols = [c for c in all_cols if c not in cols_to_drop]
    print(f"\nPreserving {len(cat_ohe_cols)} One-Hot Encoded Categorical columns.")
    df_cat_ohe = df_final[cat_ohe_cols].reset_index(drop=True)
    
    # 8. Combine: PCA Features + Existing OHE Categoricals + REAL Target
    print("Combining datasets...")
    df_pca = df_pca.reset_index(drop=True)
    target = target.reset_index(drop=True)
    
    df_hybrid = pd.concat([df_pca, df_cat_ohe, target], axis=1)
    
    print(f"Final Dataset Shape: {df_hybrid.shape}")
    print(f"  - PCA Features: {len(pca_cols)}")
    print(f"  - Categorical Features (OHE): {len(cat_ohe_cols)}")
    
    # 9. Save
    df_hybrid.to_csv(OUTPUT_DATA_FILE, index=False)
    print(f"\nHybrid dataset saved to: {OUTPUT_DATA_FILE}")

if __name__ == "__main__":
    run_hybrid_pca()