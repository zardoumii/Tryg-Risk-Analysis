import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import sys

NUMERICAL_COLS = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']

EXCLUDE_COLS = ['IDpol', 'ClaimNb', 'ClaimFrequency']


def run_hybrid_pca(input_final_file, input_cleaned_file, output_data_file, dataset_name=""):
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(input_final_file):
        return
    df_final = pd.read_csv(input_final_file)
    
    if not os.path.exists(input_cleaned_file):
        return
    df_cleaned = pd.read_csv(input_cleaned_file)

    if len(df_final) != len(df_cleaned):
        min_len = min(len(df_final), len(df_cleaned))
        df_final = df_final.iloc[:min_len]
        df_cleaned = df_cleaned.iloc[:min_len]

    target = df_cleaned['ClaimNb'] / df_cleaned['Exposure'].clip(lower=0.001)
    target.name = 'ClaimFrequency'
    
    if target.min() < 0:
        target = target.abs()

    available_num_cols = [col for col in NUMERICAL_COLS if col in df_final.columns]
    X_num = df_final[available_num_cols]

    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_num)
    
    explained_var = pca.explained_variance_ratio_ * 100
    total_var = explained_var.sum()
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Scree Plot{" - " + dataset_name if dataset_name else ""} (Numerical Features Only)\nTotal Explained: {total_var:.2f}%')
    plt.grid(True)
    plot_name = f'hybrid_pca_scree_{dataset_name.lower().replace(" ", "_")}.png' if dataset_name else 'hybrid_pca_scree.png'
    plt.savefig(os.path.join(results_dir, plot_name))
    plt.close()
    
    pca_cols = [f'PC{i+1}' for i in range(5)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    all_cols = df_final.columns.tolist()
    cols_to_drop = available_num_cols + [c for c in EXCLUDE_COLS if c in df_final.columns]
    
    cat_ohe_cols = [c for c in all_cols if c not in cols_to_drop]
    df_cat_ohe = df_final[cat_ohe_cols].reset_index(drop=True)
    
    df_pca = df_pca.reset_index(drop=True)
    target = target.reset_index(drop=True)
    
    df_hybrid = pd.concat([df_pca, df_cat_ohe, target], axis=1)
    
    df_hybrid.to_csv(output_data_file, index=False)
    
    return df_hybrid


def rpca():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    train_final = os.path.join(data_dir, 'claims_train_cleaned_final.csv')
    train_cleaned = os.path.join(data_dir, 'claims_train_cleaned.csv')
    train_output = os.path.join(data_dir, 'claims_train_pca.csv')
    run_hybrid_pca(train_final, train_cleaned, train_output, dataset_name="Training")
    
    test_final = os.path.join(data_dir, 'claims_test_cleaned_final.csv')
    test_cleaned = os.path.join(data_dir, 'claims_test_cleaned.csv')
    test_output = os.path.join(data_dir, 'claims_test_pca.csv')
    run_hybrid_pca(test_final, test_cleaned, test_output, dataset_name="Test")
    print("PCA transformation complete for both datasets.")