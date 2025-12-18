import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_data(file_path):
    """Load and explore the dataset"""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Total records: {len(df):,}")
    print(df.info())
    print(df.head())
    print(df.describe())
    return df

def assess_data(df):
    """Check and fix missing values, duplicates, and data quality issues"""
    
    missing_vals = df.isnull().sum()
    if missing_vals.sum() > 0:
        print("Missing values per column:")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Fixed {col} missing values with median: {median_val}")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"  Fixed {col} missing values with mode: {mode_val}")
    else:
        print("No missing values")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Duplicate rows: {duplicates} → Removed")
    else:
        print(f"Duplicate rows: {duplicates}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        
        if col == 'Exposure' and (min_val <= 0 or max_val > 1):
            df[col] = df[col].clip(0.001, 1.0)
            print(f"  Fixed: Capped {col} to [0.001, 1.0]")
            
        if col == 'ClaimNb' and min_val < 0:
            df[col] = df[col].clip(lower=0)
            print(f"  Fixed: Set negative {col} to 0")
            
        if col in ['VehAge', 'DrivAge'] and min_val < 0:
            df[col] = df[col].abs()
            print(f"  Fixed: Converted negative {col} to absolute values")

def analyze_features(df, results_dir):
    """Analyze categorical and numerical features and save plot"""
    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    for col in categorical_cols:
        if col in df.columns:  
            print(f"\n{col} - Unique values: {df[col].nunique()}")
            print(df[col].value_counts().head())
    
    numerical_cols = ['ClaimNb', 'Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=50, alpha=0.7, color='steelblue', edgecolor='white')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

    for i in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'feature_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

def detect_outliers(df, results_dir):
    """Detect outliers and save boxplots"""
    numerical_cols = ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols):
        axes[i].boxplot(df[col])
        axes[i].set_title(f'Box Plot: {col}')
        axes[i].set_ylabel(col)

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'outlier_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_complete_analysis(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    log_file_path = os.path.join(results_dir, 'basic_analysis_results.txt')
    
    with open(log_file_path, 'w') as f:
        sys.stdout = f 
        
        print("BASIC DATA ANALYSIS RESULTS")
        print("="*50)
        
        df = load_data(file_path)
        assess_data(df)  
        analyze_features(df, results_dir)
        detect_outliers(df, results_dir) 
        
        output_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(output_path, index=False)
        print(f"\nCleaned dataset saved to: {output_path}")
        print(f"Final shape: {df.shape}")
        
        sys.stdout = sys.__stdout__   
        
    print(f"Complete results log saved to: {log_file_path}")
    print(f"Plots saved in: {results_dir}")
    return df

if __name__ == "__main__":
    file_path = r"c:\Users\walde\OneDrive - ITU\Documents\Machine learning\data\claims_train.csv"
    
    try:
        dataset = run_complete_analysis(file_path)
        print(f"Dataset successfully cleaned. Final shape: {dataset.shape}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
    except Exception as e:
        print(f"An error occurred: {e}")