import pandas as pd         
import numpy as np          
import matplotlib.pyplot as plt 

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
        print(missing_vals[missing_vals > 0])
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Fixed {col} missing values with median: {median_val}")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_series = df[col].mode()
                mode_val = mode_series[0] if len(mode_series) > 0 else 'Unknown'
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
        print(f"{col}: min={min_val}, max={max_val}")
        
        if col == 'Exposure' and (min_val <= 0 or max_val > 1):
            print(f"  {col} values outside expected range [0,1]")
            df[col] = df[col].clip(0.001, 1.0)
            print(f"  Fixed: Capped {col} to [0.001, 1.0]")
            
        if col == 'ClaimNb' and min_val < 0:
            print(f"  {col} has negative values")
            df[col] = df[col].clip(lower=0)
            print(f"  Fixed: Set negative {col} to 0")
            
        if col in ['VehAge', 'DrivAge'] and min_val < 0:
            print(f"  {col} has negative values")
            df[col] = df[col].abs()
            print(f"  Fixed: Converted negative {col} to absolute values")

def analyze_features(df):
    """Analyze categorical and numerical features"""

    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']

    for col in categorical_cols:
        if col in df.columns:  
            print(f"\n{col} - Unique values: {df[col].nunique()}")
            print(df[col].value_counts().head())
        else:
            print(f"\n{col} - Column not found in dataset (skipped)")
    
    numerical_cols = ['ClaimNb', 'Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=50, alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

    for i in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def detect_outliers(df):
    """Detect outliers"""
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
 
        if col == 'Exposure':
            exposure_issues = df[(df[col] <= 0) | (df[col] > 1)]
            print(f"  Exposure outside [0,1]: {len(exposure_issues)} records")
    
    plt.tight_layout()
    plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_complete_analysis(file_path):
    df = load_data(file_path)
    
    assess_data(df)  
    analyze_features(df)
    detect_outliers(df)  
    
    output_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    file_path = r"c:\Users\walde\OneDrive - ITU\Documents\Machine learning\Project\claims_train.csv"
    
    print(f"Looking for data file: {file_path}")
    
    try:
        dataset = run_complete_analysis(file_path)
        print(f"Dataset loaded with shape: {dataset.shape}")
        
    except FileNotFoundError:
        print(f" Error: File '{file_path}' not found!")
 
    except Exception as e:
        print(f"Error: {e}")