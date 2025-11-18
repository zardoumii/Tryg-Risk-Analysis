import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

def analyze_correlations(df, threshold=0.3, method='pearson', save_plot=True):
    """Analyze correlations with clean visualization"""
    numerical_data = df.select_dtypes(include=[np.number])
    
    if method == 'spearman':
        correlation_matrix = numerical_data.corr(method='spearman')
        title = 'Spearman Correlation Matrix of Numerical Features'
    else:
        correlation_matrix = numerical_data.corr()
        title = 'Pearson Correlation Matrix of Numerical Features'
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, 
                mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                fmt='.3f', annot_kws={'size': 9})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    if save_plot:
        filename = f'correlation_matrix_{method}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.close()  # Close instead of show for non-interactive
    return correlation_matrix

def analyze_target(df, target_col='ClaimFrequency', output_dir=None):
    """Target analysis for insurance claims"""
    if output_dir is None:
        output_dir = r"C:\Users\walde\OneDrive - ITU\Documents\Machine learning"
    os.makedirs(output_dir, exist_ok=True)

    df = _create_target_variable(df, target_col)
    if target_col not in df.columns:
        print(f"Target variable '{target_col}' not found")
        return False, df
    
    target = df[target_col]

    _print_target_stats(target, target_col)
    _plot_distributions(target, target_col, output_dir)
    _analyze_risk_factors(df, target_col)
    _analyze_categorical_factors(df, target_col)
    _analyze_correlations_with_target(df, target_col)
    
    return True, df

def _create_target_variable(df, target_col):
    """Create ClaimFrequency if it doesn't exist"""
    if target_col == 'ClaimFrequency' and 'ClaimFrequency' not in df.columns:
        if 'ClaimNb' in df.columns and 'Exposure' in df.columns:
            df = df.copy()
            df['ClaimFrequency'] = df['ClaimNb'] / df['Exposure'].replace(0, np.finfo(float).eps)
            print("Created ClaimFrequency = ClaimNb / Exposure")
        else:
            print(f"Cannot create {target_col}: 'ClaimNb' or 'Exposure' missing")
    return df

def _print_target_stats(target, target_col):
    """Print basic statistics"""
    zeros = (target == 0).sum()
    stats = {
        'Mean': target.mean(),
        'Std': target.std(), 
        'Range': f"{target.min():.4f} - {target.max():.4f}",
        'Skewness': target.skew(),
        'Kurtosis': target.kurt(),
        f'Zero {target_col}': f"{zeros} ({zeros/len(target)*100:.2f}%)"
    }
    
    print(f"\n {target_col} Summary Statistics ")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def _plot_distributions(target, target_col, output_dir):
    """Create distribution plots"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(target, bins=50, kde=True, color='steelblue', ax=axes[0])
    axes[0].set_title(f'Distribution of {target_col}')
    
    sns.histplot(np.log1p(target), bins=50, kde=True, color='coral', ax=axes[1])
    axes[1].set_title(f'Log-Transformed Distribution of {target_col}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_col}_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show

def _analyze_risk_factors(df, target_col):
    """Analyze numerical risk factors"""
    risk_factors = ['VehAge', 'DrivAge', 'BonusMalus', 'VehPower']
    
    for factor in risk_factors:
        if factor not in df.columns:
            continue
            
        try:
            quartiles = pd.qcut(df[factor], 4, duplicates='drop')
            quartile_means = df.groupby(quartiles)[target_col].mean()
            print(f"\n{factor} quartile analysis:")
            for quartile, mean_val in quartile_means.items():
                print(f"   {quartile}: {mean_val:.4f}")
        except ValueError:
            median_val = df[factor].median()
            high_risk = df[df[factor] >= median_val][target_col].mean()
            low_risk = df[df[factor] < median_val][target_col].mean()
            print(f"\n{factor}: High={high_risk:.4f}, Low={low_risk:.4f}")

def _analyze_categorical_factors(df, target_col):
    """Analyze categorical factors"""
    categorical_cols = ['Region', 'VehBrand', 'VehGas', 'Area']
    
    for col in categorical_cols:
        if col in df.columns:
            mean_target = df.groupby(col)[target_col].mean().sort_values(ascending=False)
            print(f"\nTop 5 {col} by average {target_col}:")
            for category, mean_val in mean_target.head(5).items():
                print(f"   {category}: {mean_val:.4f}")

def _analyze_correlations_with_target(df, target_col):
    """Show correlations with target"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        print(f"\nTop correlations with {target_col}:")
        for feature, corr_val in correlations.head(10).items():
            if feature != target_col:
                print(f"   {feature}: {corr_val:.4f}")

def scale_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['IDpol', 'ClaimNb']  
    
    df_processed = df.copy()
    
    # Only select numeric columns for scaling
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    features_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    if not features_to_scale:
        return None

    features_data = df_processed[features_to_scale].copy()

    features_transformed = log_transformation(features_data)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_transformed)

    df_scaled = pd.DataFrame(
        scaled_features, 
        columns=features_to_scale, 
        index=df_processed.index
    )
    
    # Add back excluded columns AND categorical columns
    for col in exclude_cols:
        if col in df_processed.columns:
            df_scaled[col] = df_processed[col]
    
    # Add back categorical columns (strings)
    categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in categorical_cols:
        df_scaled[col] = df_processed[col]
    
    return {
        'scaled_data': df_scaled,
        'scaler': scaler,
        'feature_names': features_to_scale,
        'target_scaled': 'ClaimFrequency' in features_to_scale,
        'original_shape': df.shape,
        'final_shape': df_scaled.shape
    }

def encode_categorical_features(df, exclude_cols):
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    if not categorical_cols:
        return df
    
    total_categories = 0
    for col in categorical_cols:
        n_categories = df[col].nunique()
        total_categories += n_categories
        onehot_df = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, onehot_df], axis=1)
    
    df = df.drop(columns=categorical_cols)
    
    print(f"Encoded {len(categorical_cols)} categorical → {total_categories} binary features")
    return df

def log_transformation(features_data):

    features_transformed = features_data.copy()
    
    for col in features_transformed.columns:
        if features_transformed[col].min() <= 0:
            features_transformed[col] = np.log1p(features_transformed[col])
        else:
            features_transformed[col] = np.log(features_transformed[col])
    
    return features_transformed


def run_complete_pipeline(file_path=None, correlation_threshold=0.3):
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    try:
        from basic_analysis import run_complete_analysis as basic_analysis
    except ImportError:
        return None
   
    if file_path is None:
        file_path = r"c:\Users\walde\OneDrive - ITU\Documents\Machine learning\Project\claims_train_cleaned.csv"

    try:
        if not os.path.exists(file_path) and file_path.endswith('_cleaned.csv'):
            original_file = file_path.replace('_cleaned.csv', '.csv')
            print(f"Cleaned data not found. Running basic analysis on {original_file}...")
            df_cleaned = basic_analysis(original_file) 
        else:
            print(f"Loading cleaned data from: {file_path}")
            df_cleaned = pd.read_csv(file_path)
            print(f"Loaded cleaned dataset: {df_cleaned.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    target_success, df_with_target = analyze_target(df_cleaned)
    if not target_success:
        df_with_target = df_cleaned

    # Skip correlation plots to avoid hanging - just get the matrices
    print("Calculating correlations...")
    pearson_corr = df_with_target.select_dtypes(include=[np.number]).corr()
    spearman_corr = df_with_target.select_dtypes(include=[np.number]).corr(method='spearman')
    print("Correlations calculated successfully")

    # Feature scaling
    print("Starting feature scaling...")
    scaling_results = scale_features(df_with_target)
    
    processed_file = None
    if scaling_results:
        print(f"Scaled features: {scaling_results['final_shape']}")
        print(f"Features processed: {len(scaling_results['feature_names'])}")

        processed_file = file_path.replace('.csv', '_fully_processed.csv')
        scaling_results['scaled_data'].to_csv(processed_file, index=False)
        print(f"The dataset saved to: {processed_file}")
    
    return {
        'cleaned_data': df_cleaned,
        'data_with_target': df_with_target,
        'fully_processed_data': scaling_results['scaled_data'] if scaling_results else None,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'scaling_results': scaling_results,
        'scaler': scaling_results['scaler'] if scaling_results else None,
        'file_path': file_path,
        'processed_file_path': processed_file
    }

if __name__ == "__main__":
    complete_results = run_complete_pipeline()
    print("Analysis completed successfully!")