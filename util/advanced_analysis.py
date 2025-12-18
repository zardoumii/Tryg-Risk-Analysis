import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

def analyze_correlations(df, threshold=0.3, method='pearson', save_plot=True):
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
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(results_dir, f'correlation_matrix_{method}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    return correlation_matrix

def analyze_target(df, target_col='ClaimFrequency', output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
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
    
    sns.histplot(target, bins=20, kde=True, color='steelblue', ax=axes[0])
    axes[0].set_title('Distribution of ClaimFrequency')
    
    sns.histplot(np.log1p(target), bins=15, kde=True, color='coral', ax=axes[1])
    axes[1].set_title(f'Log-Transformed Distribution of {target_col}')
    
    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'{target_col}_distributions.png'), dpi=300, bbox_inches='tight')

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
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    features_to_transform = [col for col in numeric_cols if col not in exclude_cols]
    
    if features_to_transform:
        features_data = df_processed[features_to_transform].copy()
        features_transformed = log_transformation(features_data)
        for col in features_to_transform:
            df_processed[col] = features_transformed[col]
        print(f"Applied log transformation to: {features_to_transform}")
    
    df_processed = encode_categorical_features(df_processed, exclude_cols)
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    features_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    if not features_to_scale:
        return None
    
    print(f"Scaling {len(features_to_scale)} features")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed[features_to_scale])
    df_scaled = df_processed.copy()  

    for i, col in enumerate(features_to_scale):
        df_scaled[col] = scaled_features[:, i]
    
    onehot_count = len([col for col in df_scaled.columns if '_' in col and 
                       any(col.startswith(prefix) for prefix in ['Area_', 'VehBrand_', 'VehGas_', 'Region_'])])
    print(f"  - {onehot_count} one-hot encoded features")
    print(f"  - {len([col for col in exclude_cols if col in df_scaled.columns])} excluded features")
    
    return {
        'scaled_data': df_scaled,
        'scaler': scaler,
        'feature_names': features_to_scale,
        'target_scaled': 'ClaimFrequency' in features_to_scale,
        'original_shape': df.shape,
        'final_shape': df_scaled.shape
    }

def encode_categorical_features(df, exclude_cols):
    """One-hot encode categorical features"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    if not categorical_cols:
        print("No categorical features found to encode")
        return df
    
    print(f"Encoding categorical features: {categorical_cols}")
    
    df_encoded = df.copy()
    total_categories = 0
    
    for col in categorical_cols:
        n_categories = df[col].nunique()
        total_categories += n_categories
        print(f"  {col}: {n_categories} categories -> {n_categories} binary features")
        
        onehot_df = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df_encoded = pd.concat([df_encoded, onehot_df], axis=1)
    
    df_encoded = df_encoded.drop(columns=categorical_cols)
    
    print(f"Encoded {len(categorical_cols)} categorical features -> {total_categories} binary features")
    print(f"Dataset shape: {df.shape} -> {df_encoded.shape}")
    
    return df_encoded

def log_transformation(features_data):

    features_transformed = features_data.copy()
    
    for col in features_transformed.columns:
        if features_transformed[col].min() <= 0:
            features_transformed[col] = np.log1p(features_transformed[col])
        else:
            features_transformed[col] = np.log(features_transformed[col])
    
    return features_transformed

def plot_scaled_feature_distributions_and_outliers(df, feature_names, results_dir):

    n_features = len(feature_names)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel()
    for i, col in enumerate(feature_names):
        axes[i].hist(df[col], bins=40, alpha=0.7, color='steelblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    dist_path = os.path.join(results_dir, 'scaled_feature_distributions.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scaled feature distributions to: {dist_path}")

    target_vars = ['ClaimFrequency', 'ClaimNb']
    outlier_features = [col for col in feature_names if col not in target_vars]
    n_outlier_features = len(outlier_features)
    n_rows_out = int(np.ceil(n_outlier_features / n_cols))
    fig, axes = plt.subplots(n_rows_out, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel()
    for i, col in enumerate(outlier_features):
        axes[i].boxplot(df[col], vert=True)
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel(col)
    for i in range(n_outlier_features, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    box_path = os.path.join(results_dir, 'scaled_feature_outliers.png')
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved scaled feature outlier boxplots to: {box_path}")

def run_complete_pipeline(file_path=None, correlation_threshold=0.3):
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    try:
        from basic_analysis import run_complete_analysis as basic_analysis
    except ImportError:
        print("Warning: basic_analysis module not found.")
        return None 
    
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'claims_train_cleaned.csv')

    try:
        if not os.path.exists(file_path) and file_path.endswith('_cleaned.csv'):
            original_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'claims_train.csv')
            print(f"Cleaned data not found. Running basic analysis on {original_file}")
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

    pearson_corr = analyze_correlations(df_with_target, threshold=correlation_threshold, method='pearson')
    spearman_corr = analyze_correlations(df_with_target, threshold=correlation_threshold, method='spearman')

    scaling_results = scale_features(df_with_target)
    
    processed_file = None
    if scaling_results:
        print(f"Scaled features: {scaling_results['final_shape']}")
        print(f"Features processed: {len(scaling_results['feature_names'])}")

        processed_file = file_path.replace('.csv', '_final.csv')
        scaling_results['scaled_data'].to_csv(processed_file, index=False)
        print(f"The dataset saved to: {processed_file}")
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plot_scaled_feature_distributions_and_outliers(scaling_results['scaled_data'], scaling_results['feature_names'], results_dir)
    
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

def advanced_analysis():
    import sys
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Process test data
    test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'claims_test_cleaned.csv')
    output_txt_test = os.path.join(results_dir, 'advanced_analysis_test_output.txt')
    
    print("Running complete pipeline on claims_test_cleaned.csv...")
    with open(output_txt_test, 'w') as f:
        sys.stdout = f
        print("="*70)
        print("TEST DATA ANALYSIS")
        print("="*70)
        complete_results_test = run_complete_pipeline(file_path=test_file)
        print("Test data analysis completed successfully!")
    sys.stdout = sys.__stdout__
    print(f"Test analysis completed! Output saved to {output_txt_test}")
    
    print("\nBoth analyses completed successfully!")
# Process training data
    train_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'claims_train_cleaned.csv')
    output_txt_train = os.path.join(results_dir, 'advanced_analysis_train_output.txt')
    
    print("Running complete pipeline on claims_train_cleaned.csv...")
    with open(output_txt_train, 'w') as f:
        sys.stdout = f
        print("="*70)
        print("TRAINING DATA ANALYSIS")
        print("="*70)
        complete_results_train = run_complete_pipeline(file_path=train_file)
        print("Training data analysis completed successfully!")
    sys.stdout = sys.__stdout__
    print(f"Training analysis completed! Output saved to {output_txt_train}")