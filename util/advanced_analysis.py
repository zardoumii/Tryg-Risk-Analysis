import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
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
    
    plt.show()
    return correlation_matrix

def analyze_target(df, target_col='ClaimFrequency', output_dir=None):
    """Enhanced target analysis for insurance claims with visualizations"""
    if output_dir is None:
        output_dir = r"C:\Users\walde\OneDrive - ITU\Documents\Machine learning"
    os.makedirs(output_dir, exist_ok=True)
    if target_col == 'ClaimFrequency' and 'ClaimFrequency' not in df.columns:
        if 'ClaimNb' in df.columns and 'Exposure' in df.columns:
            df = df.copy()
            df['ClaimFrequency'] = df['ClaimNb'] / df['Exposure'].replace(0, np.finfo(float).eps)
        else:
            print(f"Cannot create {target_col}: 'ClaimNb' or 'Exposure' missing")
            return None
    
    if target_col not in df.columns:
        print(f"Target variable '{target_col}' not found")
        return None
    
    target = df[target_col]

    print(f"\n===== {target_col} Summary Statistics =====")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std: {target.std():.4f}")
    print(f"Range: {target.min():.4f} - {target.max():.4f}")
    print(f"Skewness: {target.skew():.4f}, Kurtosis: {target.kurt():.4f}")
    
    # Zero-claim analysis
    zeros = (target == 0).sum()
    print(f"Zero {target_col}: {zeros} ({zeros/len(df)*100:.2f}%)")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original distribution
    sns.histplot(target, bins=50, kde=True, color='steelblue', ax=axes[0])
    axes[0].set_title(f'Distribution of {target_col}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frequency')
    
    # Log-transformed distribution  
    sns.histplot(np.log1p(target), bins=50, kde=True, color='coral', ax=axes[1])
    axes[1].set_title(f'Log-Transformed Distribution of {target_col}')
    axes[1].set_xlabel(f'log1p({target_col})')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_col}_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Risk factor analysis
    risk_factors = ['VehAge', 'DrivAge', 'BonusMalus', 'VehPower']
    
    for factor in risk_factors:
        if factor in df.columns:

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
    
    categorical_cols = ['Region', 'VehBrand', 'VehGas', 'Area']
    for col in categorical_cols:
        if col in df.columns:
            mean_target = df.groupby(col)[target_col].mean().sort_values(ascending=False)
            print(f"\nTop 5 {col} by average {target_col}:")
            for category, mean_val in mean_target.head(5).items():
                print(f"   {category}: {mean_val:.4f}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        print(f"\nTop correlations with {target_col}:")
        for feature, corr_val in correlations.head(10).items():
            if feature != target_col:
                print(f"   {feature}: {corr_val:.4f}")
    
    return True

def scale_features(df, exclude_cols=None, include_categorical=False):
    if exclude_cols is None:
        exclude_cols = ['IDpol', 'ClaimNb', 'ClaimFrequency']
    
    df_processed = df.copy()

    if include_categorical:
        df_processed = encode_categorical_features(df_processed, exclude_cols)
    
    features_to_scale = [col for col in df_processed.columns if col not in exclude_cols]
    
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
    
    for col in exclude_cols:
        if col in df_processed.columns:
            df_scaled[col] = df_processed[col]
    
    return {
        'scaled_data': df_scaled,
        'scaler': scaler,
        'feature_names': features_to_scale,
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

class PCAAnalysis:
    def __init__(self, scaled_data, variance_explained=0.95):
        self.scaled_data = scaled_data
        self.variance_explained = variance_explained
        self.X_scaled = None
        self.pca = None
        self.X_pca = None
        self.n_components = None
        self.feature_names = None
    
    def prepare_data(self):
        exclude_from_pca = ['IDpol', 'ClaimNb', 'ClaimFrequency']
        feature_cols = [col for col in self.scaled_data.columns 
                       if col not in exclude_from_pca and 
                       self.scaled_data[col].dtype in ['float64', 'int64']]
        
        if not feature_cols:
            raise ValueError("No valid feature columns found for PCA")

        features_only = self.scaled_data[feature_cols]
        self.X_scaled = features_only.values
        self.feature_names = feature_cols
        
        print(f"Using pre-scaled FEATURES ONLY: {self.X_scaled.shape}")
        print(f"Features for PCA: {len(self.feature_names)}")
        print(f"Excluded from PCA: {exclude_from_pca}")
        return self

    def fit(self):
        pca_full = PCA()
        pca_full.fit(self.X_scaled)
        
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_components = np.argmax(cumsum_var >= self.variance_explained) + 1
        
        print(f"\nPCA Results:")
        print(f"  Target variance: {self.variance_explained*100:.0f}%")
        print(f"  Components needed: {self.n_components}")
        print(f"  Dimensionality reduction: {(1 - self.n_components/self.X_scaled.shape[1])*100:.1f}%")
        
        self.pca = PCA(n_components=self.n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"\nExplained variance by component:")
        for i, var in enumerate(self.pca.explained_variance_ratio_[:min(5, self.n_components)]):
            print(f"  PC{i+1}: {var*100:.2f}%")
        return self
    
    def plot_variance(self, output_dir="visualizations"):
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                    self.pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Variance Explained by Each Component')
        axes[0].grid(alpha=0.3)
        
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2)
        axes[1].axhline(y=self.variance_explained, color='r', linestyle='--', 
                       label=f'Target ({self.variance_explained*100:.0f}%)')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'pca_variance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved variance plot to: {output_path}")
        plt.close()
        return self
    
    # Optional: 2D PCA scatter 
    # Keeping the original 2D plotting code here as commented lines so it can
    # be re-enabled manually if you want the PC1-PC2 scatter later.
    # def plot_2d(self, output_dir="visualizations"):
    #     os.makedirs(output_dir, exist_ok=True)
    #     
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     scatter = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
    #                         alpha=0.6, c=range(len(self.X_pca)), cmap='viridis', s=50)
    #     
    #     ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.2f}%)')
    #     ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.2f}%)')
    #     ax.set_title('Data in PC1-PC2 Space')
    #     ax.grid(alpha=0.3)
    #     plt.colorbar(scatter, ax=ax, label='Sample Index')
    #     plt.tight_layout()
    #     
    #     output_path = os.path.join(output_dir, 'pca_2d.png')
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"Saved 2D plot to: {output_path}")
    #     plt.close()
    #     return self
    
    # The active 2D plotting method has been intentionally removed from
    # automatic execution. The commented block above contains the original
    # implementation; re-enable it there if you want to run the PC1-PC2 scatter.
    
    def plot_loadings(self, output_dir="visualizations", top_n=10):
        os.makedirs(output_dir, exist_ok=True)
        
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.feature_names
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, ax in enumerate(axes):
            col = f'PC{i+1}'
            loadings_sorted = loadings[col].abs().sort_values(ascending=False)[:top_n]
            colors = ['steelblue' if loadings[col][name] > 0 else 'coral' 
                     for name in loadings_sorted.index]
            
            ax.barh(range(len(loadings_sorted)), loadings[col][loadings_sorted.index], color=colors)
            ax.set_yticks(range(len(loadings_sorted)))
            ax.set_yticklabels(loadings_sorted.index, fontsize=9)
            ax.set_xlabel('Loading Value')
            ax.set_title(f'Feature Loadings for {col}')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'pca_loadings.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved loadings plot to: {output_path}")
        plt.close()
        return self
    
    def get_pca_dataframe(self):
        return pd.DataFrame(
            self.X_pca,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
    
    def save_pca_data(self, output_path=None):
        if output_path is None:
            output_path = r"C:\Users\walde\OneDrive - ITU\Documents\Machine learning\claims_pca.csv"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pca_df = self.get_pca_dataframe()
        pca_df.to_csv(output_path, index=False)
        print(f"Saved PCA data to: {output_path}")
        return self
    
    def run_pipeline(self, output_dir=None):
        if output_dir is None:
            output_dir = r"C:\Users\walde\OneDrive - ITU\Documents\Machine learning"
        
        self.prepare_data() 
        self.fit()
        self.plot_variance(output_dir)
        self.plot_loadings(output_dir)
        self.save_pca_data()
        
        return self


def run_complete_pipeline(file_path=None, run_pca=True, correlation_threshold=0.3):
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    try:
        from basic_analysis import run_complete_analysis as basic_analysis
    except ImportError:
        return None
   
    if file_path is None:
        file_path = r"c:\Users\walde\OneDrive - ITU\Documents\Machine learning\Project\claims_train.csv"

    try:
        df = basic_analysis(file_path)
    except Exception as e:
        return None

    analyze_target(df)

    pearson_corr = analyze_correlations(df, threshold=correlation_threshold, method='pearson')
    spearman_corr = analyze_correlations(df, threshold=correlation_threshold, method='spearman')

    scaling_results = scale_features(df, include_categorical=True)
  
    pca_results = None
    if run_pca and scaling_results:
        pca_analyzer = PCAAnalysis(scaled_data=scaling_results['scaled_data'], variance_explained=0.95)
        pca_analyzer.prepare_data()
        pca_analyzer.fit()
        pca_analyzer.plot_variance()
        pca_analyzer.plot_loadings()
        pca_analyzer.save_pca_data()
        
        pca_results = {
            'n_components': pca_analyzer.n_components,
            'explained_variance_ratio': pca_analyzer.pca.explained_variance_ratio_,
            'pca_data': pca_analyzer.get_pca_dataframe()
        }
    
    if scaling_results:
        print(f"Scaled features: {scaling_results['final_shape']}")
        print(f"Features processed: {len(scaling_results['feature_names'])}")
    
    if pca_results:
        print(f"PCA components: {pca_results['n_components']}")
        reduction = (1 - pca_results['n_components']/len(scaling_results['feature_names']))*100
        print(f"Dimensionality reduction: {reduction:.1f}%")
    
    return {
        'original_data': df,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'scaling_results': scaling_results,
        'pca_results': pca_results,
        'file_path': file_path
    }

if __name__ == "__main__":
    complete_results = run_complete_pipeline()
    pass