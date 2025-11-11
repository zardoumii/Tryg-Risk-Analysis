import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


class PCAAnalysis:
    def __init__(self, data_path, variance_explained=0.95, use_log_transform=False):
        self.data_path = data_path
        self.variance_explained = variance_explained
        self.use_log_transform = use_log_transform
        self.df = None
        self.X = None
        self.X_scaled = None
        self.scaler = None
        self.pca = None
        self.X_pca = None
        self.n_components = None
        self.feature_names = None
        

        #when combining to advanced analysis-delete loading part since we have t already 
    def load_and_prepare(self):
        self.df = pd.read_csv(self.data_path)
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            print(f"One-hot encoding categorical columns: {categorical_cols}")
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in ['ClaimFrequency', 'ClaimNb', 'Exposure']:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        self.X = self.df[numeric_cols].copy()
        self.feature_names = numeric_cols
        
        if self.use_log_transform:
            print("Applying log transformation...")
            self.X = np.log1p(self.X)
        
        print(f"Data prepared: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        return self
    
    def standardize(self):
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        print(f"Data standardized (mean=0, std=1)")
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
    
    def plot_2d(self, output_dir="visualizations"):
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                            alpha=0.6, c=range(len(self.X_pca)), cmap='viridis', s=50)
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.2f}%)')
        ax.set_title('Data in PC1-PC2 Space')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'pca_2d.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D plot to: {output_path}")
        plt.close()
        return self
    
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
            output_path = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_pca.csv"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pca_df = self.get_pca_dataframe()
        pca_df.to_csv(output_path, index=False)
        print(f"Saved PCA data to: {output_path}")
        return self
    
    def run_pipeline(self, output_dir=None):
        if output_dir is None:
            output_dir = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\visualizations"
        
        print("=" * 60)
        print("PCA Analysis Pipeline")
        print("=" * 60)
        
        self.load_and_prepare()
        self.standardize()
        self.fit()
        self.plot_variance(output_dir)
        self.plot_2d(output_dir)
        self.plot_loadings(output_dir)
        self.save_pca_data()
        
        print("\n" + "=" * 60)
        print("PCA Complete!")
        print("=" * 60)
        return self


if __name__ == "__main__":
    pca = PCAAnalysis(
        data_path=r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_train.csv",
        variance_explained=0.95,
        use_log_transform=True
    )
    pca.run_pipeline()
