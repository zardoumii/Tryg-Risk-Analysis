import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_data_pca.csv"
RESULTS_DIR = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\results"
OUTPUT_FILE = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_data_clustered.csv"

PCA_COLS = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return None
    df = pd.read_csv(INPUT_FILE)
    print(f"Data loaded: {df.shape}")
    return df

def determine_optimal_k(df, max_k=8, sample_size=10000):
    """
    Runs K-Means for k=2 to max_k using ONLY Silhouette Score.
    """
    print("\n--- Determining Optimal K (Silhouette Analysis) ---")
    
    X = df[PCA_COLS]
    
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        print(f"Processing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate Silhouette Score on a sample (for speed)
        # If dataset < 10k rows, you can remove sample_size
        score = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        silhouette_scores.append(score)
        print(f"  k={k}: Silhouette={score:.4f}")

    # --- Plotting (Single Plot) ---
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2)
    
    # Visual styling
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score (Higher is better)')
    plt.title('Silhouette Analysis for Optimal k')
    plt.grid(True)
    
    # Highlight the max point
    best_idx = np.argmax(silhouette_scores)
    best_k = K_range[best_idx]
    best_score = silhouette_scores[best_idx]
    plt.axvline(x=best_k, color='green', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    plt.legend()
    
    plot_path = os.path.join(RESULTS_DIR, 'clustering_silhouette.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()
    
    print(f"\nSuggested Optimal k: {best_k} (Score: {best_score:.4f})")
    return best_k

def apply_clustering(df, k):
    print(f"\n--- Applying K-Means with k={k} ---")
    X = df[PCA_COLS]
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    print("Cluster distribution:")
    print(df['Cluster'].value_counts().sort_index())
    
    if 'ClaimFrequency' in df.columns:
        print("\nMean ClaimFrequency per Cluster:")
        print(df.groupby('Cluster')['ClaimFrequency'].mean())

    return df

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df = load_data()
    
    if df is not None:
        # 1. Find Optimal K using only Silhouette
        suggested_k = determine_optimal_k(df, max_k=8)
        
        # NOTE: If the script suggests k=2 but you want more granularity for insurance 
        # (e.g., Low, Medium, High risk), you can manually set k=3 here.
        # k = 3
        
        # 2. Apply Clustering
        df_clustered = apply_clustering(df, k=suggested_k)
        
        # 3. Save
        df_clustered.to_csv(OUTPUT_FILE, index=False)
        print(f"\nClustered dataset saved to: {OUTPUT_FILE}")