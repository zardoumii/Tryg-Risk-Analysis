import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# --- CONFIGURATION ---
PCA_COLS = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

def load_data(input_file):
    if not os.path.exists(input_file):
        return None
    df = pd.read_csv(input_file)
    return df

def determine_optimal_k(df, max_k=8, sample_size=10000, results_dir=None, dataset_name=""):
    """
    Runs K-Means for k=2 to max_k using ONLY Silhouette Score.
    """
    X = df[PCA_COLS]
    
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate Silhouette Score on a sample (for speed)
        # If dataset < 10k rows, you can remove sample_size
        score = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        silhouette_scores.append(score)

    # --- Plotting (Single Plot) ---
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2)
    
    # Visual styling
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score (Higher is better)')
    title = f'Silhouette Analysis for Optimal k{" - " + dataset_name if dataset_name else ""}'
    plt.title(title)
    plt.grid(True)
    
    # Highlight the max point
    best_idx = np.argmax(silhouette_scores)
    best_k = K_range[best_idx]
    best_score = silhouette_scores[best_idx]
    plt.axvline(x=best_k, color='green', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    plt.legend()
    
    plot_name = f'clustering_silhouette_{dataset_name.lower().replace(" ", "_")}.png' if dataset_name else 'clustering_silhouette.png'
    plot_path = os.path.join(results_dir, plot_name)
    plt.savefig(plot_path)
    plt.close()
    
    return best_k

def apply_clustering(df, k):
    X = df[PCA_COLS]
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    return df

def cluster_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Process training data
    train_input = os.path.join(data_dir, 'claims_train_pca.csv')
    train_output = os.path.join(data_dir, 'claims_train_clustered.csv')
    
    df_train = load_data(train_input)
    
    if df_train is not None:
        # 1. Find Optimal K using only Silhouette
        suggested_k_train = determine_optimal_k(df_train, max_k=8, results_dir=results_dir, dataset_name="Training")
        
        # 2. Apply Clustering
        df_train_clustered = apply_clustering(df_train, k=suggested_k_train)
        
        # 3. Save
        df_train_clustered.to_csv(train_output, index=False)
    
    # Process test data
    test_input = os.path.join(data_dir, 'claims_test_pca.csv')
    test_output = os.path.join(data_dir, 'claims_test_clustered.csv')
    
    df_test = load_data(test_input)
    
    if df_test is not None:
        # 1. Find Optimal K using only Silhouette
        suggested_k_test = determine_optimal_k(df_test, max_k=8, results_dir=results_dir, dataset_name="Test")
        
        # 2. Apply Clustering
        df_test_clustered = apply_clustering(df_test, k=suggested_k_test)
        
        # 3. Save
        df_test_clustered.to_csv(test_output, index=False)
    
    print("Clustering complete for both datasets.")