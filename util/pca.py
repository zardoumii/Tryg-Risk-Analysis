import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Prevent plots from appearing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========== CONFIG ===========
DATA_FILE = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\data\claims_train.csv"
SAVE_DIR = r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Tryg-Risk-Analysis\results"
os.makedirs(SAVE_DIR, exist_ok=True)
print("Plots will be saved to:", os.path.abspath(SAVE_DIR))

# =========== LOAD DATA ===========
df = pd.read_csv(DATA_FILE)
print("Original shape:", df.shape)

# =========== FEATURE SELECTION ===========
categorical = ["Area", "BonusMalus", "VehBrand", "VehGas", "Region"]
numerical = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "Density"]

# One-hot encode categoricals
df_cat = pd.get_dummies(df[categorical], drop_first=True)

# Combine numeric + encoded categorical
X = pd.concat([df[numerical], df_cat], axis=1)
print("Final feature matrix:", X.shape)

# =========== SCALING ===========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========== PCA ===========
pca = PCA(n_components=5)
PCs = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_ * 100
print("\nExplained variance per component:")
for i, var in enumerate(explained_var, 1):
    print(f"  PC{i}: {var:.2f}%")
print(f"Total variance captured (5 PCs): {explained_var.sum():.2f}%")

# =========== PLOTS ===========

def plot_scree(pca, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Scree plot saved to: {save_path}")

def plot_pca_scatter(PCs, labels, save_path):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(PCs[:, 0], PCs[:, 1], c=labels, cmap="viridis", alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (Colored by ClaimNb)")
    plt.colorbar(scatter, label="ClaimNb")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"PCA scatter plot saved to: {save_path}")

# =========== SAVE PLOTS ===========
scree_path = os.path.join(SAVE_DIR, "scree_plot.png")
scatter_path = os.path.join(SAVE_DIR, "pca_scatter.png")

plot_scree(pca, save_path=scree_path)
plot_pca_scatter(PCs, labels=df["ClaimNb"], save_path=scatter_path)

# Quick test plot to ensure saving works
test_path = os.path.join(SAVE_DIR, "test_plot.png")
plt.plot([1,2,3],[4,5,6])
plt.savefig(test_path)
plt.close()
print(f"Test plot saved to: {test_path}")

print("\nAll plots saved successfully!")
