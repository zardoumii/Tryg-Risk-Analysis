"""
Decision Tree Regressor Implementation from Scratch
Using only NumPy and standard Python libraries
Handles both numerical and categorical features
"""

import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime


class DecisionTreeRegressor:
    """Decision Tree Regressor with support for categorical variables"""
    
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10):
        """
        Initialize Decision Tree Regressor
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required in a leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names = None
        self.categorical_features = None
        self.feature_types = None
        
    class Node:
        """Tree node structure"""
        def __init__(self, feature=None, threshold=None, left=None, right=None, 
                     value=None, categories_left=None, is_categorical=False):
            self.feature = feature          # Feature index to split on
            self.threshold = threshold      # Threshold for numerical features
            self.left = left               # Left child node
            self.right = right             # Right child node
            self.value = value             # Prediction value for leaf nodes
            self.categories_left = categories_left  # Categories that go left
            self.is_categorical = is_categorical    # Whether this is a categorical split
            
    def _mse(self, y):
        """Calculate Mean Squared Error"""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)
    
    def _variance_reduction(self, parent, left, right):
        """Calculate variance reduction (information gain)"""
        n = len(parent)
        n_left = len(left)
        n_right = len(right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_variance = self._mse(parent)
        left_variance = self._mse(left)
        right_variance = self._mse(right)
        
        # Weighted variance of children
        child_variance = (n_left / n) * left_variance + (n_right / n) * right_variance
        
        return parent_variance - child_variance
    
    def _best_split_numerical(self, X_column, y):
        """Find best split for numerical feature"""
        best_gain = -float('inf')
        best_threshold = None
        
        # Get unique values as potential split points
        unique_values = np.unique(X_column)
        
        # Try splits between consecutive unique values
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            
            left_mask = X_column <= threshold
            right_mask = ~left_mask
            
            # Check minimum samples constraint
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
            
            # Calculate variance reduction
            gain = self._variance_reduction(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_gain, best_threshold
    
    def _best_split_categorical(self, X_column, y):
        """Find best split for categorical feature"""
        best_gain = -float('inf')
        best_categories_left = None
        
        unique_categories = np.unique(X_column)
        
        if len(unique_categories) <= 1:
            return best_gain, best_categories_left
        
        # For categorical features, we try different binary partitions
        # For efficiency, we use a greedy approach: sort categories by mean target value
        category_means = {}
        for cat in unique_categories:
            mask = X_column == cat
            if np.sum(mask) > 0:
                category_means[cat] = np.mean(y[mask])
        
        # Sort categories by their mean target value
        sorted_categories = sorted(category_means.keys(), key=lambda x: category_means[x])
        
        # Try different split points
        for i in range(1, len(sorted_categories)):
            categories_left = set(sorted_categories[:i])
            
            left_mask = np.array([x in categories_left for x in X_column])
            right_mask = ~left_mask
            
            # Check minimum samples constraint
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
            
            # Calculate variance reduction
            gain = self._variance_reduction(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_categories_left = categories_left
        
        return best_gain, best_categories_left
    
    def _best_split(self, X, y):
        """Find the best split across all features"""
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_categories_left = None
        best_is_categorical = False
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            
            if self.feature_types[feature_idx] == 'categorical':
                gain, categories_left = self._best_split_categorical(X_column, y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_categories_left = categories_left
                    best_threshold = None
                    best_is_categorical = True
            else:  # numerical
                gain, threshold = self._best_split_numerical(X_column, y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_categories_left = None
                    best_is_categorical = False
        
        return best_feature, best_threshold, best_categories_left, best_is_categorical
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return self.Node(value=np.mean(y))
        
        # Find best split
        feature, threshold, categories_left, is_categorical = self._best_split(X, y)
        
        # If no valid split found, create leaf node
        if feature is None:
            return self.Node(value=np.mean(y))
        
        # Split the data
        if is_categorical:
            left_mask = np.array([x in categories_left for x in X[:, feature]])
        else:
            left_mask = X[:, feature] <= threshold
        
        right_mask = ~left_mask
        
        # Check if split produces valid children
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self.Node(value=np.mean(y))
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return self.Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
            categories_left=categories_left,
            is_categorical=is_categorical
        )
    
    def fit(self, X, y, feature_names=None, feature_types=None):
        """
        Train the decision tree
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix (n_samples, n_features)
        y : numpy array
            Target values (n_samples,)
        feature_names : list, optional
            Names of features
        feature_types : list, optional
            Types of features ('categorical' or 'numerical')
        """
        self.feature_names = feature_names
        self.feature_types = feature_types if feature_types else ['numerical'] * X.shape[1]
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """Predict single sample by traversing the tree"""
        # If leaf node, return the value
        if node.value is not None:
            return node.value
        
        # Navigate based on split type
        if node.is_categorical:
            if x[node.feature] in node.categories_left:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
        else:
            if x[node.feature] <= node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Predict on multiple samples
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix (n_samples, n_features)
            
        Returns:
        --------
        predictions : numpy array
            Predicted values (n_samples,)
        """
        return np.array([self._predict_sample(x, self.tree) for x in X])


def load_and_prepare_data():
    """Load and prepare the claims data"""
    data_path = Path('data')
    
    # Load CSV files
    train_file = data_path / 'claims_train.csv'
    test_file = data_path / 'claims_test.csv'
    
    print("Loading data...")
    train_data = []
    with open(train_file, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            train_data.append(line.strip().split(','))
    
    test_data = []
    with open(test_file, 'r') as f:
        f.readline()  # Skip header
        for line in f:
            test_data.append(line.strip().split(','))
    
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    # Convert to numpy arrays
    train_array = np.array(train_data)
    test_array = np.array(test_data)
    
    # Define column information
    feature_names = header[1:]  # Skip IDpol
    claim_nb_idx = feature_names.index('ClaimNb')
    exposure_idx = feature_names.index('Exposure')
    
    # Define categorical and numerical features
    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    
    # Prepare data
    X_train = train_array[:, 1:]  # Skip IDpol
    X_test = test_array[:, 1:]    # Skip IDpol
    
    # Extract target as claim frequency (ClaimNb / Exposure)
    claim_nb_train = X_train[:, claim_nb_idx].astype(float)
    exposure_train = X_train[:, exposure_idx].astype(float)
    claim_nb_test = X_test[:, claim_nb_idx].astype(float)
    exposure_test = X_test[:, exposure_idx].astype(float)
    
    # Calculate claim frequency, avoiding division by zero
    y_train = np.where(exposure_train > 0, claim_nb_train / exposure_train, 0)
    y_test = np.where(exposure_test > 0, claim_nb_test / exposure_test, 0)
    
    # Remove ClaimNb from features (but keep Exposure as it's a predictor)
    feature_indices = [i for i in range(len(feature_names)) if i != claim_nb_idx]
    X_train = X_train[:, feature_indices]
    X_test = X_test[:, feature_indices]
    feature_names = [feature_names[i] for i in feature_indices]
    
    # Create feature types list
    feature_types = ['categorical' if name in categorical_cols else 'numerical' 
                     for name in feature_names]
    
    # Process features
    # For numerical features, convert to float
    # For categorical features, keep as strings but in object array
    X_train_list = []
    X_test_list = []
    
    for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
        if ftype == 'numerical':
            X_train_list.append(X_train[:, i].astype(float))
            X_test_list.append(X_test[:, i].astype(float))
        else:  # categorical
            X_train_list.append(X_train[:, i])
            X_test_list.append(X_test[:, i])
    
    # Stack with object dtype to preserve mixed types
    X_train = np.empty((len(train_data), len(feature_names)), dtype=object)
    X_test = np.empty((len(test_data), len(feature_names)), dtype=object)
    
    for i in range(len(feature_names)):
        X_train[:, i] = X_train_list[i]
        X_test[:, i] = X_test_list[i]
    
    print(f"\nFeatures: {feature_names}")
    print(f"Categorical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'categorical']}")
    print(f"Numerical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'numerical']}")
    print(f"\nTarget variable: Claim Frequency (ClaimNb / Exposure)")
    print(f"Training target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"Training target mean: {y_train.mean():.4f}")
    print(f"Non-zero claims: {np.sum(y_train > 0)} ({100 * np.sum(y_train > 0) / len(y_train):.2f}%)")
    
    return X_train, X_test, y_train, y_test, feature_names, feature_types


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate and print evaluation metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\n{dataset_name} Results:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    
    return mse, rmse, mae, r2


if __name__ == "__main__":
    print("=" * 70)
    print("Decision Tree Regressor - From Scratch Implementation")
    print("=" * 70)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, feature_types = load_and_prepare_data()
    
    print("\n" + "=" * 70)
    print("Training Decision Tree Regressor...")
    print("=" * 70)
    
    # Initialize and train the model
    model = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=50,
        min_samples_leaf=20
    )
    
    print(f"\nModel parameters:")
    print(f"  max_depth: {model.max_depth}")
    print(f"  min_samples_split: {model.min_samples_split}")
    print(f"  min_samples_leaf: {model.min_samples_leaf}")
    
    # Train the model
    print("\nFitting model...")
    model.fit(X_train, y_train, feature_names=feature_names, feature_types=feature_types)
    print("Model trained successfully!")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation Metrics")
    print("=" * 70)
    
    mse_train, rmse_train, mae_train, r2_train = evaluate_model(y_train, y_pred_train, "Training Set")
    mse_test, rmse_test, mae_test, r2_test = evaluate_model(y_test, y_pred_test, "Test Set")
    
    # Show some sample predictions
    print("\n" + "=" * 70)
    print("Sample Predictions (first 10 test samples)")
    print("=" * 70)
    print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 30)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_pred_test[i])
        print(f"{y_test[i]:<10.2f} {y_pred_test[i]:<10.4f} {error:<10.4f}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # Save results to file
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'model1_results_{timestamp}.txt'
    
    print(f"\nSaving results to: {results_file}")
    
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Decision Tree Regressor - Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Training samples: {len(y_train)}\n")
        f.write(f"  Test samples: {len(y_test)}\n")
        f.write(f"  Features: {len(feature_names)}\n")
        f.write(f"  Feature names: {feature_names}\n")
        f.write(f"  Categorical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'categorical']}\n")
        f.write(f"  Numerical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'numerical']}\n\n")
        
        f.write("Target Variable: Claim Frequency (ClaimNb / Exposure)\n")
        f.write(f"  Training target range: [{y_train.min():.4f}, {y_train.max():.4f}]\n")
        f.write(f"  Training target mean: {y_train.mean():.4f}\n")
        f.write(f"  Non-zero claims: {np.sum(y_train > 0)} ({100 * np.sum(y_train > 0) / len(y_train):.2f}%)\n\n")
        
        f.write("Model Hyperparameters:\n")
        f.write(f"  max_depth: {model.max_depth}\n")
        f.write(f"  min_samples_split: {model.min_samples_split}\n")
        f.write(f"  min_samples_leaf: {model.min_samples_leaf}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Training Set Results:\n")
        f.write(f"  MSE:  {mse_train:.6f}\n")
        f.write(f"  RMSE: {rmse_train:.6f}\n")
        f.write(f"  MAE:  {mae_train:.6f}\n")
        f.write(f"  R²:   {r2_train:.6f}\n\n")
        
        f.write("Test Set Results:\n")
        f.write(f"  MSE:  {mse_test:.6f}\n")
        f.write(f"  RMSE: {rmse_test:.6f}\n")
        f.write(f"  MAE:  {mae_test:.6f}\n")
        f.write(f"  R²:   {r2_test:.6f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Sample Predictions (first 20 test samples):\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}\n")
        f.write("-" * 48 + "\n")
        for i in range(min(20, len(y_test))):
            error = abs(y_test[i] - y_pred_test[i])
            f.write(f"{i:<8} {y_test[i]:<12.4f} {y_pred_test[i]:<12.4f} {error:<12.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Summary Statistics:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Training predictions - Mean: {y_pred_train.mean():.4f}, Std: {y_pred_train.std():.4f}\n")
        f.write(f"Test predictions - Mean: {y_pred_test.mean():.4f}, Std: {y_pred_test.std():.4f}\n")
        f.write(f"Training actuals - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}\n")
        f.write(f"Test actuals - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}\n")
    
    print(f"Results saved successfully!")
