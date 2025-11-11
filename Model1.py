import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

class DecisionTreeRegressor:
    """Decision Tree Regressor implemented from scratch"""
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names = None
        self.categorical_features = None
        
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, categories=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.categories = categories  # For categorical splits
            
    def _variance_reduction(self, y, y_left, y_right):
        """Calculate variance reduction"""
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        parent_var = np.var(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        child_var = (n_left / n) * np.var(y_left) + (n_right / n) * np.var(y_right)
        return parent_var - child_var
    
    def _best_split_numerical(self, X_column, y):
        """Find best split for numerical feature"""
        best_gain = -np.inf
        best_threshold = None
        
        thresholds = np.unique(X_column)
        
        for threshold in thresholds:
            left_mask = X_column <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
            
            gain = self._variance_reduction(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
                
        return best_gain, best_threshold
    
    def _best_split_categorical(self, X_column, y):
        """Find best split for categorical feature"""
        best_gain = -np.inf
        best_categories = None
        
        unique_cats = np.unique(X_column)
        
        if len(unique_cats) <= 1:
            return best_gain, best_categories
        
        # Try different subsets of categories (binary split)
        for i in range(1, len(unique_cats)):
            for left_cats in self._get_category_combinations(unique_cats, i):
                left_mask = np.isin(X_column, left_cats)
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._variance_reduction(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_categories = left_cats
                    
        return best_gain, best_categories
    
    def _get_category_combinations(self, categories, size):
        """Generate combinations of categories (simplified for efficiency)"""
        return list(combinations(categories, size))[:10]  # Limit to 10 combinations
    
    def _best_split(self, X, y):
        """Find best split across all features"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_categories = None
        
        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            
            if feature_idx in self.categorical_features:
                gain, categories = self._best_split_categorical(X_column, y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_categories = categories
                    best_threshold = None
            else:
                gain, threshold = self._best_split_numerical(X_column, y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_categories = None
                    
        return best_feature, best_threshold, best_categories
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) == 0:
            return self.Node(value=np.mean(y))
        
        # Find best split
        feature, threshold, categories = self._best_split(X, y)
        
        if feature is None:
            return self.Node(value=np.mean(y))
        
        # Split data
        if categories is not None:  # Categorical split
            left_mask = np.isin(X[:, feature], categories)
        else:  # Numerical split
            left_mask = X[:, feature] <= threshold
            
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return self.Node(feature=feature, threshold=threshold, categories=categories,
                        left=left_subtree, right=right_subtree)
    
    def fit(self, X, y, categorical_features=None):
        """Train the decision tree"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.categorical_features = set(categorical_features) if categorical_features else set()
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """Predict single sample"""
        if node.value is not None:
            return node.value
        
        if node.categories is not None:  # Categorical split
            if x[node.feature] in node.categories:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
        else:  # Numerical split
            if x[node.feature] <= node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict on multiple samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict_sample(x, self.tree) for x in X])


# Load and prepare data
def load_data():
    """Load training and test data from the data folder"""
    data_path = Path('data')
    
    train_file = data_path / 'claims_train.csv'
    test_file = data_path / 'claims_test.csv'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    return train_df, test_df


if __name__ == "__main__":
    # Load data
    train_df, test_df = load_data()
    
    print(f"Training dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    print(f"\nColumns: {train_df.columns.tolist()}")
    print(f"\nFirst few rows of training data:\n{train_df.head()}")
    
    # Identify categorical and numerical columns
    categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Assume last column is target, or modify as needed
    target_col = train_df.columns[-1]
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Encode categorical variables
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Create a mapping for categorical encoding to ensure consistency
    category_mappings = {}
    for col in categorical_cols:
        if col in feature_cols or col == target_col:
            # Fit on training data
            categories = pd.Categorical(train_encoded[col])
            category_mappings[col] = dict(zip(categories.categories, range(len(categories.categories))))
            
            # Transform both train and test
            train_encoded[col] = train_encoded[col].map(category_mappings[col]).fillna(-1).astype(int)
            test_encoded[col] = test_encoded[col].map(category_mappings[col]).fillna(-1).astype(int)
    
    # Prepare features and target
    X_train = train_encoded[feature_cols]
    y_train = train_encoded[target_col]
    X_test = test_encoded[feature_cols]
    y_test = test_encoded[target_col]
    
    # Get indices of categorical features
    categorical_feature_indices = [i for i, col in enumerate(feature_cols) if col in categorical_cols]
    
    # Train the model
    print("\nTraining Decision Tree Regressor...")
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    model.fit(X_train, y_train, categorical_features=categorical_feature_indices)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    mse_train = np.mean((y_train - y_pred_train) ** 2)
    mse_test = np.mean((y_test - y_pred_test) ** 2)
    r2_train = 1 - (np.sum((y_train - y_pred_train) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
    r2_test = 1 - (np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    print(f"\nResults:")
    print(f"Train MSE: {mse_train:.4f}")
    print(f"Test MSE: {mse_test:.4f}")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")