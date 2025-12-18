import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime


class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names = None
        self.categorical_features = None
        self.feature_types = None
        
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, 
                     value=None, categories_left=None, is_categorical=False):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.categories_left = categories_left
            self.is_categorical = is_categorical
            
    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)
    
    def _variance_reduction(self, parent, left, right):
        n = len(parent)
        n_left = len(left)
        n_right = len(right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_variance = self._mse(parent)
        left_variance = self._mse(left)
        right_variance = self._mse(right)
        
        child_variance = (n_left / n) * left_variance + (n_right / n) * right_variance
        
        return parent_variance - child_variance
    
    def _best_split_numerical(self, X_column, y):
        best_gain = -float('inf')
        best_threshold = None
    
        n_splits = 10
        if len(X_column) < 100:
            thresholds = np.unique(X_column)
        else:
            percentiles = np.linspace(0, 100, n_splits + 2)[1:-1]
            thresholds = np.unique(np.percentile(X_column, percentiles))
    
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
        best_gain = -float('inf')
        best_categories_left = None
        
        unique_categories = np.unique(X_column)
        
        if len(unique_categories) <= 1:
            return best_gain, best_categories_left
        
        category_means = {}
        for cat in unique_categories:
            mask = X_column == cat
            if np.sum(mask) > 0:
                category_means[cat] = np.mean(y[mask])
        
        sorted_categories = sorted(category_means.keys(), key=lambda x: category_means[x])
        
        for i in range(1, len(sorted_categories)):
            categories_left = set(sorted_categories[:i])
            
            left_mask = np.isin(X_column, list(categories_left))
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
            
            gain = self._variance_reduction(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_categories_left = categories_left
        
        return best_gain, best_categories_left
    
    def _best_split(self, X, y):
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
            else:
                gain, threshold = self._best_split_numerical(X_column, y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_categories_left = None
                    best_is_categorical = False
        
        return best_feature, best_threshold, best_categories_left, best_is_categorical
    
    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return self.Node(value=np.mean(y))
        
        feature, threshold, categories_left, is_categorical = self._best_split(X, y)
        
        if feature is None:
            return self.Node(value=np.mean(y))
        
        if is_categorical:
            left_mask = np.array([x in categories_left for x in X[:, feature]])
        else:
            left_mask = X[:, feature] <= threshold
        
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self.Node(value=np.mean(y))
        
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
        self.feature_names = feature_names
        self.feature_types = feature_types if feature_types else ['numerical'] * X.shape[1]
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        
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
        return np.array([self._predict_sample(x, self.tree) for x in X])


def load_and_prepare_data():
    import pandas as pd
    
    data_path = Path('data')
    
    train_file = data_path / 'claims_train_cleaned.csv'
    test_file = data_path / 'claims_test_cleaned.csv'
    
    output_lines = []
    output_lines.append("Loading data...")
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    output_lines.append(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples")
    
    y_train = df_train['ClaimNb'] / df_train['Exposure']
    y_test = df_test['ClaimNb'] / df_test['Exposure']
    
    drop_cols = ['ClaimFrequency', 'ClaimNb', 'IDpol']
    
    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    
    feature_names = [c for c in df_train.columns if c not in drop_cols]
    
    X_train = df_train[feature_names].values
    X_test = df_test[feature_names].values
    
    y_train = y_train.values
    y_test = y_test.values
    
    feature_types = ['categorical' if name in categorical_cols else 'numerical' 
                     for name in feature_names]
    
    X_train_list = []
    X_test_list = []
    
    for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
        if ftype == 'numerical':
            X_train_list.append(X_train[:, i].astype(float))
            X_test_list.append(X_test[:, i].astype(float))
        else:
            X_train_list.append(X_train[:, i].astype(str))
            X_test_list.append(X_test[:, i].astype(str))
    
    X_train = np.empty((len(df_train), len(feature_names)), dtype=object)
    X_test = np.empty((len(df_test), len(feature_names)), dtype=object)
    
    for i in range(len(feature_names)):
        X_train[:, i] = X_train_list[i]
        X_test[:, i] = X_test_list[i]
    
    output_lines.append(f"\nFeatures: {feature_names}")
    output_lines.append(f"Categorical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'categorical']}")
    output_lines.append(f"Numerical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'numerical']}")
    output_lines.append(f"\nTarget variable: ClaimFrequency")
    output_lines.append(f"Training target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    output_lines.append(f"Training target mean: {y_train.mean():.4f}")
    output_lines.append(f"Non-zero claims: {np.sum(y_train > 0)} ({100 * np.sum(y_train > 0) / len(y_train):.2f}%)")
    
    return X_train, X_test, y_train, y_test, feature_names, feature_types, output_lines


def evaluate_model(y_true, y_pred, dataset_name=""):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    output_lines = []
    output_lines.append(f"\n{dataset_name} Results:")
    output_lines.append(f"  MSE:  {mse:.6f}")
    output_lines.append(f"  RMSE: {rmse:.6f}")
    output_lines.append(f"  MAE:  {mae:.6f}")
    output_lines.append(f"  R²:   {r2:.6f}")
    
    return mse, rmse, mae, r2, output_lines

def train_and_evaluate_decision_tree():
    output_lines = []
    
    output_lines.append("=" * 70)
    output_lines.append("Decision Tree Regressor - From Scratch Implementation")
    output_lines.append("=" * 70)
    
    X_train, X_test, y_train, y_test, feature_names, feature_types, load_lines = load_and_prepare_data()
    output_lines.extend(load_lines)
    
    output_lines.append("\n" + "=" * 70)
    output_lines.append("Training Decision Tree Regressor...")
    output_lines.append("=" * 70)
    
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=200,
        min_samples_leaf=100
    )
    
    output_lines.append(f"\nModel parameters:")
    output_lines.append(f"  max_depth: {model.max_depth}")
    output_lines.append(f"  min_samples_split: {model.min_samples_split}")
    output_lines.append(f"  min_samples_leaf: {model.min_samples_leaf}")
    output_lines.append(f"\nTraining on full dataset: {len(X_train)} samples")
    
    output_lines.append("\nFitting model...")
    model.fit(X_train, y_train, feature_names=feature_names, feature_types=feature_types)
    output_lines.append("Model trained successfully!")
    
    output_lines.append("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Evaluation Metrics")
    output_lines.append("=" * 50)
    
    mse_train, rmse_train, mae_train, r2_train, train_lines = evaluate_model(y_train, y_pred_train, "Training Set")
    output_lines.extend(train_lines)
    
    mse_test, rmse_test, mae_test, r2_test, test_lines = evaluate_model(y_test, y_pred_test, "Test Set")
    output_lines.extend(test_lines)
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Sample Predictions (first 10 test samples)")
    output_lines.append("=" * 50)
    output_lines.append(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
    output_lines.append("-" * 30)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_pred_test[i])
        output_lines.append(f"{y_test[i]:<10.2f} {y_pred_test[i]:<10.4f} {error:<10.4f}")
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Training Complete!")
    output_lines.append("=" * 50)
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Detailed Results")
    output_lines.append("=" * 50)
    output_lines.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_lines.append("\nDataset Information:")
    output_lines.append(f"  Training samples: {len(y_train)}")
    output_lines.append(f"  Test samples: {len(y_test)}")
    output_lines.append(f"  Features: {len(feature_names)}")
    output_lines.append(f"  Feature names: {feature_names}")
    output_lines.append(f"  Categorical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'categorical']}")
    output_lines.append(f"  Numerical features: {[name for name, ftype in zip(feature_names, feature_types) if ftype == 'numerical']}")
    
    output_lines.append("\nModel Hyperparameters:")
    output_lines.append(f"  max_depth: {model.max_depth}")
    output_lines.append(f"  min_samples_split: {model.min_samples_split}")
    output_lines.append(f"  min_samples_leaf: {model.min_samples_leaf}")
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Sample Predictions (first 20 test samples):")
    output_lines.append("=" * 50)
    output_lines.append(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    output_lines.append("-" * 48)
    for i in range(min(20, len(y_test))):
        error = abs(y_test[i] - y_pred_test[i])
        output_lines.append(f"{i:<8} {y_test[i]:<12.4f} {y_pred_test[i]:<12.4f} {error:<12.4f}")
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Summary Statistics:")
    output_lines.append("=" * 50)
    output_lines.append(f"Training predictions - Mean: {y_pred_train.mean():.4f}, Std: {y_pred_train.std():.4f}")
    output_lines.append(f"Test predictions - Mean: {y_pred_test.mean():.4f}, Std: {y_pred_test.std():.4f}")
    output_lines.append(f"Training actuals - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    output_lines.append(f"Test actuals - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / 'decision_tree_results.txt'
    
    with open(results_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Training report saved to: {results_file}")
    
    return model, y_test, y_pred_test


if __name__ == "__main__":
    model, y_test, y_pred_test = train_and_evaluate_decision_tree()
    print(f"Results saved successfully!")