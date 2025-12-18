"""
Decision Tree Regressor - sklearn Implementation
Reference implementation for validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_and_prepare_data():
    """Load and prepare the claims data - creates ClaimFrequency from ClaimNb/Exposure"""
    data_path = Path('data')
    
    # Load CSV files
    train_file = data_path / 'claims_train_cleaned.csv'
    test_file = data_path / 'claims_test_cleaned.csv'
    
    output_lines = []
    output_lines.append("Loading data...")
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    output_lines.append(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples")
    
    # 1. Create the Target (y) - WE KEEP THE ANSWER HERE
    y_train = (df_train['ClaimNb'] / df_train['Exposure']).values
    y_test = (df_test['ClaimNb'] / df_test['Exposure']).values
    
    # 2. Create the Features (X) - WE REMOVE THE ANSWER HERE
    # We must drop 'ClaimNb' because it reveals the answer
    drop_cols = ['ClaimFrequency', 'ClaimNb', 'IDpol']
    
    # Define categorical features
    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    
    # This list of names is used for BOTH X_train and X_test
    feature_names = [c for c in df_train.columns if c not in drop_cols]
    
    # Separate categorical and numerical features
    categorical_features = [f for f in feature_names if f in categorical_cols]
    numerical_features = [f for f in feature_names if f not in categorical_cols]
    
    # Get data
    X_train = df_train[feature_names].copy()
    X_test = df_test[feature_names].copy()
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    # Convert to numpy arrays
    X_train = X_train.values.astype(float)
    X_test = X_test.values.astype(float)
    
    output_lines.append(f"\nFeatures: {feature_names}")
    output_lines.append(f"Categorical features (encoded): {categorical_features}")
    output_lines.append(f"Numerical features: {numerical_features}")
    output_lines.append(f"\nTarget variable: ClaimFrequency (ClaimNb / Exposure)")
    output_lines.append(f"Training target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    output_lines.append(f"Training target mean: {y_train.mean():.4f}")
    output_lines.append(f"Non-zero claims: {np.sum(y_train > 0)} ({100 * np.sum(y_train > 0) / len(y_train):.2f}%)")
    
    return X_train, X_test, y_train, y_test, feature_names, output_lines


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate evaluation metrics and return as strings"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    output_lines = []
    output_lines.append(f"\n{dataset_name} Results:")
    output_lines.append(f"  MSE:  {mse:.6f}")
    output_lines.append(f"  RMSE: {rmse:.6f}")
    output_lines.append(f"  MAE:  {mae:.6f}")
    output_lines.append(f"  R²:   {r2:.6f}")
    
    return mse, rmse, mae, r2, output_lines


def train_and_evaluate_sklearn_tree():
    """Train and evaluate sklearn Decision Tree model - can be called from other scripts"""
    output_lines = []
    
    output_lines.append("=" * 50)
    output_lines.append("Decision Tree Regressor - sklearn Implementation")
    output_lines.append("=" * 50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, load_lines = load_and_prepare_data()
    output_lines.extend(load_lines)
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Training sklearn Decision Tree Regressor...")
    output_lines.append("=" * 50)
    
    # Initialize and train the model
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=200,
        min_samples_leaf=100,
        random_state=3326
    )
    
    output_lines.append(f"\nModel parameters:")
    output_lines.append(f"  max_depth: {model.max_depth}")
    output_lines.append(f"  min_samples_split: {model.min_samples_split}")
    output_lines.append(f"  min_samples_leaf: {model.min_samples_leaf}")
    output_lines.append(f"  random_state: {model.random_state}")
    output_lines.append(f"\nTraining on full dataset: {len(X_train)} samples")
    
    # Train the model
    output_lines.append("\nFitting model...")
    model.fit(X_train, y_train)
    output_lines.append("Model trained successfully!")
    
    # Make predictions
    output_lines.append("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Evaluation Metrics")
    output_lines.append("=" * 50)
    
    mse_train, rmse_train, mae_train, r2_train, train_lines = evaluate_model(y_train, y_pred_train, "Training Set")
    output_lines.extend(train_lines)
    
    mse_test, rmse_test, mae_test, r2_test, test_lines = evaluate_model(y_test, y_pred_test, "Test Set")
    output_lines.extend(test_lines)
    
    # Show some sample predictions
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Sample Predictions (first 10 test samples)")
    output_lines.append("=" * 50)
    output_lines.append(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
    output_lines.append("-" * 50)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_pred_test[i])
        output_lines.append(f"{y_test[i]:<10.2f} {y_pred_test[i]:<10.4f} {error:<10.4f}")
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Training Complete!")
    output_lines.append("=" * 50)
    
    # Additional details for file
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Detailed Results")
    output_lines.append("=" * 50)

    
    output_lines.append("\nDataset Information:")
    output_lines.append(f"  Training samples: {len(y_train)}")
    output_lines.append(f"  Test samples: {len(y_test)}")
    output_lines.append(f"  Features: {len(feature_names)}")
    output_lines.append(f"  Feature names: {feature_names}")
    
    output_lines.append("\nModel Hyperparameters:")
    output_lines.append(f"  max_depth: {model.max_depth}")
    output_lines.append(f"  min_samples_split: {model.min_samples_split}")
    output_lines.append(f"  min_samples_leaf: {model.min_samples_leaf}")
    output_lines.append(f"  random_state: {model.random_state}")
    
    # Feature importance
    output_lines.append("\nTop 10 Feature Importances:")
    feature_importance = sorted(zip(feature_names, model.feature_importances_), 
                                key=lambda x: x[1], reverse=True)
    for i, (name, importance) in enumerate(feature_importance[:10], 1):
        output_lines.append(f"  {i}. {name}: {importance:.4f}")
    
    output_lines.append("\n" + "=" * 50)
    output_lines.append("Sample Predictions (first 20 test samples):")
    output_lines.append("=" * 50)
    output_lines.append(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    output_lines.append("-" * 50)
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
    
    # Save results to file
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / 'sklearn_decision_tree_results.txt'
    
    with open(results_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Training report saved to: {results_file}")
    
    return model, y_test, y_pred_test


if __name__ == "__main__":
    model, y_test, y_pred_test = train_and_evaluate_sklearn_tree()

