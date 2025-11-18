"""
Decision Tree Regressor - Reference Implementation using scikit-learn
To validate the correctness of the from-scratch implementation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_and_prepare_data():
    """Load and prepare the claims data"""
    data_path = Path('data')
    
    # Load CSV files using pandas
    train_file = data_path / 'claims_train.csv'
    test_file = data_path / 'claims_test.csv'
    
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Calculate claim frequency as target variable
    train_df['ClaimFrequency'] = np.where(
        train_df['Exposure'] > 0,
        train_df['ClaimNb'] / train_df['Exposure'],
        0
    )
    test_df['ClaimFrequency'] = np.where(
        test_df['Exposure'] > 0,
        test_df['ClaimNb'] / test_df['Exposure'],
        0
    )
    
    # Define features (exclude IDpol and ClaimNb, keep Exposure)
    feature_cols = ['Exposure', 'Area', 'VehPower', 'VehAge', 'DrivAge', 
                    'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region']
    categorical_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
    
    # Prepare features
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # Extract target
    y_train = train_df['ClaimFrequency'].values
    y_test = test_df['ClaimFrequency'].values
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        # Handle unseen categories in test set
        X_test[col] = X_test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {[col for col in feature_cols if col not in categorical_cols]}")
    print(f"\nTarget variable: Claim Frequency (ClaimNb / Exposure)")
    print(f"Training target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"Training target mean: {y_train.mean():.4f}")
    print(f"Non-zero claims: {np.sum(y_train > 0)} ({100 * np.sum(y_train > 0) / len(y_train):.2f}%)")
    
    return X_train, X_test, y_train, y_test, feature_cols, categorical_cols


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate and print evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{dataset_name} Results:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    
    return mse, rmse, mae, r2


if __name__ == "__main__":
    print("=" * 70)
    print("Decision Tree Regressor - Reference Implementation (scikit-learn)")
    print("=" * 70)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, categorical_features = load_and_prepare_data()
    
    print("\n" + "=" * 70)
    print("Training Decision Tree Regressor (sklearn)...")
    print("=" * 70)
    
    # Initialize scikit-learn model with similar hyperparameters
    model = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    
    print(f"\nModel parameters:")
    print(f"  max_depth: {model.max_depth}")
    print(f"  min_samples_split: {model.min_samples_split}")
    print(f"  min_samples_leaf: {model.min_samples_leaf}")
    print(f"  random_state: {model.random_state}")
    
    # Train the model
    print("\nFitting model...")
    model.fit(X_train, y_train)
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
    
    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importances")
    print("=" * 70)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
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
    results_file = results_dir / f'model1_sklearn_results_{timestamp}.txt'
    
    print(f"\nSaving results to: {results_file}")
    
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Decision Tree Regressor - Reference Implementation (scikit-learn)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Training samples: {len(y_train)}\n")
        f.write(f"  Test samples: {len(y_test)}\n")
        f.write(f"  Features: {len(feature_names)}\n")
        f.write(f"  Feature names: {feature_names}\n")
        f.write(f"  Categorical features: {categorical_features}\n")
        f.write(f"  Numerical features: {[col for col in feature_names if col not in categorical_features]}\n\n")
        
        f.write("Target Variable: Claim Frequency (ClaimNb / Exposure)\n")
        f.write(f"  Training target range: [{y_train.min():.4f}, {y_train.max():.4f}]\n")
        f.write(f"  Training target mean: {y_train.mean():.4f}\n")
        f.write(f"  Non-zero claims: {np.sum(y_train > 0)} ({100 * np.sum(y_train > 0) / len(y_train):.2f}%)\n\n")
        
        f.write("Model Hyperparameters:\n")
        f.write(f"  max_depth: {model.max_depth}\n")
        f.write(f"  min_samples_split: {model.min_samples_split}\n")
        f.write(f"  min_samples_leaf: {model.min_samples_leaf}\n")
        f.write(f"  random_state: {model.random_state}\n\n")
        
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
        f.write("Feature Importances:\n")
        f.write("=" * 70 + "\n")
        f.write(feature_importance.to_string(index=False))
        f.write("\n\n")
        
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
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Model Tree Structure:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Number of leaves: {model.get_n_leaves()}\n")
        f.write(f"Tree depth: {model.get_depth()}\n")
    
    print(f"Results saved successfully!")
    
