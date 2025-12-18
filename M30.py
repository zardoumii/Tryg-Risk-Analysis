"""
Final Model: Robust Gradient Boosting (MAE Loss)
Fixed to prevent gradient explosion from outliers
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

def train_final_champion_model_robust():
    print("=" * 80)
    print("Training FINAL CHAMPION: Robust Gradient Boosting (MAE Loss)")
    print("=" * 80)

    # 1. Load Data
    print("\n1. Loading data...")
    train_file = 'data/claims_data_clustered.csv'
    test_file = 'data/claims_data_test_clustered.csv'

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    target_col = 'ClaimFrequency'
    exclude_cols = [target_col]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    # SAFETY CHECK: Clip training target outliers to stabilize training
    # Anyone with > 20 claims is capped at 20 for training purposes.
    # This prevents the "732" outlier from breaking the model.
    y_train_clipped = np.clip(y_train, 0, 20)
    print(f"   Note: Clipped training target max to 20 (True Max: {y_train.max()}) to prevent explosion.")

    # 2. Define the Model
    # CHANGED: loss='absolute_error' (Robust) instead of 'poisson' (Unstable)
    print("\n2. Initializing HistGradientBoostingRegressor...")
    
    model = HistGradientBoostingRegressor(
        loss='absolute_error',   # <--- ROBUST LOSS (Minimizes MAE, ignores outliers)
        learning_rate=0.1,       # Standard learning rate
        max_iter=300,            # Moderate trees
        max_leaf_nodes=31,       
        l2_regularization=0.5,   
        early_stopping=True,     
        validation_fraction=0.1, 
        random_state=3326,
        verbose=1
    )

    # 3. Train
    print("\n3. Training...")
    model.fit(X_train, y_train_clipped)

    # 4. Predict
    print("\n4. Predicting...")
    y_pred = model.predict(X_test)
    
    # Enforce non-negativity
    y_pred = np.maximum(0, y_pred) 

    # 5. Evaluate
    print("\n5. Evaluation:")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("-" * 40)
    print(f"   R² Score: {r2:.6f}")
    print(f"   RMSE:     {rmse:.6f}")
    print(f"   MAE:      {mae:.6f}")
    print("-" * 40)

    # 6. Save
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    results_df = pd.DataFrame({'True': y_test, 'Pred_RobustGBM': y_pred})
    results_df.to_csv('results/robust_gbm_predictions.csv', index=False)
    
    with open('models/final_gbm_robust.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    return model

if __name__ == "__main__":
    train_final_champion_model_robust()