import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

def train_and_evaluate_gbm():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    output_file = 'results/gbm_report.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Gradient Boosting (MSE Loss)\n")
        f.write("=" * 80 + "\n")

        f.write("\n1. Loading data...\n")
        train_file = 'data/claims_train_clustered.csv'
        test_file = 'data/claims_test_clustered.csv'

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        target_col = 'ClaimFrequency'
        exclude_cols = [target_col]
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]

        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[target_col].values
        

        
        model = HistGradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.1,
            max_iter=300,
            max_leaf_nodes=31,       
            l2_regularization=1.0,   
            early_stopping=True,     
            validation_fraction=0.1, 
            random_state=3326,
            verbose=0
        )

        f.write("\n3. Training...\n")
        model.fit(X_train, y_train)

        f.write("\n4. Predicting...\n")
        y_pred = model.predict(X_test)
        
        y_pred = np.maximum(0, y_pred)

        f.write("\n5. Evaluation:\n")
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        f.write("-" * 40 + "\n")
        f.write(f"   R² Score: {r2:.6f}\n")
        f.write(f"   RMSE:     {rmse:.6f}\n")
        f.write(f"   MAE:      {mae:.6f}\n")
        f.write("-" * 40 + "\n")
        
        results_df = pd.DataFrame({'True': y_test, 'Pred_GBM': y_pred})
        results_df.to_csv('results/gbm_predictions.csv', index=False)
        
        with open('models/final_gbm.pkl', 'wb') as f_model:
            pickle.dump(model, f_model)
        
        f.write(f"\nResults saved to {output_file}\n")
        f.write("Predictions saved to results/gbm_predictions.csv\n")
        f.write("Model saved to models/final_gbm.pkl\n")
        
    return model

if __name__ == "__main__":
    train_and_evaluate_gbm()