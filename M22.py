"""
Feed-Forward Neural Network Regressor - Reference Implementation
Using scikit-learn's MLPRegressor for validation
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


class SKLearnFFNNRegressor:
    """
    Wrapper for scikit-learn's MLPRegressor to match the interface of custom implementation
    """
    
    def __init__(
        self,
        hidden_layers=(64, 32),
        activation='relu',
        learning_rate=0.001,
        batch_size=512,
        max_epochs=100,
        random_seed=3326,
        early_stopping=True,
        validation_fraction=0.2,
        verbose=True
    ):
        """
        Initialize the sklearn FFNN regressor
        
        Parameters:
        -----------
        hidden_layers : tuple
            Sizes of hidden layers
        activation : str
            Activation function ('relu', 'tanh', 'logistic')
        learning_rate : float
            Learning rate for weight updates
        batch_size : int
            Mini-batch size
        max_epochs : int
            Maximum number of training epochs
        random_seed : int
            Random seed for reproducibility
        early_stopping : bool
            Whether to use early stopping
        validation_fraction : float
            Fraction of training data to use for validation
        verbose : bool
            Whether to print training progress
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate_init = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        
        # Create the MLPRegressor
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver='adam',  # Adam optimizer (similar to what custom implementation might use)
            alpha=0.0001,  # L2 regularization
            batch_size=self.batch_size if self.batch_size != 'auto' else 'auto',
            learning_rate='constant',
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_epochs,
            shuffle=True,
            random_state=self.random_seed,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction if self.early_stopping else 0.1,
            n_iter_no_change=10,  # Patience for early stopping
            verbose=self.verbose
        )
        
        self.is_fitted = False
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_val : array-like, optional
            Validation features (not used if early_stopping=True)
        y_val : array-like, optional
            Validation targets (not used if early_stopping=True)
        """
        # Train the model
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        predictions : array
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions (claim frequency can't be negative)
        predictions = np.maximum(0, predictions)
        
        return predictions
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            True targets
            
        Returns:
        --------
        r2 : float
            R² score
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def save(self, filepath):
        """
        Save the model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """
        Load a model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        model : SKLearnFFNNRegressor
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


def train_and_evaluate_sklearn_ffnn():
    """
    Train and evaluate the sklearn FFNN model on insurance claims data
    """
    # Initialize output list to collect all information
    output_lines = []
    
    output_lines.append("=" * 80)
    output_lines.append("Training scikit-learn MLPRegressor on Insurance Claims Data")
    output_lines.append("=" * 80)
    
    # 1. Load the data
    output_lines.append("\n1. Loading data...")
    train_file = 'data/claims_train_clustered.csv'
    test_file = 'data/claims_test_clustered.csv'
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Shuffle training data
    df_train = df_train.sample(frac=1, random_state=3326).reset_index(drop=True)
    output_lines.append("   Training data shuffled for random validation split")
    
    output_lines.append(f"   Training data shape: {df_train.shape}")
    output_lines.append(f"   Test data shape: {df_test.shape}")
    
    # 2. Prepare features and target
    output_lines.append("\n2. Preparing features and target...")
    target_col = 'ClaimFrequency'
    exclude_cols = [target_col, 'ClaimNb', 'IDpol']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    output_lines.append(f"   Number of features: {len(feature_cols)}")
    output_lines.append(f"   Target variable: {target_col}")
    
    # Separate X and y
    X_train = df_train[feature_cols].astype(float).values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].astype(float).values
    y_test = df_test[target_col].values
    
    output_lines.append(f"\n   X_train shape: {X_train.shape}")
    output_lines.append(f"   y_train shape: {y_train.shape}")
    output_lines.append(f"   X_test shape: {X_test.shape}")
    output_lines.append(f"   y_test shape: {y_test.shape}")
    
    # 3. Check target statistics
    output_lines.append("\n3. Target variable statistics:")
    output_lines.append(f"   Train - Mean: {y_train.mean():.6f}, Std: {y_train.std():.6f}")
    output_lines.append(f"   Train - Min: {y_train.min():.6f}, Max: {y_train.max():.6f}")
    output_lines.append(f"   Train - Zeros: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
    output_lines.append(f"   Test - Mean: {y_test.mean():.6f}, Std: {y_test.std():.6f}")
    output_lines.append(f"   Test - Min: {y_test.min():.6f}, Max: {y_test.max():.6f}")
    output_lines.append(f"   Test - Zeros: {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")
    
    # 4. Create and configure model
    output_lines.append("\n4. Creating sklearn neural network model...")
    model = SKLearnFFNNRegressor(
        hidden_layers=(64, 32),
        activation='relu',
        learning_rate=0.001,
        batch_size=512,
        max_epochs=80,
        random_seed=3326,
        early_stopping=True,
        validation_fraction=0.2,
        verbose=False
    )
    
    output_lines.append(f"   Architecture: {model.hidden_layers}")
    output_lines.append(f"   Activation: {model.activation}")
    output_lines.append(f"   Learning rate: {model.learning_rate_init}")
    output_lines.append(f"   Batch size: {model.batch_size}")
    output_lines.append(f"   Max epochs: {model.max_epochs}")
    output_lines.append(f"   Early stopping: {model.early_stopping}")
    
    # 5. Train the model
    output_lines.append("\n5. Training model...")
    output_lines.append("-" * 80)
    model.fit(X_train, y_train)
    
    output_lines.append(f"Training completed!")
    output_lines.append(f"Final training iterations: {model.model.n_iter_}")
    if hasattr(model.model, 'best_loss_') and model.model.best_loss_ is not None:
        output_lines.append(f"Best validation loss: {model.model.best_loss_:.6f}")
    output_lines.append(f"Final loss: {model.model.loss_:.6f}")
    
    # 6. Make predictions
    output_lines.append("\n6. Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 7. Evaluate performance
    output_lines.append("\n7. Model Performance:")
    output_lines.append("-" * 80)
    
    # Training metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    output_lines.append(f"\nTraining Set:")
    output_lines.append(f"   R² Score: {train_r2:.6f}")
    output_lines.append(f"   MSE:      {train_mse:.6f}")
    output_lines.append(f"   RMSE:     {train_rmse:.6f}")
    output_lines.append(f"   MAE:      {train_mae:.6f}")
    
    output_lines.append(f"\nTest Set:")
    output_lines.append(f"   R² Score: {test_r2:.6f}")
    output_lines.append(f"   MSE:      {test_mse:.6f}")
    output_lines.append(f"   RMSE:     {test_rmse:.6f}")
    output_lines.append(f"   MAE:      {test_mae:.6f}")
    
    # 8. Sample predictions
    output_lines.append("\n8. Sample Predictions (first 20 test samples):")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Index':<8} {'True Value':<15} {'Predicted':<15} {'Error':<15}")
    output_lines.append("-" * 80)
    for i in range(min(20, len(y_test))):
        error = abs(y_test[i] - y_test_pred[i])
        output_lines.append(f"{i:<8} {y_test[i]:<15.6f} {y_test_pred[i]:<15.6f} {error:<15.6f}")
    
    # 9. Predictions for samples with claims
    output_lines.append("\n9. Predictions for samples WITH claims (y > 0):")
    output_lines.append("-" * 80)
    claim_indices = np.where(y_test > 0)[0]
    output_lines.append(f"Found {len(claim_indices)} samples with claims in test set")
    if len(claim_indices) > 0:
        output_lines.append(f"\n{'Index':<8} {'True Value':<15} {'Predicted':<15} {'Error':<15}")
        output_lines.append("-" * 80)
        for i in claim_indices[:10]:
            error = abs(y_test[i] - y_test_pred[i])
            output_lines.append(f"{i:<8} {y_test[i]:<15.6f} {y_test_pred[i]:<15.6f} {error:<15.6f}")
    
    # 10. Save model
    output_lines.append("\n10. Saving model...")
    model_path = 'models/sklearn_ffnn_claims_model.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    output_lines.append(f"   Model saved to: {model_path}")
    
    # 11. Save predictions
    output_lines.append("\n11. Saving predictions to CSV...")
    results_df = pd.DataFrame({
        'True_ClaimFrequency': y_test,
        'Predicted_ClaimFrequency': y_test_pred,
        'Absolute_Error': np.abs(y_test - y_test_pred),
        'Squared_Error': (y_test - y_test_pred) ** 2
    })
    
    results_path = 'results/sklearn_predictions.csv'
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(results_path, index=False)
    output_lines.append(f"   Predictions saved to: {results_path}")
    
    # 12. Prediction statistics
    output_lines.append("\n12. Prediction Statistics:")
    output_lines.append("-" * 80)
    output_lines.append(f"   Predictions - Mean: {y_test_pred.mean():.6f}")
    output_lines.append(f"   Predictions - Std: {y_test_pred.std():.6f}")
    output_lines.append(f"   Predictions - Min: {y_test_pred.min():.6f}")
    output_lines.append(f"   Predictions - Max: {y_test_pred.max():.6f}")
    output_lines.append(f"   Predictions < 0: {(y_test_pred < 0).sum()}")
    
    output_lines.append("\n" + "=" * 80)
    output_lines.append("scikit-learn Training and Testing Completed Successfully!")
    output_lines.append("=" * 80)
    
    # Write all output to file
    output_file = 'results/sklearn_report.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Training report saved to: {output_file}")
    
    return model, y_test, y_test_pred


if __name__ == "__main__":
    # Train and evaluate the sklearn model
    model, y_test, y_test_pred = train_and_evaluate_sklearn_ffnn()
