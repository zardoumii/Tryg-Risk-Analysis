"""
Feed-Forward Neural Network Regressor - From Scratch Implementation
Using only NumPy and SciPy
"""

import numpy as np
from typing import List, Tuple, Optional
import pickle


# activation functions and their derivatives

def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation"""
    # Clip to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_activation(x):
    """Tanh activation"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x) ** 2

def linear(x):
    """Linear activation (identity)"""
    return x

def linear_derivative(x):
    """Derivative of linear"""
    return np.ones_like(x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return np.where(x > 0, 1, alpha)


class FeedForwardNeuralNetwork:
    """
    Feed-Forward Neural Network Regressor from scratch.
    
    Parameters:
    -----------
    hidden_layers : List[int]
        List of hidden layer sizes. E.g., [64, 32] creates two hidden layers.
    activation : str
        Activation function for hidden layers: 'relu', 'sigmoid', 'tanh', 'leaky_relu'
    learning_rate : float
        Learning rate for gradient descent
    batch_size : int
        Mini-batch size for training
    epochs : int
        Number of training epochs
    alpha : float
        L2 regularization parameter (default: 0.0001)
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Print training progress
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        activation: str = 'relu',
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        alpha: float = 0.0001,
        random_seed: int = 3326,
        verbose: bool = True
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.alpha = alpha
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Network parameters (initialized during fit)
        self.weights = []
        self.biases = []
        self.layer_sizes = []
        
        # Training history
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Set activation functions
        self._set_activation_functions()
        
        # Set random seed
        np.random.seed(self.random_seed)
    
    def _set_activation_functions(self):
        """Set activation function and its derivative"""
        if self.activation == 'relu':
            self.activation_func = relu
            self.activation_derivative = relu_derivative
        elif self.activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif self.activation == 'tanh':
            self.activation_func = tanh_activation
            self.activation_derivative = tanh_derivative
        elif self.activation == 'leaky_relu':
            self.activation_func = leaky_relu
            self.activation_derivative = leaky_relu_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _initialize_weights(self, input_size: int, output_size: int):
        """
        Initialize network weights using He initialization for ReLU
        or Xavier initialization for sigmoid/tanh
        """
        self.layer_sizes = [input_size] + self.hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for ReLU-like activations
            if self.activation in ['relu', 'leaky_relu']:
                std = np.sqrt(2.0 / self.layer_sizes[i])
            else:  # Xavier for sigmoid/tanh
                std = np.sqrt(1.0 / self.layer_sizes[i])
            
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * std
            b = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network
        
        Returns:
        --------
        activations : List of activation outputs for each layer
        z_values : List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.activation_func(z)
            activations.append(a)
        
        # Output layer (linear activation for regression)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        a_output = z_output  # Linear activation
        activations.append(a_output)
        
        return activations, z_values
    
    def _backward_propagation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        z_values: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation to compute gradients
        
        Returns:
        --------
        weight_gradients : List of weight gradients
        bias_gradients : List of bias gradients
        """
        m = X.shape[0]  # Number of samples
        n_layers = len(self.weights)
        
        # Initialize gradient lists
        weight_gradients = [None] * n_layers
        bias_gradients = [None] * n_layers
        
        # Output layer error (MSE derivative)
        delta = (activations[-1] - y) / m
        
        # Backpropagate through all layers
        for i in range(n_layers - 1, -1, -1):
            # Compute gradients
            weight_gradients[i] = np.dot(activations[i].T, delta)
            
            # Add L2 regularization gradient
            if self.alpha > 0:
                weight_gradients[i] += self.alpha * self.weights[i]
            
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True)
            
            # Propagate error to previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def _update_parameters(
        self,
        weight_gradients: List[np.ndarray],
        bias_gradients: List[np.ndarray]
    ):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error loss with L2 regularization"""
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Add L2 regularization term
        l2_penalty = 0
        if self.alpha > 0:
            for w in self.weights:
                l2_penalty += np.sum(w ** 2)
            l2_penalty = (self.alpha / 2) * l2_penalty
        
        return mse + l2_penalty
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the neural network
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features)
        y_train : np.ndarray
            Training targets, shape (n_samples,) or (n_samples, 1)
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        """
        # Ensure y is 2D
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        
        n_samples, n_features = X_train.shape
        n_outputs = y_train.shape[1]
        
        # Initialize weights
        self._initialize_weights(n_features, n_outputs)
        
        # Training loop
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward propagation
                activations, z_values = self._forward_propagation(X_batch)
                
                # Compute batch loss
                batch_loss = self._compute_loss(y_batch, activations[-1])
                epoch_loss += batch_loss
                
                # Backward propagation
                weight_gradients, bias_gradients = self._backward_propagation(
                    X_batch, y_batch, activations, z_values
                )
                
                # Update parameters
                self._update_parameters(weight_gradients, bias_gradients)
            
            # Average epoch loss
            avg_train_loss = epoch_loss / n_batches
            self.train_loss_history.append(avg_train_loss)
            
            # Validation loss
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = self._compute_loss(y_val, y_val_pred.reshape(y_val.shape))
                self.val_loss_history.append(val_loss)
            
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {avg_train_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {avg_train_loss:.6f}")
        
        if self.verbose:
            print("\nTraining completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : np.ndarray
            Features, shape (n_samples, n_features)
        
        Returns:
        --------
        predictions : np.ndarray
            Predicted values, shape (n_samples,)
        """
        activations, _ = self._forward_propagation(X)
        return activations[-1].flatten()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            True targets
        
        Returns:
        --------
        r2_score : float
            R² coefficient of determination
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters"""
        return {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'alpha': self.alpha,
            'random_seed': self.random_seed
        }
    
    def save(self, filepath: str):
        """Save model to file"""
        model_data = {
            'params': self.get_params(),
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore parameters
        params = model_data['params']
        self.__init__(**params, verbose=False)
        
        # Restore weights and structure
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']
        self.train_loss_history = model_data['train_loss_history']
        self.val_loss_history = model_data['val_loss_history']
        
        print(f"Model loaded from {filepath}")


def train_and_evaluate_custom_model():
    """
    Train and evaluate the custom FFNN model on insurance claims data
    """
    import pandas as pd
    import os
    
    # Initialize output list to collect all information
    output_lines = []
    
    output_lines.append("=" * 80)
    output_lines.append("Training Custom Feed-Forward Neural Network on Insurance Claims Data")
    output_lines.append("=" * 80)
    
    # 1. Load the data
    output_lines.append("\n1. Loading data...")
    train_file = 'data/claims_data_clustered.csv'
    test_file = 'data/claims_data_test_clustered.csv'
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Shuffle training data to ensure random validation split
    df_train = df_train.sample(frac=1, random_state=3326).reset_index(drop=True)
    output_lines.append("   Training data shuffled for random validation split")
    
    output_lines.append(f"   Training data shape: {df_train.shape}")
    output_lines.append(f"   Test data shape: {df_test.shape}")
    
    # 2. Prepare features and target
    output_lines.append("\n2. Preparing features and target...")
    target_col = 'ClaimFrequency'
    exclude_cols = [target_col]
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    output_lines.append(f"   Number of features: {len(feature_cols)}")
    output_lines.append(f"   Target variable: {target_col}")
    
    # Separate X and y (convert boolean columns to numeric)
    X_train_full = df_train[feature_cols].astype(float).values
    y_train_full = df_train[target_col].values
    
    X_test = df_test[feature_cols].astype(float).values
    y_test = df_test[target_col].values
    
    # Create validation split from training data (80% train, 20% validation)
    val_split = int(0.8 * len(X_train_full))
    X_train = X_train_full[:val_split]
    y_train = y_train_full[:val_split]
    X_val = X_train_full[val_split:]
    y_val = y_train_full[val_split:]
    
    output_lines.append(f"\n   X_train shape: {X_train.shape}")
    output_lines.append(f"   y_train shape: {y_train.shape}")
    output_lines.append(f"   X_val shape: {X_val.shape}")
    output_lines.append(f"   y_val shape: {y_val.shape}")
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
    output_lines.append("\n4. Creating neural network model...")
    model = FeedForwardNeuralNetwork(
        hidden_layers=[64, 32],
        activation='relu',
        learning_rate=0.001,
        batch_size=512,
        epochs=80,
        alpha=0.0001,
        random_seed=3326,
        verbose=False
    )
    
    output_lines.append(f"   Architecture: {model.hidden_layers}")
    output_lines.append(f"   Activation: {model.activation}")
    output_lines.append(f"   Learning rate: {model.learning_rate}")
    output_lines.append(f"   Batch size: {model.batch_size}")
    output_lines.append(f"   Epochs: {model.epochs}")
    output_lines.append(f"   L2 Regularization (alpha): {model.alpha}")
    
    # 5. Train the model
    output_lines.append("\n5. Training model (with validation monitoring)...")
    output_lines.append("-" * 80)
    model.fit(X_train, y_train, X_val, y_val)
    output_lines.append("Training completed!")
    
    # 6. Make predictions
    output_lines.append("\n6. Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Apply ReLU logic: clip negative predictions to 0 (claim frequency can't be negative)
    y_train_pred = np.maximum(0, y_train_pred)
    y_test_pred = np.maximum(0, y_test_pred)
    output_lines.append("   Applied ReLU clipping to predictions (no negative values)")
    
    # 7. Evaluate performance
    output_lines.append("\n7. Model Performance:")
    output_lines.append("-" * 80)
    
    # R² Score
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    # MSE
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    
    # RMSE
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    # MAE
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    
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
        for i in claim_indices[:10]:  # Show first 10
            error = abs(y_test[i] - y_test_pred[i])
            output_lines.append(f"{i:<8} {y_test[i]:<15.6f} {y_test_pred[i]:<15.6f} {error:<15.6f}")
    
    # 10. Save model
    output_lines.append("\n10. Saving model...")
    model_path = 'models/ffnn_claims_model.pkl'
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    output_lines.append(f"   Model saved to: {model_path}")
    
    # 11. Save predictions
    output_lines.append("\n11. Saving predictions to CSV...")
    results_df = pd.DataFrame({
        'True_ClaimFrequency': y_test,
        'Predicted_ClaimFrequency': y_test_pred,
        'Absolute_Error': np.abs(y_test - y_test_pred),
        'Squared_Error': (y_test - y_test_pred) ** 2
    })
    
    results_path = 'results/ffnn_predictions.csv'
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(results_path, index=False)
    output_lines.append(f"   Predictions saved to: {results_path}")
    
    # 12. Summary statistics of predictions
    output_lines.append("\n12. Prediction Statistics:")
    output_lines.append("-" * 80)
    output_lines.append(f"   Predictions - Mean: {y_test_pred.mean():.6f}")
    output_lines.append(f"   Predictions - Std: {y_test_pred.std():.6f}")
    output_lines.append(f"   Predictions - Min: {y_test_pred.min():.6f}")
    output_lines.append(f"   Predictions - Max: {y_test_pred.max():.6f}")
    output_lines.append(f"   Predictions < 0: {(y_test_pred < 0).sum()}")
    
    output_lines.append("\n" + "=" * 80)
    output_lines.append("Training and Testing Completed Successfully!")
    output_lines.append("=" * 80)
    
    # Write all output to file
    output_file = 'results/ffnn_report.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Training M21 report saved to: {output_file}")
    
    return model, y_test, y_test_pred


if __name__ == "__main__":
    # Train and evaluate the custom model
    model, y_test, y_test_pred = train_and_evaluate_custom_model()
