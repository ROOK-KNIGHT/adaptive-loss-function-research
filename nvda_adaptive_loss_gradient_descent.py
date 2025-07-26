"""
Adaptive Loss Function Gradient Descent for NVDA Stock Price Prediction
Modified version to handle NVDA historical stock data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class AdaptiveLossFunction(nn.Module):
    """
    Adaptive Loss Function that combines MSE with weighted correlation terms
    based on feature-target covariances, adapted for stock price prediction.
    """
    
    def __init__(self, feature_names: List[str], initial_weights: Dict[str, float] = None):
        super(AdaptiveLossFunction, self).__init__()
        self.feature_names = feature_names
        self.mse_loss = nn.MSELoss()
        
        # Initialize weights for correlation terms
        if initial_weights is None:
            # Default weights for stock features
            self.weights = {name: 0.1 if i == 0 else 0.05 for i, name in enumerate(feature_names)}
        else:
            self.weights = initial_weights.copy()
        
        # Weight constraints [0.01, 0.5] for stability
        self.min_weight = 0.01
        self.max_weight = 0.5
        
        # Track covariances for analysis
        self.covariance_history = {name: [] for name in feature_names}
        self.weight_history = {name: [] for name in feature_names}
        
    def compute_covariance_loss(self, features: torch.Tensor, target: torch.Tensor, 
                               feature_idx: int) -> torch.Tensor:
        """Compute covariance-based loss term for a specific feature"""
        feature_col = features[:, feature_idx]
        
        # Center the data (subtract mean)
        feature_centered = feature_col - torch.mean(feature_col)
        target_centered = target.squeeze() - torch.mean(target.squeeze())
        
        # Compute covariance
        covariance = torch.mean(feature_centered * target_centered)
        
        # Return squared covariance as loss term (we want to minimize large covariances)
        return torch.abs(covariance)
    
    def forward(self, predictions: torch.Tensor, target: torch.Tensor, 
                features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass computing total loss with adaptive correlation terms
        """
        # Base MSE loss
        mse_loss = self.mse_loss(predictions, target)
        
        # Correlation losses
        correlation_losses = {}
        total_correlation_loss = 0.0
        
        for i, feature_name in enumerate(self.feature_names):
            cov_loss = self.compute_covariance_loss(features, target, i)
            correlation_losses[feature_name] = cov_loss.item()
            total_correlation_loss += self.weights[feature_name] * cov_loss
        
        # Total adaptive loss
        total_loss = mse_loss + total_correlation_loss
        
        # Store covariances for weight updates
        for feature_name, cov_val in correlation_losses.items():
            self.covariance_history[feature_name].append(cov_val)
        
        return total_loss, correlation_losses
    
    def update_weights(self, epoch: int):
        """
        Update weights every 5 epochs based on covariance strength
        """
        if epoch % 5 != 0 or epoch == 0:
            return
        
        print(f"\nUpdating weights at epoch {epoch}:")
        
        for feature_name in self.feature_names:
            if len(self.covariance_history[feature_name]) > 0:
                # Get recent covariances (last 5 epochs)
                recent_covs = self.covariance_history[feature_name][-5:]
                avg_cov = np.mean(recent_covs)
                
                # Normalize covariance to [0, 1] range
                normalized_cov = 2 / (1 + np.exp(-avg_cov)) - 1
                normalized_cov = max(0, min(1, normalized_cov))
                
                # Update weight: new_weight = old_weight * (1 + 0.1 * normalized_covariance)
                old_weight = self.weights[feature_name]
                new_weight = old_weight * (1 + 0.1 * normalized_cov)
                
                # Apply constraints [0.01, 0.5]
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                
                self.weights[feature_name] = new_weight
                self.weight_history[feature_name].append(new_weight)
                
                print(f"  {feature_name}: {old_weight:.4f} -> {new_weight:.4f} "
                      f"(avg_cov: {avg_cov:.4f}, norm_cov: {normalized_cov:.4f})")

class StockPricePredictor(nn.Module):
    """Neural network for stock price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super(StockPricePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def load_nvda_data() -> pd.DataFrame:
    """
    Load and preprocess the NVDA historical stock data
    """
    # Load from CSV file
    df = pd.read_csv('NVDA_historical_data-2.csv')
    print("Loaded NVDA data from CSV file")
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    
    # Convert datetime to proper format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Clean numerical columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing data
    df = df.dropna(subset=numeric_columns)
    
    # Create technical indicators as features
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Price-based features
    df['price_range'] = df['high'] - df['low']
    df['price_change'] = df['close'] - df['open']
    df['price_volatility'] = df['price_range'] / df['open']
    
    # Moving averages (using rolling windows)
    df['ma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['ma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['ma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    # Relative strength indicators
    df['rsi_signal'] = (df['close'] - df['ma_10']) / df['ma_10']
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Lag features (previous values)
    df['prev_close'] = df['close'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    # Time-based features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0
    
    # Create final dataset with selected features
    feature_columns = [
        'open', 'high', 'low', 'volume',
        'price_range', 'price_change', 'price_volatility',
        'ma_5', 'ma_10', 'ma_20', 'rsi_signal',
        'volume_ratio', 'prev_close', 'prev_volume',
        'prev_high', 'prev_low', 'time_of_day'
    ]
    
    # Remove rows with NaN values (from lag features)
    df = df.dropna()
    
    # Create processed dataframe
    processed_df = df[feature_columns + ['close']].copy()
    processed_df.rename(columns={'close': 'target_price'}, inplace=True)
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Features: {feature_columns}")
    
    return processed_df

def check_multicollinearity(X: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Check for multicollinearity using Variance Inflation Factor (VIF)
    """
    print("Checking multicollinearity using VIF...")
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("\nVIF Results:")
    print(vif_data)
    
    # Identify problematic features
    problematic_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
    if problematic_features:
        print(f"\nWarning: Features with VIF > {threshold}: {problematic_features}")
        print("Consider removing or combining these features to reduce multicollinearity.")
    else:
        print(f"\nAll features have VIF <= {threshold}. No multicollinearity issues detected.")
    
    return vif_data

def train_model(model, train_loader, loss_function, optimizer, epochs: int, 
                model_name: str) -> Dict[str, List[float]]:
    """
    Train model and return training history
    """
    history = {
        'loss': [],
        'mse_loss': [],
        'correlation_losses': {name: [] for name in loss_function.feature_names} if hasattr(loss_function, 'feature_names') else {}
    }
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_corr_losses = {name: 0.0 for name in history['correlation_losses'].keys()}
        
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            
            predictions = model(batch_features)
            
            if hasattr(loss_function, 'feature_names'):
                # Adaptive loss function
                loss, corr_losses = loss_function(predictions, batch_targets, batch_features)
                
                # Track correlation losses
                for name, val in corr_losses.items():
                    epoch_corr_losses[name] += val
                
                # Also compute MSE for comparison
                mse_loss = nn.MSELoss()(predictions, batch_targets)
                epoch_mse += mse_loss.item()
                
            else:
                # Standard loss function
                loss = loss_function(predictions, batch_targets)
                epoch_mse += loss.item()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Average losses over batches
        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        
        history['loss'].append(avg_loss)
        history['mse_loss'].append(avg_mse)
        
        # Average correlation losses
        for name in epoch_corr_losses.keys():
            avg_corr_loss = epoch_corr_losses[name] / len(train_loader)
            history['correlation_losses'][name].append(avg_corr_loss)
        
        # Update weights for adaptive loss
        if hasattr(loss_function, 'update_weights'):
            loss_function.update_weights(epoch)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{model_name} - Epoch {epoch:3d}: Loss = {avg_loss:.6f}, MSE = {avg_mse:.6f}")
    
    return history

def plot_training_comparison(adaptive_history: Dict, standard_history: Dict, 
                           adaptive_loss_fn: AdaptiveLossFunction):
    """
    Create comprehensive plots comparing adaptive vs standard training
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(len(adaptive_history['loss']))
    
    # Plot 1: Loss comparison
    axes[0, 0].plot(epochs, adaptive_history['loss'], label='Adaptive Loss', linewidth=2)
    axes[0, 0].plot(epochs, standard_history['loss'], label='Standard MSE', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison - NVDA Stock Prediction')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MSE comparison (fair comparison)
    axes[0, 1].plot(epochs, adaptive_history['mse_loss'], label='Adaptive (MSE component)', linewidth=2)
    axes[0, 1].plot(epochs, standard_history['mse_loss'], label='Standard MSE', linewidth=2)
    axes[0, 1].set_title('MSE Component Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weight evolution
    for feature_name in adaptive_loss_fn.feature_names:
        weight_epochs = range(0, len(epochs), 5)  # Weights updated every 5 epochs
        weights = adaptive_loss_fn.weight_history[feature_name]
        if weights:
            axes[1, 0].plot(weight_epochs[:len(weights)], weights, 
                          label=f'{feature_name}', marker='o', linewidth=2)
    
    axes[1, 0].set_title('Adaptive Weight Evolution')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Weight Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation losses
    for feature_name, losses in adaptive_history['correlation_losses'].items():
        axes[1, 1].plot(epochs, losses, label=f'{feature_name}', linewidth=2)
    
    axes[1, 1].set_title('Correlation Loss Components')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Correlation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nvda_training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(model, X_test, y_test, scaler_y, title="Stock Price Predictions"):
    """
    Plot actual vs predicted stock prices
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    # Inverse transform to get actual price values
    y_test_actual = scaler_y.inverse_transform(y_test.numpy())
    predictions_actual = scaler_y.inverse_transform(predictions)
    
    plt.figure(figsize=(12, 6))
    
    # Plot first 100 predictions for clarity
    n_plot = min(100, len(y_test_actual))
    x_axis = range(n_plot)
    
    plt.plot(x_axis, y_test_actual[:n_plot], label='Actual Price', linewidth=2, alpha=0.8)
    plt.plot(x_axis, predictions_actual[:n_plot], label='Predicted Price', linewidth=2, alpha=0.8)
    
    plt.title(f'{title} - NVDA Stock Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'nvda_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and return metrics
    mse = np.mean((y_test_actual - predictions_actual) ** 2)
    mae = np.mean(np.abs(y_test_actual - predictions_actual))
    mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
    
    return mse, mae, mape

def main():
    """
    Main function implementing the adaptive loss function experiment for NVDA stock prediction
    """
    print("=== Adaptive Loss Function for NVDA Stock Price Prediction ===")
    print("Implementation based on the research paper, adapted for financial data\n")
    
    # 1. Load NVDA dataset
    print("1. Loading NVDA historical stock data...")
    df = load_nvda_data()
    print(f"Dataset shape: {df.shape}")
    print("\nDataset preview:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
    
    # 2. Prepare features and target
    feature_columns = [col for col in df.columns if col != 'target_price']
    X = df[feature_columns]
    y = df['target_price'].values.reshape(-1, 1)
    
    # 3. Check multicollinearity
    print("\n2. Checking multicollinearity...")
    vif_results = check_multicollinearity(X)
    
    # 4. Standardize features and target
    print("\n3. Standardizing features and target...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 5. Train-test split (use temporal split for time series)
    print("\n4. Creating train-test split (temporal split)...")
    split_idx = int(0.8 * len(X_scaled))
    
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y_scaled[:split_idx]
    y_test = y_scaled[split_idx:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)  # Don't shuffle time series
    
    # 6. Initialize models and loss functions
    print("\n5. Initializing models and loss functions...")
    
    # Models
    adaptive_model = StockPricePredictor(input_size=len(feature_columns))
    standard_model = StockPricePredictor(input_size=len(feature_columns))
    
    # Loss functions - select most correlated features for adaptive loss
    correlations = df.corr()['target_price'].abs().sort_values(ascending=False)
    top_features = correlations.index[1:6].tolist()  # Exclude 'target_price' itself, take top 5
    print(f"Selected features for adaptive loss: {top_features}")
    
    # Map to indices in feature_columns
    selected_indices = [feature_columns.index(feat) for feat in top_features if feat in feature_columns]
    selected_feature_names = [feature_columns[i] for i in selected_indices]
    
    adaptive_loss = AdaptiveLossFunction(
        feature_names=selected_feature_names,
        initial_weights={name: 0.15 if i == 0 else 0.08 for i, name in enumerate(selected_feature_names)}
    )
    standard_loss = nn.MSELoss()
    
    # Optimizers
    adaptive_optimizer = optim.Adam(adaptive_model.parameters(), lr=0.001, weight_decay=1e-5)
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 7. Train models
    print("\n6. Training models...")
    epochs = 100
    
    print("\nTraining with Adaptive Loss Function:")
    adaptive_history = train_model(
        adaptive_model, train_loader, adaptive_loss, adaptive_optimizer, 
        epochs, "Adaptive"
    )
    
    print("\nTraining with Standard MSE Loss:")
    standard_history = train_model(
        standard_model, train_loader, standard_loss, standard_optimizer, 
        epochs, "Standard"
    )
    
    # 8. Evaluate models
    print("\n7. Evaluating models...")
    
    adaptive_model.eval()
    standard_model.eval()
    
    with torch.no_grad():
        adaptive_pred = adaptive_model(X_test_tensor)
        standard_pred = standard_model(X_test_tensor)
        
        adaptive_mse = nn.MSELoss()(adaptive_pred, y_test_tensor).item()
        standard_mse = nn.MSELoss()(standard_pred, y_test_tensor).item()
        
        print(f"\nTest Results (Normalized):")
        print(f"Adaptive Loss MSE: {adaptive_mse:.6f}")
        print(f"Standard MSE: {standard_mse:.6f}")
        print(f"Improvement: {((standard_mse - adaptive_mse) / standard_mse * 100):.2f}%")
    
    # 9. Calculate actual price metrics
    print("\n8. Calculating actual price prediction metrics...")
    
    adaptive_mse_actual, adaptive_mae, adaptive_mape = plot_predictions(
        adaptive_model, X_test_tensor, y_test_tensor, scaler_y, "Adaptive Model Predictions"
    )
    
    standard_mse_actual, standard_mae, standard_mape = plot_predictions(
        standard_model, X_test_tensor, y_test_tensor, scaler_y, "Standard Model Predictions"
    )
    
    print(f"\nActual Price Prediction Metrics:")
    print(f"Adaptive Model:")
    print(f"  MSE: ${adaptive_mse_actual:.4f}")
    print(f"  MAE: ${adaptive_mae:.4f}")
    print(f"  MAPE: {adaptive_mape:.2f}%")
    
    print(f"Standard Model:")
    print(f"  MSE: ${standard_mse_actual:.4f}")
    print(f"  MAE: ${standard_mae:.4f}")
    print(f"  MAPE: {standard_mape:.2f}%")
    
    # 10. Analyze convergence
    print("\n9. Convergence Analysis:")
    
    # Find epoch where each method reaches 90% of final performance
    adaptive_final = adaptive_history['mse_loss'][-1]
    standard_final = standard_history['mse_loss'][-1]
    
    adaptive_target = adaptive_final * 1.1  # 90% of final performance
    standard_target = standard_final * 1.1
    
    adaptive_convergence = next((i for i, loss in enumerate(adaptive_history['mse_loss']) 
                               if loss <= adaptive_target), epochs)
    standard_convergence = next((i for i, loss in enumerate(standard_history['mse_loss']) 
                               if loss <= standard_target), epochs)
    
    print(f"Adaptive method converged at epoch: {adaptive_convergence}")
    print(f"Standard method converged at epoch: {standard_convergence}")
    
    if standard_convergence > adaptive_convergence:
        speedup = ((standard_convergence - adaptive_convergence) / standard_convergence * 100)
        print(f"Convergence speedup: {speedup:.1f}%")
    
    # 11. Create visualizations
    print("\n10. Creating visualizations...")
    plot_training_comparison(adaptive_history, standard_history, adaptive_loss)
    
    # 12. Summary
    print("\n=== NVDA STOCK PREDICTION EXPERIMENT SUMMARY ===")
    print(f"Dataset size: {len(df)} samples")
    print(f"Features used: {feature_columns}")
    print(f"Selected correlated features: {selected_feature_names}")
    print(f"Training epochs: {epochs}")
    print(f"\nFinal Training Loss:")
    print(f"  Adaptive: {adaptive_history['loss'][-1]:.6f}")
    print(f"  Standard: {standard_history['loss'][-1]:.6f}")
    print(f"\nFinal Test MSE (Normalized):")
    print(f"  Adaptive: {adaptive_mse:.6f}")
    print(f"  Standard: {standard_mse:.6f}")
    print(f"  Improvement: {((standard_mse - adaptive_mse) / standard_mse * 100):.2f}%")
    print(f"\nActual Price Prediction Performance:")
    print(f"  Adaptive MAE: ${adaptive_mae:.4f}")
    print(f"  Standard MAE: ${standard_mae:.4f}")
    print(f"  MAE Improvement: {((standard_mae - adaptive_mae) / standard_mae * 100):.2f}%")
    print(f"\nConvergence Speed:")
    print(f"  Adaptive: {adaptive_convergence} epochs")
    print(f"  Standard: {standard_convergence} epochs")
    if standard_convergence > adaptive_convergence:
        print(f"  Speedup: {speedup:.1f}%")
    
    print(f"\nFinal adaptive weights:")
    for name, weight in adaptive_loss.weights.items():
        print(f"  {name}: {weight:.4f}")

if __name__ == "__main__":
    main()
