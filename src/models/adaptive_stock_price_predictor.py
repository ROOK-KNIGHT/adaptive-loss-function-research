"""
Ticker-Agnostic Adaptive Loss Function with Improved Evaluation
Integrates bias correction and comprehensive financial metrics with fresh data fetching

Usage:
    python3 adaptive_stock_price_predictor.py TICKER [--use-existing]
    
Examples:
    # Fetch fresh data and run analysis
    python3 adaptive_stock_price_predictor.py TSLA
    python3 adaptive_stock_price_predictor.py NVDA
    
    # Use existing data file (if available)
    python3 adaptive_stock_price_predictor.py TSLA --use-existing
    python3 adaptive_stock_price_predictor.py AAPL --use-existing

Features:
    - Automatically fetches fresh 10-year historical data using Schwab API
    - Ticker-agnostic: works with any stock symbol
    - Adaptive loss function with temporal consistency, volatility, and smoothness penalties
    - Comprehensive bias-corrected evaluation metrics
    - Financial performance metrics (directional accuracy, return correlation, etc.)
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
import sys
import os
import argparse
import subprocess
import glob
from datetime import datetime
warnings.filterwarnings('ignore')

# Import our improved evaluation functions
from stock_evaluation_metrics import (
    comprehensive_model_evaluation, 
    compare_models_comprehensive,
    calculate_bias_corrected_metrics,
    calculate_directional_accuracy
)

class AdaptiveLossFunction(nn.Module):
    """
    Enhanced Adaptive Loss Function with multiple improvements:
    1. Temporal consistency penalty
    2. Volatility-aware weighting  
    3. Dynamic learning rate adjustment
    4. Pattern smoothness regularization
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
        
        # Rebalanced weight constraints and parameters
        self.min_weight = 0.01
        self.max_weight = 0.2
        self.learning_rate_factor = 0.1  # More conservative weight updates
        
        # Rebalanced enhancement components - reduce penalty weights
        self.temporal_consistency_weight = 0.005  # Reduced from 0.03
        self.volatility_penalty_weight = 0.002   # Reduced from 0.015
        self.smoothness_penalty_weight = 0.001   # Reduced from 0.01
        
        # Track covariances and additional metrics for analysis
        self.covariance_history = {name: [] for name in feature_names}
        self.weight_history = {name: [] for name in feature_names}
        self.prediction_history = []
        self.target_history = []
        
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
    
    def compute_temporal_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency penalty to encourage smoother predictions"""
        if len(predictions) < 2:
            return torch.tensor(0.0)
        
        # Calculate differences between consecutive predictions
        pred_diffs = torch.diff(predictions.squeeze())
        
        # Penalize large jumps in predictions
        temporal_loss = torch.mean(torch.abs(pred_diffs))
        return temporal_loss
    
    def compute_volatility_penalty(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute volatility penalty to match target volatility patterns"""
        if len(predictions) < 3:
            return torch.tensor(0.0)
        
        # Calculate volatility (standard deviation of recent changes)
        pred_volatility = torch.std(torch.diff(predictions.squeeze()))
        target_volatility = torch.std(torch.diff(target.squeeze()))
        
        # Penalize volatility mismatch
        volatility_loss = torch.abs(pred_volatility - target_volatility)
        return volatility_loss
    
    def compute_smoothness_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute smoothness penalty using second derivatives"""
        if len(predictions) < 3:
            return torch.tensor(0.0)
        
        # Calculate second derivatives (acceleration in predictions)
        pred_squeeze = predictions.squeeze()
        first_diff = torch.diff(pred_squeeze)
        second_diff = torch.diff(first_diff)
        
        # Penalize high acceleration (non-smooth changes)
        smoothness_loss = torch.mean(torch.abs(second_diff))
        return smoothness_loss

    def forward(self, predictions: torch.Tensor, target: torch.Tensor, 
                features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Enhanced forward pass with temporal consistency, volatility, and smoothness penalties
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
        
        # Enhanced penalty terms
        temporal_loss = self.compute_temporal_consistency_loss(predictions)
        volatility_loss = self.compute_volatility_penalty(predictions, target)
        smoothness_loss = self.compute_smoothness_penalty(predictions)
        
        # Total enhanced adaptive loss
        total_loss = (mse_loss + 
                     total_correlation_loss + 
                     self.temporal_consistency_weight * temporal_loss +
                     self.volatility_penalty_weight * volatility_loss +
                     self.smoothness_penalty_weight * smoothness_loss)
        
        # Store additional metrics
        correlation_losses['temporal_consistency'] = temporal_loss.item()
        correlation_losses['volatility_penalty'] = volatility_loss.item()
        correlation_losses['smoothness_penalty'] = smoothness_loss.item()
        
        # Store covariances for weight updates
        for feature_name, cov_val in correlation_losses.items():
            if feature_name in self.covariance_history:
                self.covariance_history[feature_name].append(cov_val)
        
        # Store predictions and targets for analysis
        self.prediction_history.extend(predictions.detach().numpy().flatten().tolist())
        self.target_history.extend(target.detach().numpy().flatten().tolist())
        
        return total_loss, correlation_losses
    
    def update_weights(self, epoch: int):
        """
        Conservative weight update mechanism focused on pattern matching
        """
        if epoch % 5 != 0 or epoch == 0:  # Less frequent updates for stability
            return
        
        print(f"\nUpdating weights at epoch {epoch}:")
        
        for feature_name in self.feature_names:
            if len(self.covariance_history[feature_name]) > 0:
                # Get recent covariances (last 5 epochs for stability)
                recent_covs = self.covariance_history[feature_name][-5:]
                avg_cov = np.mean(recent_covs)
                
                # Conservative normalization
                normalized_cov = 2 / (1 + np.exp(-avg_cov * 5)) - 1  # Less sensitive scaling
                normalized_cov = max(0, min(1, normalized_cov))
                
                # Conservative weight update
                old_weight = self.weights[feature_name]
                weight_adjustment = self.learning_rate_factor * normalized_cov * 0.5  # Reduced adjustment
                new_weight = old_weight * (1 + weight_adjustment)
                
                # Apply constraints
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                
                # More conservative stability check - prevent rapid changes
                max_change = 0.02  # Maximum 2% change per update
                if abs(new_weight - old_weight) > max_change:
                    new_weight = old_weight + np.sign(new_weight - old_weight) * max_change
                
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

def fetch_fresh_data(ticker: str) -> str:
    """
    Fetch fresh data for the ticker using the fetch_10_year_daily_data.py script
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        str: Path to the newly created data file
    """
    print(f"Fetching fresh data for {ticker}...")
    
    try:
        # Run the fetch script
        result = subprocess.run([
            'python3', 'fetch_10_year_daily_data.py', ticker
        ], capture_output=True, text=True, check=True)
        
        print("Data fetch completed successfully!")
        
        # Find the most recent data file for this ticker
        pattern = f'data/historical/{ticker}_10_year_daily_data_*.csv'
        files = glob.glob(pattern)
        
        if files:
            # Sort by modification time and get the most recent
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
        else:
            raise FileNotFoundError(f"No data file found after fetching for {ticker}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error fetching fresh data for {ticker}")
        print(f"Exit code: {e.returncode}")
        
        if "ModuleNotFoundError" in e.stderr and "requests" in e.stderr:
            print("\n🔧 DEPENDENCY ISSUE DETECTED:")
            print("The data fetching script requires the 'requests' module.")
            print("Please install the missing dependency:")
            print("  pip install requests")
            print("\nOr install all requirements:")
            print("  pip install -r requirements.txt")
            print(f"\nAlternatively, you can use existing data with:")
            print(f"  python3 adaptive_stock_price_predictor.py {ticker} --use-existing")
        elif "Manual authentication required" in e.stdout or "expires_at" in e.stdout:
            print("\n🔐 SCHWAB API AUTHENTICATION REQUIRED:")
            print("The Schwab API requires interactive authentication to fetch fresh data.")
            print("This process requires manual user input and cannot be automated.")
            print("\n📋 To authenticate:")
            print("1. Run the data fetching script separately:")
            print(f"   python3 fetch_10_year_daily_data.py {ticker}")
            print("2. Follow the authentication prompts")
            print("3. Then run this script with existing data:")
            print(f"   python3 adaptive_stock_price_predictor.py {ticker} --use-existing")
            print("\n💡 Alternative: Use existing data if available:")
            print(f"   python3 adaptive_stock_price_predictor.py {ticker} --use-existing")
        else:
            print(f"\nDetailed error information:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        
        raise RuntimeError(f"Failed to fetch fresh data for {ticker}. See error details above.")
    except Exception as e:
        print(f"Unexpected error during data fetch: {e}")
        raise

def load_data(ticker: str, fetch_fresh: bool = True) -> pd.DataFrame:
    """
    Load and preprocess historical stock data for the specified ticker
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA', 'NVDA')
        fetch_fresh: Whether to fetch fresh data or use existing file
    """
    data_file = None
    
    if fetch_fresh:
        try:
            # Try to fetch fresh data
            data_file = fetch_fresh_data(ticker)
        except RuntimeError as e:
            print(f"\n⚠️  Fresh data fetching failed. Attempting to use existing data...")
            # Look for existing data files as fallback
            pattern = f'data/historical/{ticker}_10_year_daily_data*.csv'
            files = glob.glob(pattern)
            
            if files:
                # Use the most recent file
                data_file = max(files, key=os.path.getmtime)
                print(f"✅ Found existing data file: {data_file}")
                print("Proceeding with analysis using existing data...")
            else:
                print(f"\n❌ No existing data files found for {ticker}")
                print("Cannot proceed without data. Please:")
                print("1. Authenticate with Schwab API manually:")
                print(f"   python3 fetch_10_year_daily_data.py {ticker}")
                print("2. Or provide existing data files in data/historical/")
                raise FileNotFoundError(f"No data available for {ticker}")
    else:
        # Look for existing data files
        pattern = f'data/historical/{ticker}_10_year_daily_data*.csv'
        files = glob.glob(pattern)
        
        if files:
            # Use the most recent file
            data_file = max(files, key=os.path.getmtime)
            print(f"Using existing data file: {data_file}")
        else:
            print(f"No existing data file found for {ticker}")
            print("Attempting to fetch fresh data as fallback...")
            try:
                data_file = fetch_fresh_data(ticker)
            except RuntimeError as e:
                print(f"\n❌ Both existing data lookup and fresh data fetching failed")
                print("Please authenticate with Schwab API manually:")
                print(f"   python3 fetch_10_year_daily_data.py {ticker}")
                raise FileNotFoundError(f"No data available for {ticker}")
    
    # Load from CSV file
    df = pd.read_csv(data_file)
    print(f"Loaded {ticker} data from {data_file}")
    
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

def train_model(model, train_loader, loss_function, optimizer, epochs: int, 
                model_name: str) -> Dict[str, List[float]]:
    """
    Train model and return training history
    """
    # Initialize history with all possible loss components
    if hasattr(loss_function, 'feature_names'):
        all_loss_names = loss_function.feature_names + ['temporal_consistency', 'volatility_penalty', 'smoothness_penalty']
    else:
        all_loss_names = []
    
    history = {
        'loss': [],
        'mse_loss': [],
        'correlation_losses': {name: [] for name in all_loss_names}
    }
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_corr_losses = {name: 0.0 for name in all_loss_names}
        
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            
            predictions = model(batch_features)
            
            if hasattr(loss_function, 'feature_names'):
                # Adaptive loss function
                loss, corr_losses = loss_function(predictions, batch_targets, batch_features)
                
                # Track all correlation losses (including new penalty terms)
                for name, val in corr_losses.items():
                    if name in epoch_corr_losses:
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
        
        # Print progress with enhanced penalty information
        if epoch % 20 == 0 or epoch == epochs - 1:
            if hasattr(loss_function, 'feature_names'):
                temporal_loss = history['correlation_losses'].get('temporal_consistency', [0])[-1]
                volatility_loss = history['correlation_losses'].get('volatility_penalty', [0])[-1]
                smoothness_loss = history['correlation_losses'].get('smoothness_penalty', [0])[-1]
                print(f"{model_name} - Epoch {epoch:3d}: Loss = {avg_loss:.6f}, MSE = {avg_mse:.6f}")
                print(f"    Penalties - Temporal: {temporal_loss:.6f}, Volatility: {volatility_loss:.6f}, Smoothness: {smoothness_loss:.6f}")
            else:
                print(f"{model_name} - Epoch {epoch:3d}: Loss = {avg_loss:.6f}, MSE = {avg_mse:.6f}")
    
    return history

def main():
    """
    Main function with improved evaluation and scaling diagnostics
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ticker-Agnostic Adaptive Loss Function with Fresh Data Fetching')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., TSLA, NVDA)')
    parser.add_argument('--use-existing', action='store_true', 
                       help='Use existing data file instead of fetching fresh data')
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    fetch_fresh = not args.use_existing
    
    print(f"=== {ticker} Adaptive Loss Function with Improved Evaluation ===")
    print("Addresses systematic bias and provides comprehensive financial metrics")
    print(f"Data mode: {'Using existing data' if args.use_existing else 'Fetching fresh data'}\n")
    
    # 1. Load dataset
    print(f"1. Loading {ticker} historical stock data...")
    df = load_data(ticker, fetch_fresh=fetch_fresh)
    
    # 2. Prepare features and target
    feature_columns = [col for col in df.columns if col != 'target_price']
    X = df[feature_columns]
    y = df['target_price'].values.reshape(-1, 1)
    
    # 3. FIXED TEMPORAL SCALING - Split first, then scale
    print("\n2. Creating temporal split BEFORE scaling (FIXED DATA LEAKAGE)...")
    
    # First, do temporal split on raw data
    split_idx = int(0.8 * len(X))
    
    X_train_raw = X.iloc[:split_idx]
    X_test_raw = X.iloc[split_idx:]
    y_train_raw = y[:split_idx]
    y_test_raw = y[split_idx:]
    
    print(f"Training set size: {len(X_train_raw)}")
    print(f"Test set size: {len(X_test_raw)}")
    
    # Show raw data distributions BEFORE scaling
    print("\n=== RAW DATA DISTRIBUTIONS (BEFORE SCALING) ===")
    print(f"Training target range: ${y_train_raw.min():.2f} to ${y_train_raw.max():.2f}")
    print(f"Training target mean: ${y_train_raw.mean():.2f}, std: ${y_train_raw.std():.2f}")
    print(f"Test target range: ${y_test_raw.min():.2f} to ${y_test_raw.max():.2f}")
    print(f"Test target mean: ${y_test_raw.mean():.2f}, std: ${y_test_raw.std():.2f}")
    
    # Now fit scalers ONLY on training data
    print("\n3. Fitting scalers ONLY on training data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit scalers only on training data
    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw)
    
    # Apply same scaling to test data (no fitting!)
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw)
    
    # Scaling diagnostics - now showing proper temporal scaling
    print("\n=== FIXED SCALING DIAGNOSTICS ===")
    print("TRAINING DATA (used to fit scalers):")
    print(f"  Original range: ${y_train_raw.min():.2f} to ${y_train_raw.max():.2f}")
    print(f"  Original mean: ${y_train_raw.mean():.2f}, std: ${y_train_raw.std():.2f}")
    print(f"  Scaled range: {y_train.min():.4f} to {y_train.max():.4f}")
    print(f"  Scaled mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    
    print("\nTEST DATA (scalers applied, not fitted):")
    print(f"  Original range: ${y_test_raw.min():.2f} to ${y_test_raw.max():.2f}")
    print(f"  Original mean: ${y_test_raw.mean():.2f}, std: ${y_test_raw.std():.2f}")
    print(f"  Scaled range: {y_test.min():.4f} to {y_test.max():.4f}")
    print(f"  Scaled mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
    
    print(f"\nScaler statistics (fitted on training data only):")
    print(f"  Scaler_y mean: ${scaler_y.mean_[0]:.2f}, scale: ${scaler_y.scale_[0]:.2f}")
    
    # Verify no data leakage
    print(f"\n=== DATA LEAKAGE VERIFICATION ===")
    print(f"Training period: ${y_train_raw.min():.2f} to ${y_train_raw.max():.2f}")
    print(f"Test period: ${y_test_raw.min():.2f} to ${y_test_raw.max():.2f}")
    if y_test_raw.min() >= y_train_raw.max():
        print("✓ GOOD: Test data represents future prices (no temporal overlap)")
    else:
        print("⚠ WARNING: Temporal overlap detected")
    
    # Show distribution shift
    train_test_mean_diff = y_test_raw.mean() - y_train_raw.mean()
    print(f"Distribution shift: Test mean is ${train_test_mean_diff:.2f} higher than training mean")
    print(f"This represents the natural price appreciation over time")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    # 5. Initialize models and loss functions
    print("\n4. Initializing models and loss functions...")
    
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
    
    # 6. Train models
    print("\n5. Training models...")
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
    
    # 7. Comprehensive Evaluation
    print("\n6. Comprehensive Model Evaluation...")
    
    # Evaluate both models with improved metrics and distribution shift adjustment
    adaptive_results = comprehensive_model_evaluation(
        adaptive_model, X_test_tensor, y_test_tensor, scaler_y, "Adaptive Model", 
        train_test_mean_diff=train_test_mean_diff
    )
    
    standard_results = comprehensive_model_evaluation(
        standard_model, X_test_tensor, y_test_tensor, scaler_y, "Standard Model",
        train_test_mean_diff=train_test_mean_diff
    )
    
    # 8. Compare models comprehensively
    compare_models_comprehensive(adaptive_results, standard_results)
    
    # 9. Quick bias analysis summary
    print("\n" + "="*60)
    print("BIAS ANALYSIS SUMMARY")
    print("="*60)
    
    adaptive_bias = adaptive_results['bias_metrics']['bias']
    standard_bias = standard_results['bias_metrics']['bias']
    
    print(f"Systematic Bias:")
    print(f"  Adaptive Model: ${adaptive_bias:.4f}")
    print(f"  Standard Model: ${standard_bias:.4f}")
    print(f"  Difference: ${abs(adaptive_bias - standard_bias):.4f}")
    
    # Calculate improvement in bias-corrected metrics
    adaptive_corrected_mae = adaptive_results['bias_metrics']['corrected_mae']
    standard_corrected_mae = standard_results['bias_metrics']['corrected_mae']
    mae_improvement = ((standard_corrected_mae - adaptive_corrected_mae) / standard_corrected_mae * 100)
    
    print(f"\nBias-Corrected MAE Comparison:")
    print(f"  Adaptive: ${adaptive_corrected_mae:.4f}")
    print(f"  Standard: ${standard_corrected_mae:.4f}")
    print(f"  Improvement: {mae_improvement:.2f}%")
    
    # Directional accuracy comparison
    adaptive_dir_acc = adaptive_results['directional_accuracy']
    standard_dir_acc = standard_results['directional_accuracy']
    dir_improvement = (adaptive_dir_acc - standard_dir_acc) * 100
    
    print(f"\nDirectional Accuracy Comparison:")
    print(f"  Adaptive: {adaptive_dir_acc:.2%}")
    print(f"  Standard: {standard_dir_acc:.2%}")
    print(f"  Improvement: {dir_improvement:.2f} percentage points")
    
    print(f"\nFinal adaptive weights:")
    for name, weight in adaptive_loss.weights.items():
        print(f"  {name}: {weight:.4f}")

if __name__ == "__main__":
    main()
