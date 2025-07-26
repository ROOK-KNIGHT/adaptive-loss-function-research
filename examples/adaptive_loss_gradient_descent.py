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
    based on feature-target covariances, as described in the research paper.
    """
    
    def __init__(self, feature_names: List[str], initial_weights: Dict[str, float] = None):
        super(AdaptiveLossFunction, self).__init__()
        self.feature_names = feature_names
        self.mse_loss = nn.MSELoss()
        
        # Initialize weights for correlation terms
        if initial_weights is None:
            # Default weights as suggested in the paper
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
        as described in the paper
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
                # Using a simple sigmoid-like normalization
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

class RetailSalesPredictor(nn.Module):
    """Simple neural network for sales prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super(RetailSalesPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def load_global_superstore_data() -> pd.DataFrame:
    """
    Load and preprocess the Global Superstore dataset
    """
    # Load from CSV file
    df = pd.read_csv('Global Superstore (1).csv')
    print("Loaded data from CSV file")
    
    print(f"Original dataset shape: {df.shape}")
    
    # Clean and preprocess the data
    # Remove rows with missing sales data
    df = df.dropna(subset=['Sales'])
    
    # Convert categorical variables to numerical
    # Category encoding
    category_map = {'Technology': 2, 'Furniture': 1, 'Office Supplies': 0}
    df['category_encoded'] = df['Category'].map(category_map)
    
    # Sub-Category encoding (simplified - take top categories)
    subcategory_counts = df['Sub-Category'].value_counts()
    top_subcategories = subcategory_counts.head(10).index.tolist()
    subcategory_map = {cat: i for i, cat in enumerate(top_subcategories)}
    df['subcategory_encoded'] = df['Sub-Category'].map(subcategory_map).fillna(-1)
    
    # Segment encoding
    segment_map = {'Corporate': 2, 'Consumer': 1, 'Home Office': 0}
    df['segment_encoded'] = df['Segment'].map(segment_map)
    
    # Region encoding
    region_map = {'Central': 0, 'East': 1, 'West': 2, 'South': 3, 
                  'APAC': 4, 'EU': 5, 'EMEA': 6, 'LATAM': 7, 'Africa': 8, 'Oceania': 9}
    df['region_encoded'] = df['Region'].map(region_map).fillna(-1)
    
    # Market encoding
    market_map = {'US': 0, 'APAC': 1, 'EU': 2, 'LATAM': 3, 'Africa': 4}
    df['market_encoded'] = df['Market'].map(market_map).fillna(-1)
    
    # Clean numerical columns
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
    df['Shipping Cost'] = pd.to_numeric(df['Shipping Cost'], errors='coerce')
    
    # Remove rows with missing key numerical data
    df = df.dropna(subset=['Sales', 'Quantity', 'Discount', 'Shipping Cost'])
    
    # Create final dataset with selected features
    processed_df = pd.DataFrame({
        'sales': df['Sales'],
        'quantity': df['Quantity'],
        'discount': df['Discount'],
        'shipping_cost': df['Shipping Cost'],
        'category': df['category_encoded'],
        'subcategory': df['subcategory_encoded'],
        'segment': df['segment_encoded'],
        'region': df['region_encoded'],
        'market': df['market_encoded']
    })
    
    # Remove any remaining NaN values
    processed_df = processed_df.dropna()
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Features: {list(processed_df.columns[1:])}")  # Exclude 'sales' target
    
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
            print(f"{model_name} - Epoch {epoch:3d}: Loss = {avg_loss:.4f}, MSE = {avg_mse:.4f}")
    
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
    axes[0, 0].set_title('Training Loss Comparison')
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
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function implementing the complete adaptive loss function experiment
    """
    print("=== Adaptive Loss Function for Gradient Descent ===")
    print("Implementation based on the research paper\n")
    
    # 1. Load Global Superstore dataset
    print("1. Loading Global Superstore dataset...")
    df = load_global_superstore_data()
    print(f"Dataset shape: {df.shape}")
    print("\nDataset preview:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
    
    # 2. Prepare features and target
    feature_columns = ['quantity', 'discount', 'shipping_cost', 'category', 'subcategory', 'segment', 'region', 'market']
    X = df[feature_columns]
    y = df['sales'].values.reshape(-1, 1)
    
    # 3. Check multicollinearity
    print("\n2. Checking multicollinearity...")
    vif_results = check_multicollinearity(X)
    
    # 4. Standardize features
    print("\n3. Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 6. Initialize models and loss functions
    print("\n4. Initializing models and loss functions...")
    
    # Models
    adaptive_model = RetailSalesPredictor(input_size=len(feature_columns))
    standard_model = RetailSalesPredictor(input_size=len(feature_columns))
    
    # Loss functions
    # Select top correlated features for adaptive loss (as suggested in paper)
    correlations = df.corr()['sales'].abs().sort_values(ascending=False)
    top_features = correlations.index[1:5].tolist()  # Exclude 'sales' itself, take top 4
    print(f"Selected features for adaptive loss: {top_features}")
    
    # Map to indices in feature_columns
    selected_indices = [feature_columns.index(feat) for feat in top_features if feat in feature_columns]
    selected_feature_names = [feature_columns[i] for i in selected_indices]
    
    adaptive_loss = AdaptiveLossFunction(
        feature_names=selected_feature_names,
        initial_weights={name: 0.1 if i == 0 else 0.05 for i, name in enumerate(selected_feature_names)}
    )
    standard_loss = nn.MSELoss()
    
    # Optimizers
    adaptive_optimizer = optim.Adam(adaptive_model.parameters(), lr=0.001)
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=0.001)
    
    # 7. Train models
    print("\n5. Training models...")
    epochs = 50
    
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
    print("\n6. Evaluating models...")
    
    adaptive_model.eval()
    standard_model.eval()
    
    with torch.no_grad():
        adaptive_pred = adaptive_model(X_test_tensor)
        standard_pred = standard_model(X_test_tensor)
        
        adaptive_mse = nn.MSELoss()(adaptive_pred, y_test_tensor).item()
        standard_mse = nn.MSELoss()(standard_pred, y_test_tensor).item()
        
        print(f"\nTest Results:")
        print(f"Adaptive Loss MSE: {adaptive_mse:.2f}")
        print(f"Standard MSE: {standard_mse:.2f}")
        print(f"Improvement: {((standard_mse - adaptive_mse) / standard_mse * 100):.1f}%")
    
    # 9. Analyze convergence
    print("\n7. Convergence Analysis:")
    
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
    
    # 10. Create visualizations
    print("\n8. Creating visualizations...")
    plot_training_comparison(adaptive_history, standard_history, adaptive_loss)
    
    # 11. Summary
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Dataset size: {len(df)} samples")
    print(f"Features used: {feature_columns}")
    print(f"Selected correlated features: {selected_feature_names}")
    print(f"Training epochs: {epochs}")
    print(f"\nFinal Training Loss:")
    print(f"  Adaptive: {adaptive_history['loss'][-1]:.4f}")
    print(f"  Standard: {standard_history['loss'][-1]:.4f}")
    print(f"\nFinal Test MSE:")
    print(f"  Adaptive: {adaptive_mse:.2f}")
    print(f"  Standard: {standard_mse:.2f}")
    print(f"  Improvement: {((standard_mse - adaptive_mse) / standard_mse * 100):.1f}%")
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
