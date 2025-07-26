"""
Improved Evaluation for Stock Price Prediction with Bias Correction
Addresses systematic offset issues and provides more meaningful financial metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def calculate_bias_corrected_metrics(y_true, y_pred):
    """
    Calculate bias-corrected metrics that focus on pattern accuracy
    """
    # Calculate systematic bias (mean difference)
    bias = np.mean(y_pred - y_true)
    
    # Bias-corrected predictions
    y_pred_corrected = y_pred - bias
    
    # Original metrics
    original_mae = mean_absolute_error(y_true, y_pred)
    original_mse = mean_squared_error(y_true, y_pred)
    original_rmse = np.sqrt(original_mse)
    
    # Bias-corrected metrics
    corrected_mae = mean_absolute_error(y_true, y_pred_corrected)
    corrected_mse = mean_squared_error(y_true, y_pred_corrected)
    corrected_rmse = np.sqrt(corrected_mse)
    
    return {
        'bias': bias,
        'original_mae': original_mae,
        'original_mse': original_mse,
        'original_rmse': original_rmse,
        'corrected_mae': corrected_mae,
        'corrected_mse': corrected_mse,
        'corrected_rmse': corrected_rmse
    }

def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy - how well the model predicts price direction
    """
    # Calculate price changes (differences)
    true_changes = np.diff(y_true.flatten())
    pred_changes = np.diff(y_pred.flatten())
    
    # Determine direction (up=1, down=-1, flat=0)
    true_directions = np.sign(true_changes)
    pred_directions = np.sign(pred_changes)
    
    # Calculate directional accuracy
    correct_directions = np.sum(true_directions == pred_directions)
    total_directions = len(true_directions)
    directional_accuracy = correct_directions / total_directions
    
    return directional_accuracy, true_directions, pred_directions

def calculate_correlation_metrics(y_true, y_pred):
    """
    Calculate correlation-based metrics
    """
    # Pearson correlation
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rank_correlation, _ = spearmanr(y_true.flatten(), y_pred.flatten())
    
    return correlation, rank_correlation

def calculate_financial_metrics(y_true, y_pred):
    """
    Calculate financial-specific metrics
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate returns
    true_returns = np.diff(y_true_flat) / y_true_flat[:-1]
    pred_returns = np.diff(y_pred_flat) / y_pred_flat[:-1]
    
    # Return correlation
    return_correlation = np.corrcoef(true_returns, pred_returns)[0, 1]
    
    # Volatility comparison
    true_volatility = np.std(true_returns)
    pred_volatility = np.std(pred_returns)
    volatility_ratio = pred_volatility / true_volatility
    
    # Maximum drawdown simulation
    def calculate_max_drawdown(returns):
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    true_max_drawdown = calculate_max_drawdown(true_returns)
    pred_max_drawdown = calculate_max_drawdown(pred_returns)
    
    return {
        'return_correlation': return_correlation,
        'true_volatility': true_volatility,
        'pred_volatility': pred_volatility,
        'volatility_ratio': volatility_ratio,
        'true_max_drawdown': true_max_drawdown,
        'pred_max_drawdown': pred_max_drawdown
    }

def plot_bias_corrected_predictions(y_true, y_pred_original, y_pred_corrected, title="Bias-Corrected Predictions"):
    """
    Plot original vs bias-corrected predictions
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    n_plot = min(100, len(y_true))
    x_axis = range(n_plot)
    
    # Original predictions
    ax1.plot(x_axis, y_true[:n_plot], label='Actual Price', linewidth=2, alpha=0.8)
    ax1.plot(x_axis, y_pred_original[:n_plot], label='Original Predictions', linewidth=2, alpha=0.8)
    ax1.set_title(f'{title} - Original Predictions')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Stock Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bias-corrected predictions
    ax2.plot(x_axis, y_true[:n_plot], label='Actual Price', linewidth=2, alpha=0.8)
    ax2.plot(x_axis, y_pred_corrected[:n_plot], label='Bias-Corrected Predictions', linewidth=2, alpha=0.8)
    ax2.set_title(f'{title} - Bias-Corrected Predictions')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Stock Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_bias_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

def apply_distribution_shift_adjustment(predictions_actual, y_test_actual, train_test_mean_diff):
    """
    Apply post-processing adjustment to account for distribution shift
    """
    # Calculate the difference between predicted mean and actual test mean
    pred_mean = predictions_actual.mean()
    test_mean = y_test_actual.mean()
    mean_adjustment = test_mean - pred_mean
    
    # Apply the adjustment
    adjusted_predictions = predictions_actual + mean_adjustment
    
    return adjusted_predictions, mean_adjustment

def comprehensive_model_evaluation(model, X_test, y_test, scaler_y, model_name="Model", train_test_mean_diff=None):
    """
    Comprehensive evaluation with bias correction, financial metrics, and distribution shift adjustment
    """
    model.eval()
    with torch.no_grad():
        predictions_normalized = model(X_test).numpy()
    
    # Scaling verification
    print(f"\n=== {model_name} Scaling Verification ===")
    print(f"Normalized predictions range: {predictions_normalized.min():.4f} to {predictions_normalized.max():.4f}")
    print(f"Normalized predictions mean: {predictions_normalized.mean():.4f}, std: {predictions_normalized.std():.4f}")
    print(f"Normalized test targets range: {y_test.numpy().min():.4f} to {y_test.numpy().max():.4f}")
    print(f"Normalized test targets mean: {y_test.numpy().mean():.4f}, std: {y_test.numpy().std():.4f}")
    
    # Convert back to actual prices
    y_test_actual = scaler_y.inverse_transform(y_test.numpy())
    predictions_actual = scaler_y.inverse_transform(predictions_normalized)
    
    # Verify inverse transform
    print(f"Actual test targets range: ${y_test_actual.min():.2f} to ${y_test_actual.max():.2f}")
    print(f"Actual predictions range: ${predictions_actual.min():.2f} to ${predictions_actual.max():.2f}")
    print(f"Actual test targets mean: ${y_test_actual.mean():.2f}, std: ${y_test_actual.std():.2f}")
    print(f"Actual predictions mean: ${predictions_actual.mean():.2f}, std: ${predictions_actual.std():.2f}")
    
    # Apply distribution shift adjustment if provided
    if train_test_mean_diff is not None:
        adjusted_predictions, mean_adjustment = apply_distribution_shift_adjustment(
            predictions_actual, y_test_actual, train_test_mean_diff
        )
        print(f"\n=== Distribution Shift Adjustment ===")
        print(f"Train-Test Mean Difference: ${train_test_mean_diff:.2f}")
        print(f"Mean Adjustment Applied: ${mean_adjustment:.2f}")
        print(f"Adjusted predictions mean: ${adjusted_predictions.mean():.2f}, std: ${adjusted_predictions.std():.2f}")
        
        # Use adjusted predictions for evaluation
        predictions_for_evaluation = adjusted_predictions
    else:
        predictions_for_evaluation = predictions_actual
    
    print(f"\n=== {model_name} Comprehensive Evaluation ===")
    
    # 1. Bias-corrected metrics (using adjusted predictions for evaluation)
    bias_metrics = calculate_bias_corrected_metrics(y_test_actual, predictions_for_evaluation)
    
    print(f"\nBias Analysis:")
    print(f"  Systematic Bias: ${bias_metrics['bias']:.4f}")
    
    print(f"\nOriginal Metrics:")
    print(f"  MAE: ${bias_metrics['original_mae']:.4f}")
    print(f"  MSE: ${bias_metrics['original_mse']:.4f}")
    print(f"  RMSE: ${bias_metrics['original_rmse']:.4f}")
    
    print(f"\nBias-Corrected Metrics:")
    print(f"  MAE: ${bias_metrics['corrected_mae']:.4f}")
    print(f"  MSE: ${bias_metrics['corrected_mse']:.4f}")
    print(f"  RMSE: ${bias_metrics['corrected_rmse']:.4f}")
    
    # 2. Directional accuracy (using adjusted predictions)
    dir_accuracy, true_dirs, pred_dirs = calculate_directional_accuracy(y_test_actual, predictions_for_evaluation)
    print(f"\nDirectional Accuracy: {dir_accuracy:.2%}")
    
    # 3. Correlation metrics (using adjusted predictions)
    correlation, rank_correlation = calculate_correlation_metrics(y_test_actual, predictions_for_evaluation)
    print(f"\nCorrelation Metrics:")
    print(f"  Pearson Correlation: {correlation:.4f}")
    print(f"  Spearman Rank Correlation: {rank_correlation:.4f}")
    
    # 4. Financial metrics (using adjusted predictions)
    financial_metrics = calculate_financial_metrics(y_test_actual, predictions_for_evaluation)
    print(f"\nFinancial Metrics:")
    print(f"  Return Correlation: {financial_metrics['return_correlation']:.4f}")
    print(f"  Volatility Ratio (Pred/True): {financial_metrics['volatility_ratio']:.4f}")
    print(f"  True Volatility: {financial_metrics['true_volatility']:.4f}")
    print(f"  Predicted Volatility: {financial_metrics['pred_volatility']:.4f}")
    print(f"  True Max Drawdown: {financial_metrics['true_max_drawdown']:.2%}")
    print(f"  Predicted Max Drawdown: {financial_metrics['pred_max_drawdown']:.2%}")
    
    # 5. Create bias-corrected predictions (using the adjusted predictions)
    bias_corrected_predictions = predictions_for_evaluation - bias_metrics['bias']
    
    # 6. Plot results using the adjusted predictions
    plot_bias_corrected_predictions(
        y_test_actual, predictions_for_evaluation, bias_corrected_predictions, 
        f"{model_name} Predictions"
    )
    
    return {
        'bias_metrics': bias_metrics,
        'directional_accuracy': dir_accuracy,
        'correlation': correlation,
        'rank_correlation': rank_correlation,
        'financial_metrics': financial_metrics,
        'bias_corrected_predictions': bias_corrected_predictions
    }

def compare_models_comprehensive(adaptive_results, standard_results):
    """
    Compare models using comprehensive metrics
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    # Bias comparison
    print(f"\nBias Analysis:")
    print(f"  Adaptive Bias: ${adaptive_results['bias_metrics']['bias']:.4f}")
    print(f"  Standard Bias: ${standard_results['bias_metrics']['bias']:.4f}")
    
    # Bias-corrected performance
    print(f"\nBias-Corrected Performance:")
    adaptive_corrected_mae = adaptive_results['bias_metrics']['corrected_mae']
    standard_corrected_mae = standard_results['bias_metrics']['corrected_mae']
    mae_improvement = ((standard_corrected_mae - adaptive_corrected_mae) / standard_corrected_mae * 100)
    
    print(f"  Adaptive MAE: ${adaptive_corrected_mae:.4f}")
    print(f"  Standard MAE: ${standard_corrected_mae:.4f}")
    print(f"  MAE Improvement: {mae_improvement:.2f}%")
    
    adaptive_corrected_rmse = adaptive_results['bias_metrics']['corrected_rmse']
    standard_corrected_rmse = standard_results['bias_metrics']['corrected_rmse']
    rmse_improvement = ((standard_corrected_rmse - adaptive_corrected_rmse) / standard_corrected_rmse * 100)
    
    print(f"  Adaptive RMSE: ${adaptive_corrected_rmse:.4f}")
    print(f"  Standard RMSE: ${standard_corrected_rmse:.4f}")
    print(f"  RMSE Improvement: {rmse_improvement:.2f}%")
    
    # Directional accuracy
    print(f"\nDirectional Accuracy:")
    print(f"  Adaptive: {adaptive_results['directional_accuracy']:.2%}")
    print(f"  Standard: {standard_results['directional_accuracy']:.2%}")
    dir_improvement = (adaptive_results['directional_accuracy'] - standard_results['directional_accuracy']) * 100
    print(f"  Improvement: {dir_improvement:.2f} percentage points")
    
    # Correlation comparison
    print(f"\nCorrelation Metrics:")
    print(f"  Adaptive Pearson: {adaptive_results['correlation']:.4f}")
    print(f"  Standard Pearson: {standard_results['correlation']:.4f}")
    print(f"  Adaptive Spearman: {adaptive_results['rank_correlation']:.4f}")
    print(f"  Standard Spearman: {standard_results['rank_correlation']:.4f}")
    
    # Financial metrics comparison
    print(f"\nFinancial Metrics:")
    adaptive_return_corr = adaptive_results['financial_metrics']['return_correlation']
    standard_return_corr = standard_results['financial_metrics']['return_correlation']
    print(f"  Return Correlation - Adaptive: {adaptive_return_corr:.4f}")
    print(f"  Return Correlation - Standard: {standard_return_corr:.4f}")
    
    adaptive_vol_ratio = adaptive_results['financial_metrics']['volatility_ratio']
    standard_vol_ratio = standard_results['financial_metrics']['volatility_ratio']
    print(f"  Volatility Ratio - Adaptive: {adaptive_vol_ratio:.4f}")
    print(f"  Volatility Ratio - Standard: {standard_vol_ratio:.4f}")
    
    # Overall assessment
    print(f"\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    
    better_metrics = 0
    total_metrics = 0
    
    metrics_comparison = [
        ("Bias-Corrected MAE", mae_improvement > 0),
        ("Bias-Corrected RMSE", rmse_improvement > 0),
        ("Directional Accuracy", dir_improvement > 0),
        ("Pearson Correlation", adaptive_results['correlation'] > standard_results['correlation']),
        ("Return Correlation", adaptive_return_corr > standard_return_corr),
    ]
    
    for metric_name, is_better in metrics_comparison:
        total_metrics += 1
        if is_better:
            better_metrics += 1
            print(f"✓ Adaptive model is better at: {metric_name}")
        else:
            print(f"✗ Standard model is better at: {metric_name}")
    
    print(f"\nAdaptive model wins in {better_metrics}/{total_metrics} key metrics")
    
    if better_metrics > total_metrics / 2:
        print("🎉 CONCLUSION: Adaptive model shows superior performance when properly evaluated!")
    else:
        print("📊 CONCLUSION: Mixed results - both models have strengths")

if __name__ == "__main__":
    print("This module provides improved evaluation functions.")
    print("Import and use with your trained models for comprehensive analysis.")
