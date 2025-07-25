# Adaptive Loss Function Research

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements an **Adaptive Loss Function for Gradient Descent** that dynamically adjusts correlation-based weights during training to improve convergence speed and model performance. The approach combines traditional Mean Squared Error (MSE) with weighted correlation terms that adapt based on feature-target covariances.

## 🎯 Key Features

- **Dynamic Weight Adaptation**: Automatically adjusts loss function weights every 5 epochs based on feature-target correlations
- **Correlation-Aware Training**: Incorporates feature-target covariance information into the loss function
- **Real-World Dataset**: Validated on the Global Superstore retail dataset (51,290+ samples)
- **Comprehensive Analysis**: Includes multicollinearity detection, convergence analysis, and visualization
- **Performance Improvement**: Demonstrates measurable improvements over standard MSE loss

## 📊 Results Summary

- **Dataset Size**: 51,290 retail transactions
- **Features**: 8 engineered features (quantity, discount, shipping cost, category, etc.)
- **Performance Gain**: 0.5% improvement in MSE over standard gradient descent
- **Convergence**: Adaptive method shows consistent weight evolution and stable training

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Experiment

```bash
python3 adaptive_loss_gradient_descent.py
```

This will:
1. Load and preprocess the Global Superstore dataset
2. Train both adaptive and standard models
3. Generate comparison visualizations
4. Output detailed performance analysis

## 📁 Repository Structure

```
adaptive-loss-function-research/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── adaptive_loss_gradient_descent.py   # Main implementation
├── demo_adaptive_loss.py              # Demo script
├── Global Superstore (1).csv          # Dataset
├── docs/
│   └── whitepaper.md                  # Technical white paper
├── results/
│   └── training_comparison.png        # Generated visualizations
└── LICENSE                            # MIT License
```

## 🔬 Technical Approach

### Adaptive Loss Function

The adaptive loss function combines MSE with weighted correlation terms:

```
L_adaptive = L_MSE + Σ(w_i × |cov(f_i, y)|)
```

Where:
- `L_MSE` is the standard Mean Squared Error
- `w_i` are adaptive weights for each feature
- `cov(f_i, y)` is the covariance between feature `i` and target `y`

### Weight Update Mechanism

Weights are updated every 5 epochs using:

```
w_new = w_old × (1 + 0.1 × normalized_covariance)
```

With constraints: `w ∈ [0.01, 0.5]` for stability.

## 📈 Performance Metrics

| Metric | Adaptive Loss | Standard MSE | Improvement |
|--------|---------------|--------------|-------------|
| Test MSE | 64,308.72 | 64,604.57 | 0.5% |
| Convergence | 3 epochs | 3 epochs | Equal |
| Stability | High | High | Equal |

## 🔧 Implementation Details

### Key Components

1. **AdaptiveLossFunction**: PyTorch module implementing the adaptive loss
2. **RetailSalesPredictor**: Neural network for sales prediction
3. **Data Preprocessing**: Categorical encoding and feature engineering
4. **Visualization**: Comprehensive training comparison plots

### Features Used

- **Numerical**: Quantity, Discount, Shipping Cost
- **Categorical**: Category, Sub-Category, Segment, Region, Market

## 📚 Documentation

- [Technical White Paper](docs/whitepaper.md) - Detailed methodology and theoretical background
- [API Documentation](docs/api.md) - Code documentation and usage examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Global Superstore dataset for providing real-world retail data
- PyTorch team for the excellent deep learning framework
- Research community for foundational work on adaptive optimization

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

---

**Note**: This implementation is for research and educational purposes. Results may vary based on dataset characteristics and hyperparameter settings.
