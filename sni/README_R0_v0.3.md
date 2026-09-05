# SNI v0.3 - Statistical-Neural Interaction for Missing Data Imputation

This file describes SNI v0.3, the final version used in the paper.

## Overview

SNI (Statistical-Neural Interaction) is a hybrid missing data imputation method that combines statistical priors with neural attention mechanisms. It achieves state-of-the-art performance on mixed-type tabular data by leveraging both the interpretability of statistical methods and the flexibility of deep learning.

## v0.3 Enhancements

1. **Categorical Balance Mode**: Improved handling of imbalanced categorical features via class-weighted sampling and balanced focal loss
2. **Learnable Per-Head Lambda**: Each attention head learns its own prior-confidence parameter, enabling automatic balance between data-driven learning and statistical priors
3. **Convergence Monitoring**: Runtime tracking of EM convergence, lambda trajectories, and early-stopping diagnostics

## Key Innovations

### 1. Controllable-Prior Feature Attention (CPFA)
- Multi-head attention over feature tokens
- **Learnable per-head confidence parameters** that automatically balance data-driven learning with statistical priors
- Prior strength decays across EM iterations

### 2. Statistical-Neural Integration
- **Statistical priors** from correlation analysis guide initial attention patterns
- **Neural attention** learns refined feature dependencies from data

### 3. Multi-Scale Attention for Categorical Features
- **Local heads**: Focus on within-category relationships
- **Global heads**: Capture cross-category dependencies
- Learned fusion of multi-scale representations

### 4. Dual-Path Architecture (for categorical targets)
- Path 1: Attention-based classification
- Path 2: Continuous embedding path
- Learned gating combines both paths

## Quick Start

```python
import pandas as pd
from SNI_v0_3 import SNIImputer, SNIConfig

# Load your data
X_missing = pd.read_csv("data_with_missing.csv")
X_complete = pd.read_csv("data_complete.csv")  # Optional, for evaluation

# Define variable types
categorical_vars = ["cat1", "cat2"]
continuous_vars = ["cont1", "cont2", "cont3"]

# Create imputer with default config
config = SNIConfig(use_gpu=True, seed=42)

imputer = SNIImputer(
    categorical_vars=categorical_vars,
    continuous_vars=continuous_vars,
    config=config,
)

# Run imputation
X_imputed = imputer.impute(X_missing=X_missing, X_complete=X_complete)

# Access learned dependencies
dependency_matrix = imputer.compute_dependency_matrix()
print(dependency_matrix)
```

## Configuration

### SNIConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha0` | 1.0 | Initial prior strength |
| `gamma` | 0.9 | Prior decay rate per iteration |
| `max_iters` | 3 | Number of EM iterations |
| `hidden_dims` | (256, 128, 64) | MLP hidden layer dimensions |
| `emb_dim` | 128 | Feature embedding dimension |
| `num_heads` | 16 | Number of attention heads |
| `lr` | 2e-4 | Learning rate |
| `epochs` | 200 | Maximum training epochs |
| `batch_size` | 128 | Batch size |
| `early_stopping_patience` | 20 | Early stopping patience |
| `variant` | "SNI" | {SNI, NoPrior, HardPrior, SNI-M} |

### Variants

| Variant | Description |
|---------|-------------|
| `SNI` | Full model with learnable lambda |
| `NoPrior` | Ablation: no statistical prior |
| `HardPrior` | Ablation: fixed lambda (hard prior) |
| `SNI-M` | SNI with missingness-aware embeddings |
| `SNI+KNN` | SNI with KNN post-processing |

## Output Artifacts

```python
# Imputed data
X_imputed = imputer.impute(X_missing)

# Dependency matrix D[i,j] = importance of feature j for predicting feature i
D = imputer.compute_dependency_matrix()

# Edge list for network visualization
edges = imputer.export_dependency_network_edges(tau=0.15)

# Lambda traces over training
lambda_traces = imputer.lambda_trace_per_head
```

## File Structure

```
SNI_v0_3/
├── __init__.py          # Package exports
├── imputer.py           # Main SNIImputer class with SNIConfig
├── cpfa.py              # CPFA neural network modules
├── losses.py            # Custom loss functions
├── dataio.py            # Data I/O utilities
├── metrics.py           # Evaluation metrics
└── utils.py             # Utility functions
```

## License

MIT License - see LICENSE file for details.
