# SNI v0.3 - Statistical-Neural Interaction for Missing Data Imputation
# Upgraded: categorical balance, lambda ablation, convergence monitoring

from .imputer import SNIImputer, SNIConfig

__version__ = "0.3.0"
__all__ = ["SNIImputer", "SNIConfig"]
