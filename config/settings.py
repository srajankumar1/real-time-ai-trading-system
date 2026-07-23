import numpy as np

# Random State Control for System Reproducibility
RANDOM_SEED = 42

# Asset Registry Framework (Swapped Indices for Forex Pairs)
TICKERS = ["^GSPC", "INRUSD=X", "USDEUR=X", "BTC-USD", "GC=F", "SI=F", "CL=F"]

# Feature Extraction Window Adjustments
SMA_WINDOWS = [5, 10, 20]
EMA_WINDOWS = [10, 20]
RSI_WINDOW = 14
ROLLING_WINDOWS = [3, 5, 10]

# Rule-Based Signal Target Threshold Configuration
RETURN_THRESHOLD = 0.01  # +/- 1% daily return boundary condition

# Realistic Trading Frictions & Slippage Controls
TRANSACTION_FEE = 0.001  # 0.1% transaction fee deduction per turn
SLIPPAGE = 0.0005        # 0.05% price execution latency penalty