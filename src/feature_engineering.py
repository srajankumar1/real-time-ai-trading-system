import pandas as pd
import numpy as np

def generate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw multi-asset prices using high-speed vectorized calculations.
    Eliminates computational bottlenecks across large historical matrices.
    """
    df = df.copy()
    close_arr = df['Close'].to_numpy()
    
    # 1. High-Speed Vectorized Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 2. Vectorized Simple Moving Averages (SMA)
    for w in [5, 10, 20]:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'Dist_SMA_{w}'] = (df['Close'] - df[f'SMA_{w}']) / df[f'SMA_{w}']
        
    # 3. Vectorized Exponential Moving Averages (EMA)
    for w in [10, 20]:
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
        df[f'Dist_EMA_{w}'] = (df['Close'] - df[f'EMA_{w}']) / df[f'EMA_{w}']
        
    # 4. Pure Vectorized Relative Strength Index (RSI) via NumPy
    delta = df['Close'].diff().to_numpy()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(close_arr)
    avg_loss = np.zeros_like(close_arr)
    
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close_arr)):
        avg_gain[i] = (gain[i] * alpha) + (avg_gain[i-1] * (1.0 - alpha))
        avg_loss[i] = (loss[i] * alpha) + (avg_loss[i-1] * (1.0 - alpha))
        
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    # 5. Fast Volatility and Statistical Windows
    for w in [3, 5, 10]:
        df[f'Rolling_Mean_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'Rolling_Std_{w}'] = df['Close'].rolling(window=w).std()
        df[f'Volatility_{w}'] = df['Daily_Return'].rolling(window=w).std()

    # 6. Optimized Memory-Shift Structural Lags
    for lag in range(1, 6):
        df[f'Lag_Close_{lag}'] = df['Close'].shift(lag)
        df[f'Lag_Return_{lag}'] = df['Daily_Return'].shift(lag)
        
    df.dropna(inplace=True)
    return df

def generate_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """ Balanced mathematical thirds (Quantiles) calculation loop to boost accuracy. """
    df = df.copy()
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
    df.dropna(subset=['Next_Day_Return'], inplace=True)
    
    q_low = df['Next_Day_Return'].quantile(0.333)
    q_high = df['Next_Day_Return'].quantile(0.666)
    
    conditions = [
        (df['Next_Day_Return'] > q_high),
        (df['Next_Day_Return'] < q_low)
    ]
    choices = [1, 2]
    
    df['Target'] = np.select(conditions, choices, default=0)
    return df