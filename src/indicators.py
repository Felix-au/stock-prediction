import pandas as pd

def calculate_technical_indicators(df):
    """
    Calculates RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence)
    for the given stock history DataFrame.
    """
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        raise ValueError("Input must be a DataFrame with a 'Close' column")
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def calculate_sma(stock_data, short_window=20, long_window=100):
    """
    Calculates Short and Long term Simple Moving Averages (SMA).
    """
    short_sma = stock_data['Close'].rolling(window=short_window).mean()
    long_sma = stock_data['Close'].rolling(window=long_window).mean()
    return short_sma, long_sma
