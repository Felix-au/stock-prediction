def is_nse_stock(stock_code):
    """
    Checks if a stock code belongs to the National Stock Exchange (NSE) of India.
    NSE stocks usually end with '.NS'.
    """
    return stock_code.upper().endswith('.NS')
