"""
Data Collection Script for Oil Supply Prediction Project

This script collects essential data for the oil supply prediction model:
1. Industrial Production Index (as a proxy for global oil demand)
2. Additional economic indicators if needed

The data is collected using pandas_datareader from FRED database.
"""

import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime

def collect_essential_data(start_date='2002-01-01', end_date='2024-12-31'):
    """
    Collect essential economic indicators for oil supply prediction
    
    Parameters:
    - start_date: Start date for data collection (default: '2002-01-01')
    - end_date: End date for data collection (default: '2024-12-31')
    
    Returns:
    - DataFrame containing collected data
    """
    # Collect Industrial Production Index as proxy for demand
    df = pdr.get_data_fred('INDPRO', start=start_date, end=end_date)
    df.columns = ['industrial_production']
    
    # Resample to monthly frequency if needed
    df = df.resample('M').last()
    
    # Save to CSV
    df.to_csv('data/raw/demand_proxy.csv')
    print(f"Data saved with {len(df)} records")
    
    return df

if __name__ == '__main__':
    collect_essential_data() 