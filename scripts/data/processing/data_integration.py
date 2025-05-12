import pandas as pd
import numpy as np
from datetime import datetime

def load_and_process_jodi_data(file_path='data/QDL-JODI.csv'):
    """
    Load and process JODI data
    """
    print("Loading JODI data...")
    df_jodi = pd.read_csv(file_path)
    
    # Convert date to datetime (YYYY-MM-DD format)
    df_jodi['date'] = pd.to_datetime(df_jodi['date'])
    
    # Pivot the data to create separate columns for different measurements
    df_pivot = df_jodi.pivot_table(
        index=['date', 'country'],
        columns=['energy', 'code'],
        values='value',
        aggfunc='first'
    ).reset_index()
    
    return df_pivot

def load_and_process_wti_data(file_path='data/wti_monthly_prices.csv'):
    """
    Load and process WTI price data
    """
    print("Loading WTI data...")
    df_wti = pd.read_csv(file_path)
    
    # Convert date to datetime (DD/MM/YYYY format)
    df_wti['date'] = pd.to_datetime(df_wti['date'], format='%d/%m/%Y')
    
    # Create a pivot table for WTI data
    df_pivot = df_wti.pivot_table(
        index='date',
        columns=['energy', 'code'],
        values='value',
        aggfunc='first'
    ).reset_index()
    
    return df_pivot

def integrate_data():
    """
    Integrate JODI and WTI data
    """
    # Load data
    df_jodi = load_and_process_jodi_data()
    df_wti = load_and_process_wti_data()
    
    print("\nIntegrating datasets...")
    # Merge JODI and WTI data
    df_integrated = pd.merge(
        df_jodi,
        df_wti,
        on='date',
        how='left'
    )
    
    # Sort by date and country
    df_integrated.sort_values(['date', 'country'], inplace=True)
    
    # Save the integrated dataset
    output_path = 'data/integrated_data.csv'
    df_integrated.to_csv(output_path, index=False)
    print(f"\nIntegrated data saved to {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Time range: {df_integrated['date'].min()} to {df_integrated['date'].max()}")
    print(f"Number of countries: {df_integrated['country'].nunique()}")
    print(f"Number of records: {len(df_integrated)}")
    print("\nColumns in integrated dataset:")
    print(df_integrated.columns.tolist())
    
    return df_integrated

if __name__ == "__main__":
    integrate_data() 