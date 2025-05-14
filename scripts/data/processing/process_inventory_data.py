import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_eia_data(file_path='data/raw/Weekly_U.S._Ending_Stocks_excluding_SPR_of_Crude_Oil.csv'):
    """
    Load and process EIA weekly inventory data
    """
    # Skip the first 4 lines of metadata
    df = pd.read_csv(file_path, skiprows=4)
    
    # Rename columns
    df.columns = ['date', 'inventory']
    
    # Convert date format
    df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index
    df = df.set_index('date')
    
    # Sort by date ascending
    df = df.sort_index()
    
    return df

def load_jodi_data(file_path='data/raw/QDL-JODI.csv'):
    """
    Load and process JODI data
    """
    df = pd.read_csv(file_path)
    
    # Keep only oil-related data
    df = df[df['energy'] == 'OIL']
    
    # Convert date to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by country and date to calculate total inventory change
    inventory_change = df.groupby(['date', 'country'])['value'].sum().reset_index()
    
    # Pivot table to get each country as a separate column
    inventory_wide = inventory_change.pivot(index='date', columns='country', values='value')
    
    # Calculate global total inventory change
    inventory_wide['global_change'] = inventory_wide.sum(axis=1)
    
    return inventory_wide

def calculate_features(eia_data, jodi_data):
    """
    Calculate features for prediction
    """
    # Convert EIA weekly data to monthly to match JODI
    eia_monthly = eia_data.resample('M').last()
    
    # Calculate EIA features
    eia_features = pd.DataFrame(index=eia_monthly.index)
    
    # 1. Inventory level
    eia_features['us_inventory_level'] = eia_monthly['inventory']
    
    # 2. Month-over-month inventory change rate
    eia_features['us_inventory_mom_change'] = eia_monthly['inventory'].pct_change()
    
    # 3. 5-year percentile of inventory level
    rolling_window = 260  # Number of weeks in 5 years (52 weeks * 5 years)
    eia_features['us_inventory_5y_percentile'] = (
        eia_data['inventory']
        .rolling(rolling_window)
        .apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x))
    )
    
    # 4. Inventory velocity (first derivative)
    eia_features['us_inventory_velocity'] = eia_data['inventory'].diff()
    
    # 5. Inventory acceleration (second derivative)
    eia_features['us_inventory_acceleration'] = eia_data['inventory'].diff().diff()
    
    # Merge JODI features
    # Ensure date alignment
    jodi_aligned = jodi_data.reindex(eia_features.index)
    
    # 6. Global inventory change
    eia_features['global_inventory_change'] = jodi_aligned['global_change']
    
    # 7. Major region inventory changes
    for country in ['CHN', 'EU', 'JPN', 'KOR']:  # Major consumer countries
        if country in jodi_data.columns:
            eia_features[f'{country.lower()}_inventory_change'] = jodi_aligned[country]
    
    return eia_features

def main():
    # Load data
    print("Loading EIA data...")
    eia_data = load_eia_data()
    print("EIA data loaded, time range:", eia_data.index.min(), "to", eia_data.index.max())
    
    print("\nLoading JODI data...")
    jodi_data = load_jodi_data()
    print("JODI data loaded, time range:", jodi_data.index.min(), "to", jodi_data.index.max())
    
    # Calculate features
    print("\nCalculating features...")
    features = calculate_features(eia_data, jodi_data)
    
    # Save processed data
    output_path = os.path.join('data', 'processed', 'inventory_features.csv')
    features.to_csv(output_path)
    print(f"\nFeature data saved to: {output_path}")
    
    # Print data preview
    print("\nFeature data preview:")
    print(features.head())
    print("\nFeature statistics:")
    print(features.describe())
    
    # Print missing value info
    print("\nMissing value statistics:")
    print(features.isnull().sum())

if __name__ == "__main__":
    main() 