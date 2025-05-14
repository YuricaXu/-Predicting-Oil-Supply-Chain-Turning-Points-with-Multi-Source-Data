import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_eia_inventory_data(api_key=None):
    """
    Fetch US crude oil inventory data from EIA
    Includes commercial crude oil inventory and Strategic Petroleum Reserve (SPR) data
    """
    if api_key is None:
        api_key = os.getenv('EIA_API_KEY')
        if api_key is None:
            raise ValueError("Please provide the EIA API key")
    
    # EIA API endpoint
    base_url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    
    # Set request parameters
    params = {
        'api_key': api_key,
        'frequency': 'weekly',
        'data[]': ['value'],
        'facets[series][]': [
            'WCESTUS1',  # Commercial crude oil inventory
            'WCSSTUS1'   # Strategic Petroleum Reserve
        ],
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'offset': 0,
        'length': 5000  # Get the latest 5000 records
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['response']['data'])
        
        # Rename columns
        df.columns = ['date', 'series_id', 'value']
        
        # Convert to wide format
        df_wide = df.pivot(index='date', columns='series_id', values='value')
        df_wide.columns = ['commercial_stocks', 'spr']
        
        # Convert date format
        df_wide.index = pd.to_datetime(df_wide.index)
        
        # Sort by date
        df_wide.sort_index(inplace=True)
        
        # Save data
        output_path = os.path.join('data', 'raw', 'eia_inventory.csv')
        df_wide.to_csv(output_path)
        print(f"Data saved to: {output_path}")
        
        return df_wide
        
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching data: {e}")
        return None

if __name__ == "__main__":
    inventory_data = fetch_eia_inventory_data()
    if inventory_data is not None:
        print("\nData preview:")
        print(inventory_data.head())
        print("\nData statistics:")
        print(inventory_data.describe()) 