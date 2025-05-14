import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os

# Set style for better visualization
sns.set_style("whitegrid")  # Using seaborn's whitegrid style instead

# 1) Load datasets
def load_data():
    # Load WTI data
    wti_data = pd.read_csv('wti_monthly_prices.csv')
    # Load JODI data from data folder
    jodi_data = pd.read_csv('data/QDL-JODI.csv')
    return wti_data, jodi_data

# 2) Preprocess data
def preprocess_data(wti_data, jodi_data):
    # Convert date strings to datetime
    wti_data['date'] = pd.to_datetime(wti_data['date'], format='%d/%m/%Y')
    jodi_data['date'] = pd.to_datetime(jodi_data['date'])  # YYYY-MM-DD format
    
    print("\nDate ranges before merge:")
    print(f"WTI data: from {wti_data['date'].min()} to {wti_data['date'].max()}")
    print(f"JODI data: from {jodi_data['date'].min()} to {jodi_data['date'].max()}")
    
    # Merge datasets on date
    merged_data = pd.merge(jodi_data, wti_data[['date', 'value']], 
                          on='date', 
                          how='left',
                          suffixes=('_supply', '_wti'))
    
    print("\nShape of merged data:", merged_data.shape)
    
    # Handle missing values
    merged_data = merged_data.fillna(method='ffill')  # Forward fill
    
    # Sort by date
    merged_data = merged_data.sort_values('date')
    
    return merged_data

# 3) Exploratory Data Analysis
def perform_eda(data):
    # Create output directory for plots
    os.makedirs('data/plots', exist_ok=True)
    
    # Time series plot
    plt.figure(figsize=(15, 6))
    plt.plot(data['date'], data['value_wti'], label='WTI Price')
    plt.plot(data['date'], data['value_supply'], label='Supply')
    plt.title('WTI Price vs Supply Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/plots/time_series.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('data/plots/correlation_heatmap.png')
    plt.close()
    
    # Distribution plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data['value_wti'], kde=True)
    plt.title('WTI Price Distribution')
    plt.subplot(1, 2, 2)
    sns.histplot(data['value_supply'], kde=True)
    plt.title('Supply Distribution')
    plt.tight_layout()
    plt.savefig('data/plots/distributions.png')
    plt.close()
    
    return data.describe()

def main():
    print("Loading data...")
    wti_data, jodi_data = load_data()
    
    print("\nPreprocessing data...")
    merged_data = preprocess_data(wti_data, jodi_data)
    
    print("\nPerforming exploratory data analysis...")
    summary_stats = perform_eda(merged_data)
    
    print("\nSummary Statistics:")
    print(summary_stats)
    
    print("\nAnalysis complete! Check the 'data/plots' directory for visualizations.")
    
    # Save processed dataset
    merged_data.to_csv('data/processed_data.csv', index=False)
    print("\nProcessed data saved to 'data/processed_data.csv'")

if __name__ == "__main__":
    main() 