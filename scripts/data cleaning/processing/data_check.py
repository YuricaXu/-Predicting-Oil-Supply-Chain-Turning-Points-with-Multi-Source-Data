import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data():
    # Load the merged data
    data = pd.read_csv('data/processed_data.csv')
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # 1. Check missing values
    print("\nMissing values in each column:")
    print(data.isnull().sum())
    
    # 2. Check unique values in categorical columns
    print("\nUnique values in categorical columns:")
    print("Energy types:", data['energy'].unique())
    print("Country codes:", data['country'].unique())
    print("Code types:", data['code'].unique())
    
    # 3. Check zero values in supply
    zero_supply = (data['value_supply'] == 0).sum()
    print(f"\nNumber of zero values in supply: {zero_supply} ({zero_supply/len(data)*100:.2f}%)")
    
    # 4. Analyze supply by country and code
    print("\nSupply statistics by country:")
    print(data.groupby('country')['value_supply'].describe())
    
    # 5. Check for outliers
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=data['value_supply'])
    plt.title('Supply Distribution')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data['value_wti'])
    plt.title('WTI Price Distribution')
    plt.tight_layout()
    plt.savefig('data/plots/outliers.png')
    plt.close()
    
    # 6. Time series completeness
    print("\nChecking for gaps in time series...")
    data = data.sort_values('date')
    date_diff = data['date'].diff().value_counts()
    print("\nUnique time differences between consecutive records (in days):")
    print(date_diff)

if __name__ == "__main__":
    analyze_data() 