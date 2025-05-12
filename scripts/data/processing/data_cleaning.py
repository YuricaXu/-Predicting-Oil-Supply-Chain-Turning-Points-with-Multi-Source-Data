import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data():
    # Load the data
    print("Loading data...")
    data = pd.read_csv('data/processed_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    # 1. Remove countries with all zeros
    print("\nAnalyzing countries with all zeros...")
    country_stats = data.groupby('country').agg({
        'value_supply': [
            ('all_zeros', lambda x: (x == 0).all()),
            ('count', 'count')
        ]
    })
    country_stats.columns = ['all_zeros', 'count']
    
    all_zero_countries = country_stats[country_stats['all_zeros']].index.tolist()
    print(f"Countries with all zero values ({len(all_zero_countries)}):")
    print(all_zero_countries)
    
    # Remove these countries
    data_cleaned = data[~data['country'].isin(all_zero_countries)]
    
    # 2. Analyze and handle outliers
    print("\nAnalyzing outliers...")
    def get_outliers(group):
        q1 = group['value_supply'].quantile(0.25)
        q3 = group['value_supply'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        outliers = group[(group['value_supply'] < lower_bound) | 
                        (group['value_supply'] > upper_bound)]
        return outliers
    
    # Find outliers by country
    outliers = data_cleaned.groupby('country', group_keys=False).apply(get_outliers)
    print("\nNumber of outliers by country:")
    print(outliers.groupby('country').size())
    
    # 3. Create visualization of data quality
    plt.figure(figsize=(15, 6))
    
    # Plot data availability by country
    data_availability = data_cleaned.pivot_table(
        values='value_supply',
        index='date',
        columns='country',
        aggfunc='count'
    )
    
    sns.heatmap(data_availability.notna(), cmap='YlOrRd')
    plt.title('Data Availability by Country Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/plots/data_availability.png')
    plt.close()
    
    # 4. Save cleaned dataset with quality indicators
    print("\nPreparing final cleaned dataset...")
    
    # Add quality indicators
    data_cleaned['is_outlier'] = data_cleaned.index.isin(outliers.index)
    data_cleaned['zero_value'] = data_cleaned['value_supply'] == 0
    
    # Save cleaned data
    data_cleaned.to_csv('data/cleaned_data.csv', index=False)
    
    # Print summary statistics
    print("\nCleaning Summary:")
    print(f"Original number of records: {len(data)}")
    print(f"Records after removing all-zero countries: {len(data_cleaned)}")
    print(f"Number of outliers identified: {len(outliers)}")
    print(f"Number of remaining zero values: {data_cleaned['zero_value'].sum()}")
    
    # Save outliers separately for review
    outliers.to_csv('data/outliers.csv', index=False)
    
    return data_cleaned

if __name__ == "__main__":
    clean_data() 