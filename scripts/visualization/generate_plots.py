import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create images directory
Path('images').mkdir(exist_ok=True)

# Read integrated data
df = pd.read_csv('data/integrated_data.csv', header=[0, 1])
df.columns = ['date', 'country', 'oil_supply', 'wti_price']  # Rename columns
df['date'] = pd.to_datetime(df['date'])

# 1. WTI Price Time Series
plt.figure(figsize=(12, 6))
plt.plot(df['date'].unique(), df.groupby('date')['wti_price'].first(), linewidth=2)
plt.title('WTI Crude Oil Price Trend (2009-2024)')
plt.xlabel('Date')
plt.ylabel('Price (USD/Barrel)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/wti_price_trend.png')
plt.close()

# 2. Top 10 Countries by Average Supply
top_10_countries = df.groupby('country')['oil_supply'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
plt.bar(top_10_countries.index, top_10_countries.values)
plt.title('Top 10 Countries by Average Oil Supply')
plt.xlabel('Country')
plt.ylabel('Average Supply')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/top_10_countries_supply.png')
plt.close()

# 3. Supply vs Price Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['wti_price'], df['oil_supply'], alpha=0.5)
plt.title('Oil Supply vs WTI Price Relationship')
plt.xlabel('WTI Price (USD/Barrel)')
plt.ylabel('Supply')
plt.grid(True)
plt.tight_layout()
plt.savefig('images/supply_price_scatter.png')
plt.close()

# 4. Supply Distribution by Year
df['year'] = df['date'].dt.year
plt.figure(figsize=(12, 6))
plt.boxplot([group['oil_supply'].values for name, group in df.groupby('year')],
            labels=sorted(df['year'].unique()))
plt.title('Oil Supply Distribution by Year')
plt.xlabel('Year')
plt.ylabel('Supply')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/supply_by_year_box.png')
plt.close()

print("Plots generated and saved in images/ directory") 