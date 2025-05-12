import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/processed/enhanced_data_v3.csv')

# Generate correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[['value_supply', 'value_wti', 'us_inventory_level', 'inventory_supply_ratio']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# Generate supply vs price scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['value_supply'], df['value_wti'], alpha=0.5)
plt.xlabel('Supply')
plt.ylabel('WTI Price')
plt.title('Supply vs WTI Price')
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/supply_vs_price.png')
plt.close()

# Generate WTI price trend plot
plt.figure(figsize=(12, 6))
df['date'] = pd.to_datetime(df['date'])
df.set_index('date')['value_wti'].plot()
plt.title('WTI Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('WTI Price')
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/wti_price_trend.png')
plt.close()

# Generate top 10 countries by supply plot
top_10_supply = df.groupby('country')['value_supply'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
top_10_supply.plot(kind='bar')
plt.title('Top 10 Countries by Average Supply')
plt.xlabel('Country')
plt.ylabel('Average Supply')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/top_10_supply.png')
plt.close()

# Overlay WTI price, actual turning points, and predicted turning points
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['value_wti'], label='WTI Price', color='blue')

# Actual turning points
if 'is_turning_point' in df.columns:
    plt.scatter(df.loc[df['is_turning_point'] == 1, 'date'],
                df.loc[df['is_turning_point'] == 1, 'value_wti'],
                color='red', marker='o', label='Actual Turning Point')

# Predicted turning points (if available)
if 'predicted_turning_point' in df.columns:
    plt.scatter(df.loc[df['predicted_turning_point'] == 1, 'date'],
                df.loc[df['predicted_turning_point'] == 1, 'value_wti'],
                color='magenta', marker='*', s=150, label='Predicted Turning Point')

plt.title('WTI Price with Actual and Predicted Turning Points')
plt.xlabel('Date')
plt.ylabel('WTI Price')
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/wti_with_turning_points.png', dpi=150)
plt.close() 