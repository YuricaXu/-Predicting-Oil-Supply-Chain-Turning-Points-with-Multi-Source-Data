import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import os

def mark_turning_points(data, value_col='value_supply', window=3):
    """
    Mark turning points (local maxima/minima) in the supply series
    :param data: DataFrame, must contain value_col
    :param value_col: supply column name
    :param window: sliding window size (odd number, default 3)
    :return: DataFrame with new column is_turning_point
    """
    values = data[value_col].values
    is_turning = np.zeros(len(values), dtype=int)
    half_w = window // 2

    for i in range(half_w, len(values) - half_w):
        window_slice = values[i - half_w: i + half_w + 1]
        center = window_slice[half_w]
        if center == np.max(window_slice) or center == np.min(window_slice):
            is_turning[i] = 1

    data['is_turning_point'] = is_turning
    return data

def analyze_seasonality(data):
    """Analyze seasonal patterns in supply data"""
    # Set date as index
    data = data.set_index('date')
    
    # Decompose time series
    decomposition = seasonal_decompose(data['value_supply'], period=12)  # Monthly data
    
    # Plot decomposition
    plt.figure(figsize=(15, 12))
    
    plt.subplot(411)
    plt.plot(data['value_supply'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('data/plots/seasonal_decomposition.png')
    plt.close()
    
    return decomposition

def test_stationarity(data):
    """Test if the time series is stationary"""
    result = adfuller(data['value_supply'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    return result[1] < 0.05  # Return True if stationary

def create_lagged_features(data, n_lags=3):
    """Create lagged features for supply prediction"""
    for i in range(1, n_lags + 1):
        data[f'supply_lag_{i}'] = data['value_supply'].shift(i)
        data[f'price_lag_{i}'] = data['value_wti'].shift(i)
    
    return data.dropna()

def analyze_demand_patterns(data):
    """Analyze demand patterns and their relationship with supply"""
    # Calculate demand-supply ratio
    data['demand_supply_ratio'] = data['value_supply'].pct_change()
    
    # Plot demand-supply relationship
    plt.figure(figsize=(15, 6))
    plt.plot(data['date'], data['demand_supply_ratio'], label='Demand-Supply Ratio')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Demand-Supply Ratio Over Time')
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/plots/demand_supply_ratio.png')
    plt.close()
    
    return data

def build_demand_forecast_model(data):
    """Build a model to forecast demand based on historical patterns"""
    # Prepare features
    features = ['supply_lag_1', 'supply_lag_2', 'supply_lag_3',
                'price_lag_1', 'price_lag_2', 'price_lag_3']
    target = 'value_supply'
    
    X = data[features]
    y = data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Demand Forecasting')
    plt.tight_layout()
    plt.savefig('data/plots/feature_importance.png')
    plt.close()
    
    return model, scaler, feature_importance

def feature_engineering(data, n_lags=3, ma_windows=[3, 6]):
    
    for i in range(1, n_lags + 1):
        data[f'supply_lag_{i}'] = data['value_supply'].shift(i)
        data[f'price_lag_{i}'] = data['value_wti'].shift(i)
    
    data['supply_pct_change'] = data['value_supply'].pct_change()
    data['price_pct_change'] = data['value_wti'].pct_change()
    
    for w in ma_windows:
        data[f'supply_ma_{w}'] = data['value_supply'].rolling(window=w).mean()
        data[f'price_ma_{w}'] = data['value_wti'].rolling(window=w).mean()
    return data

def train_turning_point_classifier(data):
    """Train a classifier to predict supply turning points, now including OPEC event features"""
    # Original features
    base_features = [
        'supply_lag_1', 'supply_lag_2', 'supply_lag_3',
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'supply_pct_change', 'price_pct_change',
        'supply_ma_3', 'supply_ma_6',
        'price_ma_3', 'price_ma_6'
    ]
    
    # Add OPEC event features (one-hot encoded)
    opec_features = [col for col in data.columns if col.startswith('opec_')]
    feature_cols = base_features + opec_features
    
    # Clean data
    data_model = data.dropna(subset=feature_cols + ['is_turning_point'])
    X = data_model[feature_cols]
    y = data_model['is_turning_point']
    
    # Split data temporally
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nTurning Point Classification Report (with OPEC features):")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    importances = clf.feature_importances_
    feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for feat, imp in feature_importance:
        print(f"{feat}: {imp:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    feat_imp = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    sns.barplot(x='importance', y='feature', data=feat_imp.head(15))
    plt.title('Top 15 Most Important Features for Turning Point Prediction')
    plt.tight_layout()
    plt.savefig('data/plots/feature_importance_with_opec.png')
    plt.close()
    
    return clf, feature_cols

def main():
    print("Loading processed data...")
    data = pd.read_csv('data/processed/processed_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    # Create output directory
    os.makedirs('data/plots', exist_ok=True)
    
    # Mark turning points (assuming mark_turning_points function is integrated)
    print("\nMarking turning points...")
    data = mark_turning_points(data, value_col='value_supply', window=5)
    
    # Analyze seasonality
    print("\nAnalyzing seasonality...")
    decomposition = analyze_seasonality(data)
    
    # Test stationarity
    print("\nTesting stationarity...")
    is_stationary = test_stationarity(data)
    print(f"Time series is {'stationary' if is_stationary else 'not stationary'}")
    
    # Feature engineering
    print("\nFeature engineering...")
    data = feature_engineering(data, n_lags=3, ma_windows=[3, 6])
    # Clean inf/-inf and NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    
    # Train turning point classifier
    print("\nTraining turning point classifier...")
    clf, feature_cols = train_turning_point_classifier(data)
    
    # --- Data Integration: Merge OPEC event data with main dataset ---
    # Load OPEC event data
    opec_df = pd.read_csv('data/opec_events.csv', parse_dates=['date'])

    # Merge OPEC events into main data on 'date'
    data = pd.merge(data, opec_df[['date', 'event_type']], on='date', how='left')

    # Fill missing event_type with 'no_event'
    data['event_type'] = data['event_type'].fillna('no_event')

    # One-hot encode event_type for modeling
    event_dummies = pd.get_dummies(data['event_type'], prefix='opec')
    data = pd.concat([data, event_dummies], axis=1)

    # Save the integrated dataset for downstream analysis
    os.makedirs('data/processed', exist_ok=True)
    data.to_csv('data/processed/integrated_data.csv', index=False)
    print('Integrated dataset saved to data/processed/integrated_data.csv')

    print("\nAnalysis complete! Check the 'data/plots' directory for visualizations.")

if __name__ == "__main__":
    main() 