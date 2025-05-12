import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import os

def mark_turning_points(data, value_col='value_supply', window=3):
    """
    Mark turning points (local maxima/minima) in the supply series
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
    data = data.set_index('date')
    decomposition = seasonal_decompose(data['value_supply'], period=12)
    
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
    
    return result[1] < 0.05

def feature_engineering(data, inventory_data, n_lags=3, ma_windows=[3, 6]):
    """
    Enhanced feature engineering, integrating inventory data
    """
    # Basic features
    for i in range(1, n_lags + 1):
        data[f'supply_lag_{i}'] = data['value_supply'].shift(i)
        data[f'price_lag_{i}'] = data['value_wti'].shift(i)
    
    # Rate of change features
    data['supply_pct_change'] = data['value_supply'].pct_change()
    data['price_pct_change'] = data['value_wti'].pct_change()
    
    # Moving average features
    for w in ma_windows:
        data[f'supply_ma_{w}'] = data['value_supply'].rolling(window=w).mean()
        data[f'price_ma_{w}'] = data['value_wti'].rolling(window=w).mean()
    
    # Integrate inventory features
    inventory_data['date'] = pd.to_datetime(inventory_data['date'])
    data['date'] = pd.to_datetime(data['date'])
    
    # Merge inventory data
    data = pd.merge(data, inventory_data, on='date', how='left')
    
    # Inventory-related features
    data['inventory_pct_change'] = data['us_inventory_level'].pct_change()
    data['inventory_supply_ratio'] = data['us_inventory_level'] / data['value_supply']
    
    # Inventory trend features
    data['inventory_trend'] = data['us_inventory_level'].diff()
    data['inventory_acceleration'] = data['inventory_trend'].diff()
    
    # Inventory-price relationship features
    data['inventory_price_ratio'] = data['us_inventory_level'] / data['value_wti']
    
    return data

def train_turning_point_classifier(data):
    """
    Optimized turning point classifier, integrating inventory features
    """
    # Feature list
    feature_cols = [
        # Supply features
        'supply_lag_1', 'supply_lag_2', 'supply_lag_3',
        'supply_pct_change', 'supply_ma_3', 'supply_ma_6',
        
        # Price features
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'price_pct_change', 'price_ma_3', 'price_ma_6',
        
        # Inventory features
        'us_inventory_level', 'inventory_pct_change',
        'inventory_supply_ratio', 'inventory_trend',
        'inventory_acceleration', 'inventory_price_ratio',
        'us_inventory_5y_percentile', 'us_inventory_velocity'
    ]
    
    # Clean data
    data_model = data.dropna(subset=feature_cols + ['is_turning_point'])
    X = data_model[feature_cols]
    y = data_model['is_turning_point']
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize result storage
    cv_scores = []
    feature_importance_list = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test_scaled)
        cv_scores.append({
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        })
        
        # Record feature importance
        feature_importance_list.append(clf.feature_importances_)
    
    # Calculate average feature importance
    mean_importance = np.mean(feature_importance_list, axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_importance
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Turning Point Prediction')
    plt.tight_layout()
    plt.savefig('data/plots/feature_importance_with_inventory.png')
    plt.close()
    
    # Print evaluation results
    print("\nCross-validation Results:")
    avg_metrics = {
        'precision': np.mean([s['classification_report']['1']['precision'] for s in cv_scores]),
        'recall': np.mean([s['classification_report']['1']['recall'] for s in cv_scores]),
        'f1-score': np.mean([s['classification_report']['1']['f1-score'] for s in cv_scores])
    }
    print("\nAverage Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return clf, feature_cols, feature_importance

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/processed_data.csv')
    inventory_data = pd.read_csv('data/processed/inventory_features.csv')
    
    # Create output directory
    os.makedirs('data/plots', exist_ok=True)
    
    # Data preprocessing
    data['date'] = pd.to_datetime(data['date'])
    
    # Mark turning points
    print("\nMarking turning points...")
    data = mark_turning_points(data, value_col='value_supply', window=5)
    
    # Seasonal analysis
    print("\nAnalyzing seasonality...")
    decomposition = analyze_seasonality(data)
    
    # Stationarity test
    print("\nTesting stationarity...")
    is_stationary = test_stationarity(data)
    print(f"Time series is {'stationary' if is_stationary else 'not stationary'}")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    data = feature_engineering(data, inventory_data, n_lags=3, ma_windows=[3, 6])
    
    # Clean outliers
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    
    # Train turning point classifier
    print("\nTraining turning point classifier...")
    clf, feature_cols, feature_importance = train_turning_point_classifier(data)
    
    # Save processed data
    output_path = 'data/processed/enhanced_data.csv'
    data.to_csv(output_path, index=False)
    print(f"\nEnhanced dataset saved to {output_path}")
    
    print("\nAnalysis complete! Check the 'data/plots' directory for visualizations.")

if __name__ == "__main__":
    main() 