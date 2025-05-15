import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def mark_turning_points(data, value_col='value_supply', window=3):
    """Mark turning points in the supply series"""
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

def feature_engineering(data, inventory_data, opec_data, n_lags=3, ma_windows=[3, 6]):
    """
    Comprehensive feature engineering, integrating supply, inventory and OPEC data
    """
    # 1. Basic supply features
    for i in range(1, n_lags + 1):
        data[f'supply_lag_{i}'] = data['value_supply'].shift(i)
        data[f'price_lag_{i}'] = data['value_wti'].shift(i)
    
    data['supply_pct_change'] = data['value_supply'].pct_change()
    data['price_pct_change'] = data['value_wti'].pct_change()
    
    for w in ma_windows:
        data[f'supply_ma_{w}'] = data['value_supply'].rolling(window=w).mean()
        data[f'price_ma_{w}'] = data['value_wti'].rolling(window=w).mean()
    
    # 2. Inventory features
    inventory_data['date'] = pd.to_datetime(inventory_data['date'])
    data['date'] = pd.to_datetime(data['date'])
    data = pd.merge(data, inventory_data, on='date', how='left')
    
    # Basic inventory features
    data['inventory_pct_change'] = data['us_inventory_level'].pct_change()
    data['inventory_supply_ratio'] = data['us_inventory_level'] / data['value_supply']
    data['inventory_trend'] = data['us_inventory_level'].diff()
    data['inventory_acceleration'] = data['inventory_trend'].diff()
    data['inventory_price_ratio'] = data['us_inventory_level'] / data['value_wti']
    
    # 3. OPEC features
    opec_data['date'] = pd.to_datetime(opec_data['date'])
    data = pd.merge(data, opec_data[['date', 'event_type']], on='date', how='left')
    data['event_type'] = data['event_type'].fillna('no_event')
    
    # Ensure all OPEC event types exist
    event_types = ['meeting', 'cut', 'maintain', 'no_agreement', 'extend', 'increase', 'no_event']
    for event_type in event_types:
        col_name = f'opec_{event_type}'
        data[col_name] = (data['event_type'] == event_type).astype(int)
        
        # Create interaction features
        data[f'{col_name}_inventory_change'] = data[col_name] * data['inventory_pct_change']
        data[f'{col_name}_supply_change'] = data[col_name] * data['supply_pct_change']
    
    return data

def train_turning_point_classifier(data):
    """Optimized turning point classifier"""
    # Feature list
    feature_cols = [
        # 1. Supply features
        'supply_lag_1', 'supply_lag_2', 'supply_lag_3',
        'supply_pct_change', 'supply_ma_3', 'supply_ma_6',
        
        # 2. Price features
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'price_pct_change', 'price_ma_3', 'price_ma_6',
        
        # 3. Inventory features
        'us_inventory_level', 'inventory_pct_change',
        'inventory_supply_ratio', 'inventory_trend',
        'inventory_acceleration', 'inventory_price_ratio',
        'us_inventory_5y_percentile', 'us_inventory_velocity',
        
        # 4. OPEC features
        'opec_meeting', 'opec_cut', 'opec_maintain', 
        'opec_no_agreement', 'opec_extend', 'opec_increase',
        'opec_no_event',
        
        # 5. Interaction features
        'opec_meeting_inventory_change', 'opec_cut_inventory_change',
        'opec_maintain_inventory_change', 'opec_no_agreement_inventory_change',
        'opec_extend_inventory_change', 'opec_increase_inventory_change',
        'opec_no_event_inventory_change',
        
        'opec_meeting_supply_change', 'opec_cut_supply_change',
        'opec_maintain_supply_change', 'opec_no_agreement_supply_change',
        'opec_extend_supply_change', 'opec_increase_supply_change',
        'opec_no_event_supply_change'
    ]
    
    # Clean data
    data_model = data.dropna(subset=feature_cols + ['is_turning_point'])
    X = data_model[feature_cols]
    y = data_model['is_turning_point']
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    feature_importance_list = []
    predictions = []
    
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
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        )
        clf.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test_scaled)
        predictions.append({
            'test_idx': test_idx,
            'y_true': y_test,
            'y_pred': y_pred
        })
        
        cv_scores.append({
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        })
        
        feature_importance_list.append(clf.feature_importances_)
    
    # Calculate average feature importance
    mean_importance = np.mean(feature_importance_list, axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_importance
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(15, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Turning Point Prediction')
    plt.tight_layout()
    plt.savefig('data/plots/feature_importance_combined.png')
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
    
    # Print feature importance ranking
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return clf, feature_cols, feature_importance, predictions

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/processed_data.csv')
    inventory_data = pd.read_csv('data/processed/inventory_features.csv')
    opec_data = pd.read_csv('data/opec_events.csv')
    
    # Create output directory
    os.makedirs('data/plots', exist_ok=True)
    
    # Data preprocessing
    data['date'] = pd.to_datetime(data['date'])
    
    # Mark turning points
    print("\nMarking turning points...")
    data = mark_turning_points(data, value_col='value_supply', window=5)
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    data = feature_engineering(data, inventory_data, opec_data, n_lags=3, ma_windows=[3, 6])
    
    # Clean outliers
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna()
    
    # Train turning point classifier
    print("\nTraining turning point classifier...")
    clf, feature_cols, feature_importance, predictions = train_turning_point_classifier(data)
    
    # Save processed data
    output_path = 'data/processed/enhanced_data_v3.csv'
    data.to_csv(output_path, index=False)
    print(f"\nEnhanced dataset saved to {output_path}")
    
    # Save feature importance
    feature_importance.to_csv('data/processed/feature_importance_v3.csv', index=False)
    print("\nFeature importance saved to data/processed/feature_importance_v3.csv")
    
    print("\nAnalysis complete! Check the 'data/plots' directory for visualizations.")

if __name__ == "__main__":
    main() 