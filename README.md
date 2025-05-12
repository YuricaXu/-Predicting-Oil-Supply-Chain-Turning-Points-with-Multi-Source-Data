# Oil Supply Chain Turning Point Prediction Model

This project implements a machine learning model to predict turning points in oil supply chain dynamics, incorporating supply data, OPEC events, and inventory levels.

## Project Structure

```
machine_learning_project/
├── data/
│   ├── raw/           # Raw data files
│   ├── processed/     # Processed data files
│   └── plots/         # Visualization outputs
├── docs/              # Documentation and results
├── models/            # Saved model files
└── scripts/          # Analysis scripts
```

## Data Sources and Features

1. **Supply Data**
   - JODI oil supply data
   - Historical supply patterns
   - Supply change indicators

2. **OPEC Event Data**
   - OPEC meeting decisions
   - Production policy changes
   - Event impact analysis
   
3. **Inventory Data**
   - US inventory levels (EIA)
   - Global inventory changes
   - Inventory-supply ratios

4. **Price Data**
   - WTI crude oil prices
   - Price-supply relationships
   - Price trend indicators

## Feature Engineering

1. **Time Series Features**
   - Supply lags (1-3 months)
   - Moving averages (3 and 6 months)
   - Percentage changes

2. **OPEC Event Features**
   - Meeting outcomes (cut/maintain/increase)
   - Policy implementation periods
   - Supply change during events

3. **Inventory Indicators**
   - Inventory velocity
   - Inventory acceleration
   - Inventory-price ratios
   - Supply-inventory relationships

4. **Data Preprocessing**
   - Feature standardization
   - Binary classification target
   - Handling missing values
   - Outlier treatment

## Model Architecture

1. **Model Type**: Random Forest Classifier

2. **Hyperparameters**:
   - n_estimators: 200
   - max_depth: 10
   - min_samples_split: 5
   - min_samples_leaf: 2
   - random_state: 42

3. **Training Setup**:
   - Train/Test split: 80%/20%
   - Standardized features
   - Binary classification (turning point vs. non-turning point)

## Model Performance (v4)

1. **Classification Metrics**:
   - Accuracy: 82%
   - Precision: 82%
   - Recall: 81%
   - F1 Score: 82%

2. **Feature Importance**:
   ```
   Top 5 Features:
   - supply_pct_change: 15.66%
   - opec_no_event_supply_change: 15.00%
   - inventory_supply_ratio: 12.33%
   - value_supply: 9.17%
   - supply_lag_1: 6.26%
   ```

## Key Findings

1. **Turning Point Prediction**:
   - High accuracy in identifying market turning points
   - Balanced precision and recall performance
   - Robust feature importance distribution

2. **Feature Impact**:
   - Supply changes are primary indicators
   - OPEC events provide valuable context
   - Inventory levels offer predictive power

3. **Model Evolution**:
   - Improved from pure supply focus
   - Enhanced with OPEC event integration
   - Strengthened by inventory data

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

1. **Data Preparation**:
```python
from scripts.supply_chain_analysis_v4 import feature_engineering
df_features = feature_engineering(data)
```

2. **Model Training**:
```python
from scripts.supply_chain_analysis_v4 import train_model
model, feature_importance = train_model(df_features)
```

3. **Make Predictions**:
```python
# Predict turning points
y_pred = model.predict(X_test)
```

## Future Improvements

1. **Model Enhancements**:
   - Experiment with deep learning approaches
   - Implement ensemble methods
   - Add time-series specific models

2. **Feature Engineering**:
   - Incorporate more market indicators
   - Develop composite features
   - Add geopolitical factors

3. **Data Integration**:
   - Add more inventory data sources
   - Include transportation data
   - Consider weather impacts

## Documentation

For detailed information about recent updates and improvements, please refer to:
- [Version 4 Updates](docs/version4_updates.md)
#https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=W&n=PET&s=WCESTUS1