import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def feature_engineering(df):
    """
    Perform feature engineering on the dataset
    """
    # Handle categorical variables
    df = df.copy()
    
    # Ensure is_turning_point is binary
    df['is_turning_point'] = df['is_turning_point'].astype(int)
    
    # Drop unnecessary columns
    columns_to_drop = ['energy', 'code', 'country', 'date', 'notes', 'event_type']
    df = df.drop(columns=columns_to_drop)
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Ensure all features are numeric
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_columns]
    
    # Standardize features (except target variable)
    features_to_scale = [col for col in df.columns if col != 'is_turning_point']
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df

def calculate_rsi(series, window):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_additional_visualizations(df, feature_importance):
    """
    Create additional visualizations for better data presentation
    """
    # Create directory if it doesn't exist
    os.makedirs('data/plots', exist_ok=True)
    
    # 1. Feature Importance by Category
    categories = {
        'Supply': [col for col in feature_importance['feature'] if 'supply' in col.lower()],
        'Inventory': [col for col in feature_importance['feature'] if 'inventory' in col.lower()],
        'Policy': [col for col in feature_importance['feature'] if 'opec' in col.lower()]
    }
    
    category_importance = {}
    for category, features in categories.items():
        category_importance[category] = feature_importance[feature_importance['feature'].isin(features)]['importance'].sum()
    
    plt.figure(figsize=(10, 6))
    plt.pie(category_importance.values(), labels=category_importance.keys(), autopct='%1.1f%%')
    plt.title('Feature Importance by Category')
    plt.savefig('data/plots/category_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Top 15 Features Bar Plot
    plt.figure(figsize=(12, 8))
    top_15 = feature_importance.head(15)
    sns.barplot(x='importance', y='feature', data=top_15)
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('data/plots/top_15_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Correlation Heatmap
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(10)['feature'].tolist()
    correlation_matrix = df[top_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap (Top 10 Features)')
    plt.tight_layout()
    plt.savefig('data/plots/feature_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(feature_importance['importance'], bins=20)
    plt.title('Distribution of Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('data/plots/importance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_visualizations(df):
    """
    Create key visualization plots for data analysis
    """
    os.makedirs('data/plots', exist_ok=True)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Supply vs Price Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='value_supply', y='value_wti', alpha=0.6)
    plt.title('Supply vs WTI Price Relationship', fontsize=14)
    plt.xlabel('Supply', fontsize=12)
    plt.ylabel('WTI Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/plots/supply_price_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. WTI Price Trend
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['value_wti'], linewidth=2)
    plt.title('WTI Price Trend Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('WTI Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/plots/wti_price_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Top 10 Countries Supply
    top_10_countries = df.groupby('country')['value_supply'].sum().nlargest(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_countries.index, y=top_10_countries.values)
    plt.title('Top 10 Countries by Total Supply', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Total Supply', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/plots/top_10_countries_supply.png', dpi=150, bbox_inches='tight')
    plt.close()

def train_model(df):
    """
    Train the model using the preprocessed features
    """
    # Prepare features and target variable
    X = df.copy()
    y = X.pop('is_turning_point')  # Use turning point as prediction target
    
    # Split training and test sets
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Save predicted turning points to the original data
    # Load the original data to preserve all columns
    original_data = pd.read_csv('data/processed/enhanced_data_v3.csv')
    # Ensure date is datetime for alignment
    if 'date' in original_data.columns:
        original_data['date'] = pd.to_datetime(original_data['date'])
    # Assign predictions to the last len(y_pred) rows
    original_data['predicted_turning_point'] = 0
    original_data.loc[original_data.index[-len(y_pred):], 'predicted_turning_point'] = y_pred
    # Save back to CSV
    original_data.to_csv('data/processed/enhanced_data_v3.csv', index=False)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create prediction results visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title('Supply Turning Point Prediction Results', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Turning Point (0/1)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/plots/supply_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Beautified Feature Importance Plot
    plt.figure(figsize=(10, 7))
    top_n = 10
    palette = sns.color_palette("viridis", top_n)
    ax = sns.barplot(
        x='importance', y='feature',
        data=feature_importance.head(top_n),
        palette=palette
    )
    plt.title('Top 10 Feature Importances', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    # Add value labels
    for i, v in enumerate(feature_importance.head(top_n)['importance']):
        ax.text(v + 0.002, i, f"{v:.2%}", color='black', va='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('data/plots/feature_importance_presentation.png', dpi=150)
    plt.close()
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=['Non-Turning', 'Turning'], yticklabels=['Non-Turning', 'Turning'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig('data/plots/confusion_matrix_presentation.png', dpi=150)
    plt.close()
    
    # Save the feature importance
    feature_importance.to_csv('data/processed/feature_importance.csv', index=False)
    
    return model, feature_importance

def main():
    """
    Main function to run the analysis
    """
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/enhanced_data_v3.csv')
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(data)
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    df_features = feature_engineering(data)
    
    # Train model
    print("\nTraining model...")
    model, feature_importance = train_model(df_features)
    
    # Save results
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(model, 'data/models/supply_chain_model.joblib')
    
    print("\nAnalysis complete!")
    print("- Model saved to: data/models/supply_chain_model.joblib")
    print("- Feature importance saved to: data/processed/feature_importance.csv")
    print("- Visualizations saved to: data/plots/")

if __name__ == "__main__":
    main() 