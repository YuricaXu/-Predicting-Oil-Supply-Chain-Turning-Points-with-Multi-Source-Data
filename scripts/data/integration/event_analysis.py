import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_data():
    """
    Load and prepare the integrated dataset
    """
    print("Loading integrated dataset...")
    df = pd.read_csv('data/processed/integrated_data.csv', parse_dates=['date'])
    return df

def analyze_event_window(df, event_types=None, window=3):
    """
    Analyze price and supply changes around OPEC events
    
    Parameters:
    -----------
    df : DataFrame
        Integrated dataset with date, value_wti, value_supply, and event_type
    event_types : list, optional
        List of event types to analyze. If None, analyze all types
    window : int, optional
        Number of months before and after event to analyze
    """
    if event_types is None:
        event_types = df['event_type'].unique()
        event_types = event_types[event_types != 'no_event']
    
    print(f"\nAnalyzing event windows (±{window} months) for events: {', '.join(event_types)}")
    
    results = []
    for event in event_types:
        event_dates = df[df['event_type'] == event]['date']
        print(f"\nFound {len(event_dates)} {event} events")
        
        for d in event_dates:
            # Convert months to days (approximately 30 days per month)
            days_window = window * 30
            # Get data within window
            mask = (df['date'] >= d - pd.Timedelta(days=days_window)) & \
                  (df['date'] <= d + pd.Timedelta(days=days_window))
            window_df = df[mask].copy()
            window_df['event_type'] = event
            window_df['event_date'] = d
            # Calculate months by dividing days by 30
            window_df['relative_month'] = ((window_df['date'] - d).dt.days / 30).round().astype(int)
            results.append(window_df)
    
    event_window_df = pd.concat(results, ignore_index=True)
    return event_window_df

def plot_event_window_analysis(event_window_df, event_types, window=3):
    """
    Create visualizations for event window analysis
    """
    print("\nGenerating event window plots...")
    os.makedirs('data/plots', exist_ok=True)
    
    # 1. WTI Price Changes
    plt.figure(figsize=(12, 6))
    for event in event_types:
        mean_series = event_window_df[event_window_df['event_type'] == event].groupby('relative_month')['value_wti'].mean()
        plt.plot(mean_series.index, mean_series.values, marker='o', label=event)
    
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Months Relative to Event')
    plt.ylabel('Average WTI Price')
    plt.title('WTI Price Changes Around OPEC Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/plots/event_window_wti.png')
    plt.close()
    
    # 2. Supply Changes
    plt.figure(figsize=(12, 6))
    for event in event_types:
        mean_series = event_window_df[event_window_df['event_type'] == event].groupby('relative_month')['value_supply'].mean()
        plt.plot(mean_series.index, mean_series.values, marker='o', label=event)
    
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Months Relative to Event')
    plt.ylabel('Average Supply')
    plt.title('Supply Changes Around OPEC Events')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data/plots/event_window_supply.png')
    plt.close()

def calculate_event_statistics(event_window_df, event_types):
    """
    Calculate statistical measures for each event type
    """
    print("\nCalculating event statistics...")
    stats_results = []
    
    for event in event_types:
        event_data = event_window_df[event_window_df['event_type'] == event]
        
        # Before-After Analysis
        before_price = event_data[event_data['relative_month'] < 0]['value_wti'].mean()
        after_price = event_data[event_data['relative_month'] > 0]['value_wti'].mean()
        price_change = ((after_price - before_price) / before_price) * 100
        
        before_supply = event_data[event_data['relative_month'] < 0]['value_supply'].mean()
        after_supply = event_data[event_data['relative_month'] > 0]['value_supply'].mean()
        supply_change = ((after_supply - before_supply) / before_supply) * 100
        
        # T-test for price changes
        before_prices = event_data[event_data['relative_month'] < 0]['value_wti']
        after_prices = event_data[event_data['relative_month'] > 0]['value_wti']
        t_stat, p_value = stats.ttest_ind(before_prices, after_prices)
        
        stats_results.append({
            'event_type': event,
            'avg_price_before': before_price,
            'avg_price_after': after_price,
            'price_change_pct': price_change,
            'avg_supply_before': before_supply,
            'avg_supply_after': after_supply,
            'supply_change_pct': supply_change,
            'price_change_significant': p_value < 0.05,
            'p_value': p_value
        })
    
    return pd.DataFrame(stats_results)

def main():
    # Load data
    df = load_data()
    
    # Define event types to analyze
    event_types = ['cut', 'increase', 'maintain', 'extend', 'no_agreement']
    
    # Perform event window analysis
    event_window_df = analyze_event_window(df, event_types, window=3)
    
    # Create visualizations
    plot_event_window_analysis(event_window_df, event_types)
    
    # Calculate and display statistics
    stats_df = calculate_event_statistics(event_window_df, event_types)
    print("\nEvent Impact Statistics:")
    print(stats_df.round(2).to_string())
    
    # Save statistics
    stats_df.to_csv('data/processed/event_statistics.csv', index=False)
    print("\nAnalysis complete! Check the 'data/plots' directory for visualizations.")

if __name__ == "__main__":
    main() 