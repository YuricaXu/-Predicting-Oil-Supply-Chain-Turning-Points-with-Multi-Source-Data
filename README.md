Implement a machine learning model to **predict turning points in the oil supply chain** by integrating multiple data sources:
- supply data
- inventory data
- policy events
- oil price data

The goal is to detect market shifts early and support timely decision-making.

# Oil Supply Chain Turning Point Prediction

This project uses machine learning to predict turning points in the oil market, based on supply, OPEC events, inventory, and price data.

## Main Files
- `v4_final.py`: Main script for feature engineering, model training, and visualization
- `data/processed/enhanced_data_v3.csv`: Final processed dataset
- `visualizations/`: Key result plots

## How to Run
1. Make sure you have Python 3.8+ and the required packages:
   - pandas, numpy, scikit-learn, matplotlib, seaborn
2. Place your data in `data/processed/enhanced_data_v3.csv` (already included for demo)
3. Run the main script:
   ```bash
   python v4_final.py
   ```
4. Check results in the `visualizations/` folder.

## What the Results Mean
- The model predicts when the oil supply trend will change (turning points)
- Accuracy is about 82% on test data
- Most important features: supply changes, OPEC events, inventory ratios
- Visualizations show feature importance, prediction results, and key data relationships

## Future Work
- Use more detailed (daily/weekly) data
- Improve policy feature processing
- Try more advanced models

For more details, see the code and comments in `v4_final.py`.