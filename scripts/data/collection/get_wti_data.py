import datetime
import pandas as pd
from pandas_datareader import data as web

# 1) Set time range - match JODI data time period
start = datetime.datetime(2007, 9, 30)
end = datetime.datetime(2024, 12, 31)

# 2) Pull DCOILWTICO (WTI monthly closing price) from FRED
df = web.DataReader('DCOILWTICO', 'fred', start, end)
print("\nRaw data from FRED:")
print(f"Number of rows in raw data: {len(df)}")
print("First few rows of raw data:")
print(df.head())
print("\nDate range in raw data:")
print(f"Start: {df.index.min()}")
print(f"End: {df.index.max()}")

# 3) Resample to get month-end values
wti_monthly = df.resample('ME').last()  # Changed from 'M' to 'ME' to fix deprecation warning
print("\nAfter resampling:")
print(f"Number of rows after resampling: {len(wti_monthly)}")
print("First few rows after resampling:")
print(wti_monthly.head())

# 4) Rename columns and process index to match JODI format
wti_monthly.rename(columns={'DCOILWTICO': 'value'}, inplace=True)

# 5) Add fixed columns to match JODI data format
wti_monthly['energy'] = 'OIL'
wti_monthly['code'] = 'WTI'
wti_monthly['country'] = 'US'
wti_monthly['notes'] = 'WTI Crude Oil Price'

# 6) Convert index to proper date format and reset index
wti_monthly = wti_monthly.reset_index()
print("\nColumns after reset_index:", wti_monthly.columns.tolist())

# Convert date to proper format
wti_monthly['DATE'] = wti_monthly['DATE'].dt.strftime('%d/%m/%Y')
wti_monthly = wti_monthly.rename(columns={'DATE': 'date'})

# 7) Reorder columns to match JODI data structure
wti_monthly = wti_monthly[['energy', 'code', 'country', 'date', 'value', 'notes']]

# 8) Save to local CSV file
wti_monthly.to_csv('wti_monthly_prices.csv', index=False)

print("\nFinal Data Info:")
print(f"Total number of rows in final dataset: {len(wti_monthly)}")
print("\nData Preview:")
print(wti_monthly.head())
print("\nData Time Range:")
print(f"Start Date: {wti_monthly['date'].min()}")
print(f"End Date: {wti_monthly['date'].max()}")
print("\nSuccessfully saved: wti_monthly_prices.csv") 