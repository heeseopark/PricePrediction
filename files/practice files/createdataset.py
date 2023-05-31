import requests
import pandas as pd
from datetime import datetime, timedelta

# Define the URL
url = 'https://api.binance.com/api/v3/klines'
params = {'symbol': 'BTCUSDT', 'interval': '1m'}

# Get today's date and yesterday's date
today = datetime.now()
yesterday = today - timedelta(days=1)

# Initialize DataFrame
df = pd.DataFrame()

while True:
    # Make the GET request
    response = requests.get(url, params=params)
    data = response.json()

    # Convert the data to a DataFrame and append it to the existing DataFrame
    df_temp = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df = pd.concat([df, df_temp], ignore_index=True)

    # Get the time of the earliest data point in the DataFrame
    earliest_time = pd.to_datetime(df['Open time'].min(), unit='ms')

    # If the earliest time in the DataFrame is later than yesterday, break the loop
    if earliest_time <= yesterday:
        break

    # Otherwise, set the end time for the next request to be the earliest time in the DataFrame
    params['endTime'] = int(earliest_time.timestamp() * 1000)

# Save the DataFrame to a CSV file
df.to_csv('btc_data.csv', index=False)
