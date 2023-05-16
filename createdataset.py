import requests
import pandas as pd
import torch

# Define the URL and the parameters
url = 'https://api.binance.com/api/v3/klines'
params = {'symbol': 'BTCUSDT', 'interval': '1m'}

# Make the GET request
response = requests.get(url, params=params)
data = response.json()

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

# Convert the time columns to datetime
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

# Select the relevant columns
df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Convert the price and volume data to floats
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = df[col].astype(float)

# Convert the DataFrame to a PyTorch tensor
tensor = torch.tensor(df[['Open', 'High', 'Low', 'Close', 'Volume']].values)

print(tensor)
print(len(tensor))