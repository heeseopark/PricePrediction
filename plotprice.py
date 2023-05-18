import pandas as pd
import matplotlib.pyplot as plt

# Define column names
column_names = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", 
                "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]

# Load the data
df = pd.read_csv('D:\\Github\\PricePrediction\\csvfiles\\BTCUSDT-1m-2023-05-11.csv', names=column_names)

# Convert 'Open time' column to datetime
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')

# Plot the data
plt.figure(figsize=(10,6))
plt.plot(df['Open time'], df['Close'])
plt.title('Bitcoin Close Price over Time')
plt.xlabel('Open time')
plt.ylabel('Close Price')
plt.show()
