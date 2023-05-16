import torch
import torch.nn as nn
import numpy as np
from pandas import DataFrame as df

# Assuming df is a pandas DataFrame containing your data with 'top' and 'bottom' columns
lookback = 10
X = []
y = []

for i in range(len(df)-lookback-1):
    X.append(df[['top', 'bottom']].iloc[i:i+lookback].values)
    y.append(df[['top', 'bottom']].iloc[i+lookback].values)

X = np.array(X)
y = np.array(y)

# RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=period).mean()
    avg_loss = abs(down.rolling(window=period).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)


# Model Building
class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Training
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for i in range(epochs):
    for seq, labels in zip(X, y):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# Prediction
future = 10
preds = X[-1].tolist()

model.eval()

for i in range(future):
    seq = torch.FloatTensor(preds[-lookback:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        preds.append(model(seq).tolist())

predicted_top_bottom = preds[-lookback:]

