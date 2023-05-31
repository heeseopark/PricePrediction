#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.utils.data import IterableDataset, DataLoader, Subset
from datetime import datetime as dt, timedelta
import pandas as pd
import os
import random
import numpy as np
import torch.nn as nn
from pandas import DataFrame as df
import mplfinance as mpf

# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

seed = 42  # choose any seed you prefer
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class PriceDataset(torch.utils.data.Dataset):
    def __init__(self, item, timespan, start_date_str, end_date_str):
        self.directory = f'csvfiles/{item}'
        self.item = item
        self.timespan = timespan
        start_date = dt.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = dt.strptime(end_date_str, '%Y-%m-%d').date()
        self.dates = [single_date.strftime("%Y-%m-%d") for single_date in self.daterange(start_date, end_date)]
        self.columns = [1, 4]  # Selecting open and close prices
        self.filenames = self.get_filenames()

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)

    def get_filenames(self):
        filenames = []
        for date in self.dates:
            filename = f"{self.directory}/{self.item}-{self.timespan}-{date}.csv"
            if os.path.exists(filename):
                filenames.append(filename)
        return filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        df = pd.read_csv(filename, usecols=self.columns, header=None)
        df = df[df.columns[::-1]]  # Swap the columns
        df = df.diff(axis=1)[1]  # Compute difference between close and open price for each row
        return torch.tensor(df.values, dtype=torch.float32)  # Convert to tensor


def sliding_window_fn(batch):
    windows = []
    for tensor in batch:
        for i in range(tensor.shape[0] - 100 + 1):  # Create windows of 100 rows each
            windows.append(tensor[i:i+100])
    return torch.stack(windows)


# Create the dataset
dataset = PriceDataset('BTCUSDT', '1m', '2021-03-01', '2023-04-30')

# Shuffle the dataset indices
indices = list(range(len(dataset)))
random.shuffle(indices)

# Split the indices into training and test sets
split_idx = int(0.8 * len(indices))
train_indices, test_indices = indices[:split_idx], indices[split_idx:]

# Create data subsets using the indices
train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

# Create the data loaders
train_loader = DataLoader(train_data, batch_size=1, collate_fn=sliding_window_fn, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=1, collate_fn=sliding_window_fn, shuffle=False, drop_last=True)


# In[3]:


class LSTM1(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=10, num_layers=2):
        super(LSTM1, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, sequence_length, number_of_features]
        x = x.view(x.size(0), -1, 1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# In[4]:


class LSTM2(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=10, num_layers=3):
        super(LSTM2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, sequence_length, number_of_features]
        x = x.view(x.size(0), -1, 1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# In[5]:


class LSTM3(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=10, num_layers=4):
        super(LSTM3, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, sequence_length, number_of_features]
        x = x.view(x.size(0), -1, 1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# In[6]:


class LSTM4(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=10, num_layers=2, dropout_prob=0.05):
        super(LSTM4, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, sequence_length, number_of_features]
        x = x.view(x.size(0), -1, 1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Dropout
        out = self.dropout(out)

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# In[7]:


class LSTM5(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=10, num_layers=3, dropout_prob=0.05):
        super(LSTM5, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, sequence_length, number_of_features]
        x = x.view(x.size(0), -1, 1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Dropout
        out = self.dropout(out)

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# In[8]:


class LSTM6(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=10, num_layers=4, dropout_prob=0.05):
        super(LSTM6, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # Define dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to [batch_size, sequence_length, number_of_features]
        x = x.view(x.size(0), -1, 1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Dropout
        out = self.dropout(out)

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# In[9]:


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)  # Move the data to the device (CPU or GPU)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, data[:, -10:])  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # Move the data to the device
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, data[:, -10:])  # Compute the loss
            test_loss += loss.item() * data.size(0)  # Accumulate the loss
    return test_loss / len(test_loader.dataset)  # Return the average loss


# In[32]:


# Create the model, criterion, and optimizer
best_val_loss = float('inf')
model1 = LSTM1().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)
epochs = 30

# Train and evaluate the model
for epoch in range(epochs):  # Adjust the number of epochs as needed
    train(model1, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model1, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
    
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save({
            'model1_state_dict': model1.state_dict(),
            'optimizer1_state_dict': optimizer.state_dict(),
        }, 'models for report/model1.pth')
        best_val_loss = val_loss


# In[33]:


# Create the model, criterion, and optimizer
best_val_loss = float('inf')
model2 = LSTM2().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
epochs = 30

# Train and evaluate the model
for epoch in range(epochs):  # Adjust the number of epochs as needed
    train(model2, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model2, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save({
            'model2_state_dict': model2.state_dict(),
            'optimizer2_state_dict': optimizer.state_dict(),
        }, 'models for report/model2.pth')
        best_val_loss = val_loss


# In[34]:


# Create the model, criterion, and optimizer
best_val_loss = float('inf')
model3 = LSTM3().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model3.parameters(), lr=0.01)
epochs = 30

# Train and evaluate the model
for epoch in range(epochs):  # Adjust the number of epochs as needed
    train(model3, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model3, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save({
            'model3_state_dict': model3.state_dict(),
            'optimizer3_state_dict': optimizer.state_dict(),
        }, 'models for report/model3.pth')
        best_val_loss = val_loss


# In[47]:


# Create the model, criterion, and optimizer
best_val_loss = float('inf')
model4 = LSTM4().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model4.parameters(), lr=0.01)
epochs = 40

# Train and evaluate the model
for epoch in range(epochs):  # Adjust the number of epochs as needed
    train(model4, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model4, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save({
            'model4_state_dict': model4.state_dict(),
            'optimizer4_state_dict': optimizer.state_dict(),
        }, 'models for report/model4.pth')
        best_val_loss = val_loss


# In[48]:


# Create the model, criterion, and optimizer
best_val_loss = float('inf')
model5 = LSTM5().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model5.parameters(), lr=0.01)
epochs = 40

# Train and evaluate the model
for epoch in range(epochs):  # Adjust the number of epochs as needed
    train(model5, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model5, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save({
            'model5_state_dict': model5.state_dict(),
            'optimizer5_state_dict': optimizer.state_dict(),
        }, 'models for report/model5.pth')
        best_val_loss = val_loss


# In[49]:


# Create the model, criterion, and optimizer
best_val_loss = float('inf')
model6 = LSTM6().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model6.parameters(), lr=0.01)
epochs = 40

# Train and evaluate the model
for epoch in range(epochs):  # Adjust the number of epochs as needed
    train(model6, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model6, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        torch.save({
            'model6_state_dict': model6.state_dict(),
            'optimizer6_state_dict': optimizer.state_dict(),
        }, 'models for report/model6.pth')
        best_val_loss = val_loss


# In[63]:


test_dataset = PriceDataset('ETHUSDT', '1m', '2021-03-01', '2023-04-30')
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=sliding_window_fn, shuffle=False, drop_last=True)


# In[65]:


model_list = [LSTM1(), LSTM2(), LSTM3(), LSTM4(), LSTM5(), LSTM6()]

class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) * 100 / (target + 1e-10)))


for i in range(6):
    # Move model to device
    model = model_list[i].to(device)

    # Initialize the criterion
    criterion = MAPELoss()  # Using MAPELoss

    load_path = 'models for report/'
    model_filename = f'model{i+1}.pth'  # Specify the model filename
    model_path = os.path.join(load_path, model_filename)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint[f'model{i+1}_state_dict'])
    optimizer.load_state_dict(checkpoint[f'optimizer{i+1}_state_dict'])

    # Evaluate the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data[:, :-10])  # Pass the first 90 data points through the model
            loss = criterion(outputs, data[:, -10:])  # Compare the output with the actual last 10 data points
            test_loss += loss.item() * data.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Model: {model.__class__.__name__} | Test Loss (MAPE): {test_loss}%')


# In[16]:


import matplotlib.pyplot as plt
import numpy as np

# Use Mean Squared Error Loss
criterion = nn.MSELoss()

# Random indices for the evaluation
random_indices = random.sample(range(len(test_loader.dataset)), 5)

for i in range(6):
    # Move model to device
    model = model_list[i].to(device)

    load_path = 'models for report/'
    model_filename = f'model{i+1}.pth'  # Specify the model filename
    model_path = os.path.join(load_path, model_filename)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint[f'model{i+1}_state_dict'])
    optimizer.load_state_dict(checkpoint[f'optimizer{i+1}_state_dict'])

    # Evaluate the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data[:, :-10])  # Pass the first 90 data points through the model
            loss = criterion(outputs, data[:, -10:])  # Compare the output with the actual last 10 data points
            test_loss += loss.item() * data.size(0)

            # Check if the current index is in the randomly selected indices
            if j in random_indices:
                # Plot actual and predicted prices
                plt.figure(figsize=(10, 4))
                plt.plot(np.arange(10), outputs.cpu().numpy()[0], label='Predicted')
                plt.plot(np.arange(10), data[:, -10:].cpu().numpy()[0], label='Actual')
                plt.title(f'Model: {model.__class__.__name__} | Data Index: {j}')
                plt.legend()
                plt.show()

    test_loss /= len(test_loader.dataset)
    print(f'Model: {model.__class__.__name__} | Test Loss (MSE): {test_loss}')


# In[11]:


import matplotlib.pyplot as plt
import numpy as np

model_list = [LSTM1(), LSTM2(), LSTM3(), LSTM4(), LSTM5(), LSTM6()]

# Use Mean Squared Error Loss
criterion = nn.MSELoss()

predicted_first_outputs = [[] for _ in range(6)]
actual_first_outputs = []

for i in range(6):
    # Move model to device
    model = model_list[i].to(device)

    load_path = 'models for report/'
    model_filename = f'model{i+1}.pth'  # Specify the model filename
    model_path = os.path.join(load_path, model_filename)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint[f'model{i+1}_state_dict'])
    optimizer.load_state_dict(checkpoint[f'optimizer{i+1}_state_dict'])

    # Evaluate the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data[:, :-10])  # Pass the first 90 data points through the model
            loss = criterion(outputs, data[:, -10:])  # Compare the output with the actual last 10 data points
            test_loss += loss.item() * data.size(0)

            # Store the first output from the prediction and the actual value
            predicted_first_outputs[i].append(outputs.cpu().numpy()[0][0])
            if i == 0:  # Only store actual outputs once
                actual_first_outputs.append(data[:, -10:].cpu().numpy()[0][0])

    test_loss /= len(test_loader.dataset)
    print(f'Model: {model.__class__.__name__} | Test Loss (MSE): {test_loss}')

# Plot all first outputs
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.plot(predicted_first_outputs[i], label=f'Predicted by Model {i+1}')
plt.plot(actual_first_outputs, label='Actual', linewidth=2)
plt.title('First Output Comparison')
plt.legend()
plt.show()


# In[15]:


fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Only create two subplots
models_to_plot = [0, 3]  # Indices of the models to plot

for i in range(2):  # Only loop over two indices
    model_index = models_to_plot[i]
    axs[i].plot(predicted_first_outputs[model_index], label=f'Predicted by Model {model_index + 1}')
    axs[i].plot(actual_first_outputs, label='Actual', linewidth=2)
    axs[i].set_title(f'Model {model_index + 1} First Output Comparison')
    axs[i].legend()
    
plt.tight_layout()
plt.savefig("estimate.jpeg")
plt.show()


# In[17]:


# Load the model checkpoint
checkpoint = torch.load('models for report/model1.pth', map_location=torch.device('cpu'))

model_list = [LSTM1(), LSTM2(), LSTM3(), LSTM4(), LSTM5(), LSTM6()]

model = model_list[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Load model parameters
model.load_state_dict(checkpoint['model1_state_dict'])

# Load optimizer parameters (if necessary)
optimizer.load_state_dict(checkpoint['optimizer1_state_dict'])


# Print optimizer parameters
for param_group in optimizer.param_groups:
    print(param_group)


# In[20]:


import matplotlib.pyplot as plt

# List of model names
models = ['LSTM1', 'LSTM2', 'LSTM3', 'LSTM4', 'LSTM5', 'LSTM6']

# Validation loss during training for each model
training_loss = [201616.7909415593, 155444.79288817875, 112751.76595082523,
                 191259.1743042649, 137705.7925829149, 117673.68080671965]

# Test loss for each model
test_loss = [3883358.8067575432, 4060873.819767862, 4166818.7299175444,
             3844619.7296163063, 3904530.274184491, 4034545.799360251]

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Set the x-axis values (model names) and the y-axis values (losses)
ax.plot(models, training_loss, label='Training Loss', marker='o')
ax.plot(models, test_loss, label='Test Loss', marker='o')

# Set labels for the x-axis, y-axis, and the title
ax.set_xlabel('Models')
ax.set_ylabel('Loss')
ax.set_title('Training and Test Loss for each Model')

# Display the legend
ax.legend()

# Save the plot
plt.savefig('mselosses.jpeg')

# Show the plot
plt.show()

