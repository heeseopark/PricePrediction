import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import timedelta, date, datetime

class PriceDataset(Dataset):
    def __init__(self, item, timespan, start_date_str, end_date_str):
        self.directory = 'csvfiles'
        self.item = item
        self.timespan = timespan
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        self.dates = [single_date.strftime("%Y-%m-%d") for single_date in self.daterange(start_date, end_date)]
        self.columns = [0, 1, 2, 3, 4, 7]

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        filename = f"{self.directory}/{self.item}-{self.timespan}-{date}.csv"
        df = pd.read_csv(filename, usecols=self.columns, header=None)
        return torch.tensor(df.values, dtype=torch.float32)

# usage example:

dataset = PriceDataset('BTCUSDT', '1m', '2021-03-01', '2023-04-30')
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# loop over dataloader
for batch in dataloader:
    # batch is a tensor of shape [batch_size, num_rows, num_cols]
    pass
