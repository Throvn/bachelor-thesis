import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, IterableDataset
from helpers import flatten_data


class JsonDataset(IterableDataset):
    def __init__(self, filename, window_size, isTestData=False):
        with open(filename) as f:
            self.window_size = window_size
            self.json = json.load(f)

            split_index = int(len(self.json) * 0.7)
            if not isTestData:
                self.json = self.json[:split_index]
            else:
                self.json = self.json[split_index:]
    
    def __len__(self):
        return len(self.json)
    
    def __iter__(self):
        for dao in self.json:
            dao_data = flatten_data(dao)  # Assuming flatten_data flattens the DAO json
            df = pd.DataFrame(dao_data)
            feature_columns = df.columns.drop(['isActive'])  # Exclude the target column from features
            
            if len(df) >= self.window_size:
                # Create sequences of inputs and targets
                for i in range(len(df) - self.window_size):
                    X_seq = df[feature_columns].iloc[i:i + self.window_size].to_numpy()
                    y_seq = df['isActive'].iloc[i + self.window_size - 1]
                    yield torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

trainset = JsonDataset("./isDaoActiveData(2).json", 60)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, num_workers=0)