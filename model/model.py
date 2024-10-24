import torch
import torch.nn as nn
##3 Init Single Input LSTM

class SingleInputLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units=128, isBidirectional=False):
        super(SingleInputLSTMClassifier, self).__init__()
        self.hidden_units = hidden_units
        self.isBidirectional = isBidirectional
        self.num_directions = 2 if isBidirectional else 1
        self.lstm1 = nn.LSTM(input_dim, hidden_units, batch_first=True, bidirectional=isBidirectional)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_units * self.num_directions, hidden_units, batch_first=True, bidirectional=isBidirectional)
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_units * self.num_directions, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_directions, batch_size, self.hidden_units, device=x.device)
        c0 = torch.zeros(self.num_directions, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        out = self.relu1(out)

        h1 = torch.zeros(self.num_directions, batch_size, self.hidden_units, device=x.device)
        c1 = torch.zeros(self.num_directions, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm2(out, (h1, c1))
        out = self.dropout2(out)
        out = self.relu2(out)

        out = self.fc(out[:, -1, :])  # Use the last time step's output
        out = self.sigmoid(out)
        return out