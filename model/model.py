import torch
import torch.nn as nn
##3 Init Single Input LSTM

class SingleInputLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units=128):
        super(SingleInputLSTMClassifier, self).__init__()
        self.hidden_units = hidden_units
        self.lstm1 = nn.LSTM(input_dim, hidden_units, batch_first=True, bidirectional=True) #not bidirectional architecture
        self.dropout1 = nn.Dropout(0.5) #note Dropout
        self.relu1 = nn.ReLU() #note ReLU activation function
        self.lstm2 = nn.LSTM(hidden_units * 2, hidden_units, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3) #note Dropout
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_units * 2, 1)
        self.sigmoid = nn.Sigmoid() #note Sigmoid Output (min val: 0/ max val: 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        out = self.relu1(out)

        h1 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        c1 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm2(out, (h1, c1))
        out = self.dropout2(out)
        out = self.relu2(out)

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out