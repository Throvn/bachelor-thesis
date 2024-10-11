import json
import random
import torch
import numpy as np
import pandas as pd
from model import SingleInputLSTMClassifier
from preparation import create_sequences

checkpoint = torch.load("./model_file", weights_only=True)

input_dim=(3)
model = SingleInputLSTMClassifier(input_dim)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

data = pd.read_json("./isDaoActiveData(4).json")
el = data.sample(1)
print("ID: ", el.index)
el = el.iloc[0]
new_data, _ = create_sequences(el, 60)
print("Selected '" + el['slug'] + "'")
print("http://localhost:3000/coin.html?id=" + el['slug'])

# [[[devActivity, twitterFollowers, priceUSD]]]

# Convert to PyTorch tensor
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
new_data_tensor.nan_to_num_(0.0)
print(new_data_tensor)

# Make prediction
with torch.no_grad():
    new_pred_tensor = model(new_data_tensor)
new_pred = new_pred_tensor.numpy()

print("Prediction for new data:", new_pred.squeeze())
