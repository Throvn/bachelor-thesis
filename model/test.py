import json
import random
from sklearn.metrics import classification_report
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

def testByName(name):
    el = data[data.slug == name] # Should be 0
    # el = data[data.slug == 'bitcoin'] # Should be 1
    # el = data.sample(1)
    print("ID: ", el.index)
    el = el.iloc[0]
    new_data, _ = create_sequences(el, 60)
    print("Selected '" + el['slug'] + "'")
    print("http://localhost:3000/coin.html?id=" + el['slug'])

    # [[[devActivity, twitterFollowers, priceUSD]]]

    # Convert to PyTorch tensor
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
    # new_data_tensor.nan_to_num_(0.0)
    # print(new_data_tensor)

    # Make prediction
    with torch.no_grad():
        new_pred_tensor = model(new_data_tensor)
    new_pred = new_pred_tensor.numpy()

    # Set a threshold (you can experiment with different values)
    threshold = 0.5

    # Convert continuous predictions to binary based on the threshold
    # all_y_pred_class = [int(pred >= threshold) for pred in new_pred]

    # print("Prediction for new data:", new_pred.squeeze())
    # Now use classification_report
    print(np.min(np.around(new_pred.squeeze(), 4)), np.max(np.around(new_pred.squeeze(), 4)))

testByName('cindicator')
testByName('bitcoin')