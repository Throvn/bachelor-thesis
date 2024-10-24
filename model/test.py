import json
import os
import sys
import random
from sklearn.metrics import classification_report
import torch
import numpy as np
import pandas as pd
from model import SingleInputLSTMClassifier
from preparation import create_sequences

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CONSTANTS
DATA_FILE_NAME = "../classifiedDAOs.json"
WINDOW_SIZE = 60
MODEL_SAVE_PATH = "./unidirectional_bce_model_balanced"
print(MODEL_SAVE_PATH)

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

if sys.argv[1] != "batch":
    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=True)

    input_dim=(3)
    model = SingleInputLSTMClassifier(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    data = pd.read_json("./isDaoActiveData(4).json")

    testByName('cindicator')
    testByName('bitcoin')






if sys.argv[1] == "batch":
    # from preparation import create_sequences


    # Load model checkpoint if available
    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=True) if os.path.exists(MODEL_SAVE_PATH) else {}

    # Load and prepare the dataset
    file = open(DATA_FILE_NAME)
    total_training_data = json.load(file)
    file.close()
    total_training_data = pd.DataFrame(total_training_data).sample(frac=1, random_state=1337).reset_index(drop=True)

    split_index = int(len(total_training_data) * 0.7)
    grouped_test = total_training_data[split_index:]

    device = torch.device('mps')

    # Define the model
    num_features = 3  # devActivity, twitterFollowers, priceUSD
    input_dim = num_features
    model = SingleInputLSTMClassifier(input_dim).to(device)

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Prepare the test data
    num_testing_observations = 0
    all_y_true = []
    all_y_pred = []

    for index, observation in grouped_test.iterrows():
        print("\n", observation.slug, end="")

        X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
        if not len(X_seq) or not len(y_seq):
            continue
        X = torch.tensor(X_seq.squeeze().squeeze()).to(torch.float32).to(device)
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.tensor(y_seq).to(torch.float32).to(device)

        with torch.no_grad():
            model.eval()
            output = model(X)
            num_testing_observations += 1

            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(output.cpu().numpy())

    # Set a threshold for classification
    threshold = 0.5
    all_y_pred_class = [int(pred >= threshold) for pred in all_y_pred]

    # Generate the classification report
    print("Final Classification Report:")
    print("File: ", MODEL_SAVE_PATH)
    print(classification_report(all_y_true, all_y_pred_class, target_names=["Abandoned", "Operating"]))

    print(f"Evaluated on {num_testing_observations} test observations.")