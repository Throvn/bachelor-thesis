import os
import random
from shutil import ExecError
import torch
import numpy as np
from sklearn.metrics import classification_report
from model import SingleInputLSTMClassifier
from test import MODEL_SAVE_PATH, DATA_FILE_NAME
from balance import grouped_test
import pandas as pd

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CONSTANTS
WINDOW_SIZE = 64
DOWNTREND_PERIOD = 60  # days
print(MODEL_SAVE_PATH)

# Load model checkpoint if available
if not os.path.exists(MODEL_SAVE_PATH):
    raise ValueError("Path to Model file doesn't exist")
checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=True)

# Prepare the test data
num_testing_observations = 0
all_y_true = []
all_y_pred = []

# Cache check
cacheExists = os.path.exists("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy") and os.path.exists("./cache/" + MODEL_SAVE_PATH + "-all_y_pred.npy")
if cacheExists:
    print("Cache exists... Skipping repeated classification.")
    all_y_true = np.asarray(np.load("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy"))
    all_y_pred = np.asarray(np.load("cache/" + MODEL_SAVE_PATH + "-all_y_pred.npy"))
    num_testing_observations = len(all_y_pred)
else:
    raise RuntimeError("Cache does not exist. Execute 'python test.py' first.")




# Downward trend heuristic implementation
def is_downward_trend(prices, period=DOWNTREND_PERIOD):
    prices_series = pd.Series(prices)
    prices_series = pd.to_numeric(prices_series, errors='coerce').dropna()
    downward_days = (prices_series.diff() < 0).sum()  # Count price drops
    return downward_days >= period * 0.7  # At least 70% downward days in the period

# Example: Apply heuristic to flag observations
price_data = grouped_test['priceUSD'].values
downward_flags = [is_downward_trend(prices, DOWNTREND_PERIOD) for prices in price_data]


#################################

# Define ground truth labels and predictions
all_y_pred_heuristic = []  # Predictions from the heuristic

# Apply the heuristic
for index, observation in grouped_test.iterrows():
    prices = observation['priceUSD']
    if isinstance(prices, dict):
        prices = list(prices.values())
    heuristic_prediction = int(is_downward_trend(prices, DOWNTREND_PERIOD))  # Convert True/False to 1/0
    all_y_pred_heuristic.append(heuristic_prediction)

# Generate classification report for the heuristic
print("Classification Report for Downward Trend Heuristic:")
print(classification_report(all_y_true, all_y_pred_heuristic, target_names=["Operating", "Abandoned"]))


# Compare heuristic and model predictions
heuristic_accuracy = np.mean([1 if flag == true_label else 0 for flag, true_label in zip(downward_flags, all_y_true)])
print(f"Downward Trend Heuristic Accuracy: {heuristic_accuracy:.2f}")

print(f"Evaluated on {num_testing_observations} test observations.")