import itertools
import os
import random
import torch
import numpy as np
from sklearn.metrics import classification_report
from test import MODEL_SAVE_PATH
from balance import grouped_test
import pandas as pd

from preparation import create_sequences

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

# Cache check
cacheExists = os.path.exists("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy")
if cacheExists:
    print("Cache exists... Skipping repeated classification.")
    all_y_true = np.asarray(np.load("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy"))
else:
    raise RuntimeError("Cache does not exist. Execute 'python test.py' first.")


def trend_value(nums: list):
    summed_nums = sum(nums)
    multiplied_data = 0
    summed_index = 0 
    squared_index = 0

    for index, num in enumerate(nums):
        index += 1
        multiplied_data += index * num
        summed_index += index
        squared_index += index**2

    numerator = (len(nums) * multiplied_data) - (summed_nums * summed_index)
    denominator = (len(nums) * squared_index) - summed_index**2
    if denominator != 0:
        return numerator/denominator
    else:
        return 0

#################################

# Define ground truth labels and predictions
all_y_pred_heuristic = []  # Predictions from the heuristic

price_heur = []
dev_heur = []
twitter_heur = []
# Apply the heuristic
for index, observation in grouped_test.iterrows():
    X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
    if not len(X_seq) or not len(y_seq):
        continue
    
    # Process 'priceUSD', 'devActivity', 'twitterFollowers'
    if prices := observation['priceUSD']:
        prices = [entry['value'] for entry in prices]
        price_trend = trend_value(prices)
        price_heur.append(price_trend >= 0)
    
    dev_activity_trend = None
    if dev_activity := observation['devActivity']:
        dev_activity = [entry['value'] for entry in dev_activity]
        # sum because it's already the change.
        dev_activity_trend = sum(dev_activity[-14:]) > 0
    dev_heur.append(dev_activity_trend)
    
    twitter_followers_trend = None
    if twitter_followers := observation['twitterFollowers']:
        twitter_followers = [entry['value'] for entry in twitter_followers]
        # sum because it's already the change.
        twitter_followers_trend = trend_value(list(itertools.accumulate(twitter_followers))) >= 0
    twitter_heur.append(twitter_followers_trend)

    # Determine if any trend is positive
    num_testing_observations += 1
    

# Generate classification report for the heuristic
print("Prices USD:")
print(classification_report(all_y_true, price_heur, target_names=["Abandoned", "Operating"]))

print("Dev Activity:")
# Drop all values where dev_heur doesn't have an entry
dev_true = [val2 for val, val2 in zip(dev_heur, all_y_true) if val is not None]
dev_heur = [val for val in dev_heur if val is not None]
print(classification_report(dev_true, dev_heur, target_names=["Abandoned", "Operating"]))

print("Twitter Follower:")
# Drop all values where twitter_heur doesn't have an entry
twitter_true = [val2 for val, val2 in zip(twitter_heur, all_y_true) if val is not None]
twitter_heur = [val for val in twitter_heur if val is not None]
print(classification_report(twitter_true, twitter_heur, target_names=["Abandoned", "Operating"]))

print(f"Evaluated on {num_testing_observations} test observations.")