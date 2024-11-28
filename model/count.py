# This script gives you the actual number of used samples back.
# Some DAOs from JQ still need to be skipped, because we decided that you need 60 days of flawless data.


import numpy as np
import pandas as pd
import torch
import random

from typing import Final
from datetime import datetime
import os

from preparation import create_sequences
from balance import grouped_train, grouped_test
script_start = datetime.now()

def print(*args, end="\n", flush=False):
    with open('./logs/' + str(script_start) + '.log', "a") as f:
        for arg in args:
            f.write(str(arg))
        f.write(end)
    __builtins__.print(*args, end=end, flush=flush)


print("\n\n")
print("-" * 50)
print("Starting lstm.py at '" + str(script_start) + "'.")

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Is this how many 'days' the time series should be predicted?
WINDOW_SIZE: Final[int] = 60 # TODO: 120
TIMESERIES_SPLITS: Final[int] = 3

MIN_DATA_OBSERVATIONS: Final[int] = WINDOW_SIZE + (TIMESERIES_SPLITS + 1)


IS_BIDIRECTIONAL = True
MODEL_SAVE_PATH = "./" + ("bi" if IS_BIDIRECTIONAL else "uni") + "directional_focal_model_full_correct_a0.29_g3"
print("Modelname: %s", MODEL_SAVE_PATH)
checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=True) if os.path.exists(MODEL_SAVE_PATH) else {}

##2 define tensor structure
print("Preparing data for training...") 

def trainIterator():
    for _, observation in grouped_train.iterrows():
        observation: pd.Series

        print("\tCreating sequence for '" + observation.slug + "'... ", end="")
        if len(observation.priceUSD) < MIN_DATA_OBSERVATIONS:
            raise ValueError("'" + observation.slug + "' has less than MIN_DATA_OBSERVATION entries")

        X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
        # print(X_seq)
        yield X_seq, y_seq
        # print("  " + str(y_seq) + "  ", end="")
        print("Done.")

def testIterator():
    for _, observation in grouped_test.iterrows():
        observation: pd.Series

        print("\tCreating sequence for '" + observation.slug + "'... ", end="")
        if len(observation.priceUSD) < MIN_DATA_OBSERVATIONS:
            raise ValueError("'" + observation.slug + "' has less than MIN_DATA_OBSERVATION entries")

        X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
        # print(X_seq)
        yield X_seq, y_seq
        # print("  " + str(y_seq) + "  ", end="")
        print("Done.")

print()
actualNumberOfSamples = 0
actualNumberOfIncompleteSamples = 0
for name, (X, y) in enumerate(trainIterator()):
    if X.size <= 3 or y.size <= 3:
        actualNumberOfIncompleteSamples += 1
        continue
    actualNumberOfSamples += 1

for name, (X, y) in enumerate(testIterator()):
    if X.size <= 3 or y.size <= 3:
        actualNumberOfIncompleteSamples += 1
        continue
    actualNumberOfSamples += 1

print("Actual number of samples:", actualNumberOfSamples)
print("Actual number of incomplete samples:", actualNumberOfIncompleteSamples)
