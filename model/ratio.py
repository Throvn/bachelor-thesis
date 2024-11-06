
import json
import random
from typing import Final
import numpy as np
import pandas as pd
import torch
from collections import Counter


torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CONSTANTS
DATA_FILE_NAME: Final[str] = "../classifiedDAOs.json"

# Is this how many 'days' the time series should be predicted?
WINDOW_SIZE: Final[int] = 60 # TODO: 120
TIMESERIES_SPLITS: Final[int] = 3

MIN_DATA_OBSERVATIONS: Final[int] = WINDOW_SIZE + (TIMESERIES_SPLITS + 1)

print("Reading '" + DATA_FILE_NAME + "'... ", end="", flush=True)
total_training_data_json = json.load(open(DATA_FILE_NAME))
total_training_data = pd.DataFrame(total_training_data_json).sample(frac=1, random_state=1337).reset_index(drop=True)
print("Done.") 

overallCount = Counter()
for i, d in total_training_data.iterrows():
    overallCount.update(Counter(d['isActive']))

print("# DAOs:", len(total_training_data_json))
print("DEF DAOs:", len(json.load(open("../preprocessing/definitelyDaos.json"))))
print("Enriched Coins:", len(json.load(open("../preprocessing/enrichedCoins.json"))))
print("Counts:", overallCount)
print("Ratio:", overallCount[0] / (overallCount[1] + overallCount[0]))