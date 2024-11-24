from collections import Counter
from itertools import count
import json
import pandas as pd

WINDOW_SIZE = 64
DATA_FILE_NAME = "../preprocessing/allClassifications.json"

print("Reading '" + DATA_FILE_NAME + "'... ", end="", flush=True)
js = json.load(open(DATA_FILE_NAME))

# Convert js to DataFrame and shuffle it
total_training_data = pd.DataFrame(js).sample(frac=1, random_state=1337).reset_index(drop=True)

# Display the total count of entries
total_entries = len(total_training_data)
print("Done.")

def countOccurrences(js):
    finalCount = Counter()
    for observation in js:
        c = Counter(observation["isActive"])
        finalCount.update(c)
    return finalCount

# Function to filter entries with minimum WINDOW_SIZE and balance the dataset
def countSamples(js):
    finalActive = []
    finalInActive = []
    for entry in js:
        if len(entry['isActive']) < WINDOW_SIZE:
            raise ValueError(entry['slug'] + " has less than WINDOW_SIZE entries in 'isActive'")

        if entry['isActive'][-1] == 0:
            finalInActive.append(entry)
        elif entry['isActive'][-1] == 1:
            finalActive.append(entry)

    return {
        "active": finalActive,
        "inactive": finalInActive
    }

def summary(data, title = "<NO TITLE>"):
    result = countSamples(data)
    print(title)
    print("\tActive:   ", len(result['active']), "\n\tInactive: ", len(result['inactive']))
    print("\t" + "-" * 14)
    total_entries = len(data)
    print("\tTotal:   ", total_entries)
    ratio = len(result['active']) / total_entries
    print("\tBias:", ratio, "\n")
    if (len(result['active']) + len(result["inactive"])) != total_entries:
        raise ValueError("Entries went missing")
    

# for i in range(10):
# 	print("NOTE: DOING A 50/50 (active/inactive) SPLIT")
# data = countSamples(total_training_data.to_dict('records'))
# finalActive = data["active"]
# finalInActive = data["inactive"]
# min_size = min(len(finalActive), len(finalInActive))
# total_training_data = finalActive[:min_size] + finalInActive[:min_size]
# total_training_data = pd.DataFrame(total_training_data).sample(frac=1, random_state=1337).reset_index(drop=True)
# total_entries = min_size * 2


# Split total_training_data into training and testing datasets (80/20)
split_index = int(total_entries * 0.8)
grouped_train = total_training_data.iloc[:split_index]
grouped_test = total_training_data.iloc[split_index:]
countedOccurrences = countOccurrences(js)

if __name__ == '__main__':
    print("Total observations:", countedOccurrences)
    print("Inactive observations: ", countedOccurrences[0] / countedOccurrences.total())
    summary(js, "'./allClassifications.json' balance:")
    summary(grouped_test.to_dict('records'), "TEST:")
summary(grouped_train.to_dict('records'), "TRAIN:")
