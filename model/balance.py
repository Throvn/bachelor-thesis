import json
import os
import random
import pandas as pd

# Function to check balance of active and inactive entries
def check_balance(js):
    ones = sum([sum(entry['isActive']) for entry in js])
    zeros = sum([len(entry['isActive']) - sum(entry['isActive']) for entry in js])
    return [ones, zeros]

DATA_FILE_NAME = "../classifiedDAOs.json"
js = json.load(open(DATA_FILE_NAME))

WINDOW_SIZE = 64

random.seed(1337)

# Function to filter entries with minimum WINDOW_SIZE and balance the dataset
def count(js, minority=False):
    finalActive = []
    finalInActive = []
    for entry in js:
        if len(entry['isActive']) < WINDOW_SIZE:
            print("Skipping...", entry['slug'])
            continue
        if entry['isActive'][-1] == 0:
            finalInActive.append(entry)
        elif entry['isActive'][-1] == 1:
            finalActive.append(entry)

    print("Active:    ", len(finalActive), "\nInactive: ", len(finalInActive))
    shortest = min(len(finalActive), len(finalInActive))

    if minority:
        print("NOTE: 70% inactive")
        random.shuffle(finalActive)
        random.shuffle(finalInActive)
        print("Actual Inactive: ", len(finalInActive), "\nActual Active: ", int((len(finalInActive) * 3) / 7))
        final = finalInActive + finalActive[:int((len(finalInActive) * 3) / 7)]
    else:
        # Maybe could have shuffled here. but shouldn't make any difference.
        final = finalActive[:shortest] + finalInActive[:shortest]
    return final

# Apply the count function and create a balanced dataset
final = count(js, minority=True)

# Convert js to DataFrame and shuffle it
total_training_data = pd.DataFrame(js).sample(frac=1, random_state=1337).reset_index(drop=True)
print("Done.") 

# Display the total count of entries
total_entries = len(total_training_data)
print("Total training Data: ", total_entries)

# Split total_training_data into training and testing datasets (70/30)
split_index = int(total_entries * 0.7)
grouped_train = total_training_data.iloc[:split_index]
grouped_test = total_training_data.iloc[split_index:]

print("TRAIN:")
count(grouped_train.to_dict('records'))  # Convert DataFrame to list of dicts

print("TEST:")
count(grouped_test.to_dict('records'))  # Convert DataFrame to list of dicts

# Display size of training and test sets
print("Size of training set: ", len(grouped_train))
print("Size of test set: ", len(grouped_test))

# json.dump(final, open("./minorityDataset.json", "w"))