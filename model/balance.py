import json
import pandas as pd
def check_balance(js):
    for row in js:
        isActive = row['isActive']
        ones = sum(isActive)  # Count the number of 1s (active)
        zeros = len(isActive) - ones  # Count the number of 0s (inactive)


    ones_perc = (ones / len(isActive)) * 100
    zero_perc = (zeros / len(isActive)) * 100

    return [ones, zeros]


DATA_FILE_NAME = "../classifiedDAOs.json"
js = json.load(open(DATA_FILE_NAME))
finalActive = []
finalInActive = []

maxEntries = check_balance(js)
for entry in js:
    if len(entry['isActive']) < 1:
          print("Skipping...", entry['slug'])
          continue
    if entry['isActive'][-1] == 0:
            finalInActive.append(entry)
    elif entry['isActive'][-1] == 1:
            finalActive.append(entry)

print(len(finalActive), len(finalInActive))
shortest = min(len(finalActive), len(finalInActive))
final = finalActive[:shortest] + finalInActive[:shortest]
print(len(final))
json.dump(final, open("balancedDataset.json", "w"))