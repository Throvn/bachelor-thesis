import pandas as pd
def check_balance(df):
    for _, row in df.iterrows():
        isActive = row['isActive']
        ones = sum(isActive)  # Count the number of 1s (active)
        zeros = len(isActive) - ones  # Count the number of 0s (inactive)


    ones_perc = (ones / len(isActive)) * 100
    zero_perc = (zeros / len(isActive)) * 100

    return ones_perc, zero_perc

print(check_balance(pd.read_json("./isDaoActiveData(4).json")))