import json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

from preparation import create_sequences

# Constants
WINDOW_SIZE = 60
TIMESERIES_SPLITS = 3
MIN_DATA_OBSERVATIONS = WINDOW_SIZE + (TIMESERIES_SPLITS + 1)
DATA_FILE_NAME = "../classifiedDAOs.json"

# Load data
total_training_data = json.load(open(DATA_FILE_NAME))
total_training_data = pd.DataFrame(total_training_data).sample(frac=1, random_state=1337).reset_index(drop=True)

# Function to accumulate sequences
def collect_sequences():
    X_all, y_all = [], []
    for _, observation in total_training_data.iterrows():
        observation: pd.Series

        print("\tCreating sequence for '" + observation.slug + "'... ", end="")
        if len(observation.priceUSD) < MIN_DATA_OBSERVATIONS:
            print("Skipped.")
            continue

        X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
        
        # Collect sequences
        X_all.extend(X_seq)  # Append sequences to list
        y_all.extend(y_seq)  # Append labels to list

        print("Done.")
    return X_all, y_all

# Collect all sequences and labels
X, y = collect_sequences()
X = pd.DataFrame(X)  # Convert to DataFrame if necessary
y = pd.Series(y)     # Convert to Series if necessary

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the new class distribution after SMOTE
print(f"Class distribution before SMOTE: {y_train.value_counts()}")
print(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts()}")

# Train a model (RandomForest in this case) on the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
