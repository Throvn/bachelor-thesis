import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import xgboost as xgb

from typing import Final, Sequence, Any
from datetime import datetime
import json
print(torch.__version__)
script_start = datetime.now()

def print(*args, end="\n", flush=False):
    with open('./logs/' + str(script_start) + '.log', "a") as f:
        for arg in args:
            f.write(str(arg))
        f.write(end)
    __builtins__["print"](*args, end=end, flush=flush)


print("\n\n")
print("-" * 50)
print("Starting lstm.py at '" + str(script_start) + "'.")


# CONSTANTS
DATA_FILE_NAME: Final[str] = "./isDaoActiveData(4).json"

# Is this how many 'days' the time series should be predicted?
WINDOW_SIZE: Final[int] = 60 # TODO: 120
TIMESERIES_SPLITS: Final[int] = 3

MIN_DATA_OBSERVATIONS: Final[int] = WINDOW_SIZE + (TIMESERIES_SPLITS + 1)

print("Reading '" + DATA_FILE_NAME + "'... ", end="", flush=True)
total_training_data = json.load(open(DATA_FILE_NAME))
total_training_data = pd.DataFrame(total_training_data).sample(frac=1, random_state=1337).reset_index(drop=True)
print("Done.") 

total_entries = len(total_training_data)
print("Total training Data: ", total_entries)

# Split total_training_data into training and testing datasets (70/30)
split_index = int(total_entries * 0.7)
grouped_train = total_training_data[:split_index]
grouped_test = total_training_data[split_index:]
print("Size of training set: ", len(grouped_train))
print("Size of test set: ", len(grouped_test))

##1 Init function for TimeSeriesSplit (ensures robustness of prediction results, by implementing 'granular' training)

def create_sequences(observation: pd.Series, window_size: int):
    X_seq, y_seq = [], []

    # Convert each subcategory of observation into a DataFrame, indexed by datetime
    price_df = pd.DataFrame(observation['priceUSD']).rename(columns={'value': 'priceUSD'})
    dev_df = pd.DataFrame(observation['devActivity']).rename(columns={'value': 'devActivity'})
    twitter_df = pd.DataFrame(observation['twitterFollowers']).rename(columns={'value': 'twitterFollowers'})
    
    # Handle empty DataFrames by creating a 'datetime' column if necessary
    if price_df.empty or 'datetime' not in price_df.columns:
        print("Price data is missing or empty.")
        return np.array([]), np.array([])

    # Ensure 'datetime' column exists in dev_df and twitter_df, or else add it as NaN for alignment
    if dev_df.empty:
        dev_df = pd.DataFrame({'datetime': price_df['datetime'], 'devActivity': [np.nan] * len(price_df)})
    if twitter_df.empty:
        twitter_df = pd.DataFrame({'datetime': price_df['datetime'], 'twitterFollowers': [np.nan] * len(price_df)})

    # The isActive is already aligned with priceUSD, so no need to merge on datetime
    isActive = observation['isActive']

    # print(dev_df)
    # Merge the dataframes on datetime
    df = price_df.merge(dev_df, on='datetime', how='outer') \
                 .merge(twitter_df, on='datetime', how='outer')

    # Sort by datetime to ensure proper sequence
    df = df.sort_values(by='datetime').reset_index(drop=True)

    # Fill any missing values with interpolation for the features
    df[['priceUSD', 'devActivity', 'twitterFollowers']] = df[['priceUSD', 'devActivity', 'twitterFollowers']].interpolate(method='linear')

    # Ensure we have enough data points for the specified window size
    if len(df) < window_size:
        print(f"Insufficient data for window size {window_size}. Skipping.")
        return np.array([]), np.array([])

    # Iterate and create sequences
    # Limit to 3000 to not run out of mps space.
    for i in range(min(len(isActive) - window_size, 3000)):
        # Extract the window of features
        X_window = df[['priceUSD', 'devActivity', 'twitterFollowers']].iloc[i:(i + window_size)].values
        X_seq.append(X_window)

        # print("Lengths:", len(isActive), i + window_size - 1, len(df['priceUSD']))
        # Extract the target value for the end of the window from isActive array
        y_seq.append(isActive[i + window_size - 1])

    # Convert lists to numpy arrays
    X_seq = np.asarray(X_seq)
    y_seq = np.asarray(y_seq)

    return X_seq, y_seq

# TODO: GET THE FUCKING LSTM TO WORK AGAIN. FASTER THAN BEFORE. AND IMPLEMENT SAVING OF THE MODEL!

##2 define tensor structure
print("Preparing data for training...") 
num_training_observations = 0
num_testing_observations = 0

class TrainingDataset(torch.utils.data.Dataset):
    def _check_lengths(self, row):
        return len(row['twitterFollowers'] or "") >= WINDOW_SIZE and len(row['priceUSD'] or "") >= WINDOW_SIZE and len(row['devActivity'] or "") >= WINDOW_SIZE

    def __init__(self, transform=None, target_transform=None):
        self.data = pd.read_json(DATA_FILE_NAME)
        self.data = self.data[self.data.apply(self._check_lengths, axis=1)]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        global num_training_observations
        observation = self.data.iloc[idx]
        print("\tCreating sequence for '" + observation.slug + "'... ", end="")
        if len(observation.priceUSD) < MIN_DATA_OBSERVATIONS:
            print("Skipped.")
            return np.array([]), np.array([])

        X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
        # print(X_seq)
        num_training_observations += 1
        print("Done.")

        return X_seq.squeeze(), y_seq.squeeze()
        # print("  " + str(y_seq) + "  ", end="")

training_data = TrainingDataset()
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True)
print(train_dataloader)

# def trainIterator():
#     global num_training_observations
#     for index, observation in grouped_train.iterrows():
#         index: int
#         observation: pd.Series

#         print("\tCreating sequence for '" + observation.slug + "'... ", end="")
#         if len(observation.priceUSD) < MIN_DATA_OBSERVATIONS:
#             print("Skipped.")
#             continue

#         X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
#         # print(X_seq)
#         yield X_seq, y_seq
#         # print("  " + str(y_seq) + "  ", end="")
#         num_training_observations += 1
#         print("Done.")

print("Preparing data for testing...") 
# Check if a GPU is available
device = torch.device('mps')

# TODO: Figure out what we need this for. Currently unused.
# def testIterator():
#     global num_testing_observations
#     for index, observation in grouped_test.iterrows():
#         index: int
#         observation: pd.Series

#         print("\tCreating sequence for '" + observation.slug + "'... ", end="")
#         if len(observation.priceUSD) < MIN_DATA_OBSERVATIONS:
#             print("Skipped.")
#             continue

#         X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
#         val_input = torch.tensor(X_seq, dtype=torch.float32, device=device)
#         val_target = torch.tensor(y_seq, dtype=torch.float32, device=device)
#         yield val_input, val_target
#         num_testing_observations += 1
#         print("Done.")

##3 Init Single Input LSTM

class SingleInputLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units=128):
        super(SingleInputLSTMClassifier, self).__init__()
        self.hidden_units = hidden_units
        self.lstm1 = nn.LSTM(input_dim, hidden_units, batch_first=True, bidirectional=True) #not bidirectional architecture
        self.dropout1 = nn.Dropout(0.5) #note Dropout
        self.relu1 = nn.ReLU() #note ReLU activation function
        self.lstm2 = nn.LSTM(hidden_units * 2, hidden_units, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3) #note Dropout
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_units * 2, 1)
        self.sigmoid = nn.Sigmoid() #note Sigmoid Output (min val: 0/ max val: 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        out = self.relu1(out)

        h1 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        c1 = torch.zeros(2, batch_size, self.hidden_units, device=x.device)
        out, _ = self.lstm2(out, (h1, c1))
        out = self.dropout2(out)
        out = self.relu2(out)

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out



##4 extract input_dim and initialize model. 
num_features = 3 # devActivity, twitterFollowers, priceUSD # input_streams[0].shape[2] #Input_dim = length of vector that contains unique feature columns.
input_dim=(num_features)

model = SingleInputLSTMClassifier(input_dim).to(device)

checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])


##5 Run the Model -> Gives output as classification report

# Parameters
print("Parameters:")
num_epochs = 30 # 60
print("\tNum epochs: ", num_epochs)
initial_lr = 0.001 # 0.005
print("\tInitial learning rate: ", initial_lr)
patience = 12
print("\tPatience: ", patience)
print("\tWindow size: ", WINDOW_SIZE)
print("\tTimeseries splits: ", TIMESERIES_SPLITS)
print("\tMinimum number of timeseries datapoints: ", MIN_DATA_OBSERVATIONS)

# Gradient boosting model
gbm_params = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    # TODO: Check that this value is actually true according to credible research.
    # 'scale_pos_weight': 0.8
}
print("Gradient boosting params: \n\t", json.dumps(gbm_params))
# TODO: Explain in thesis: https://stackoverflow.com/a/68368157
criterion = nn.BCEWithLogitsLoss()

# Training and validation
model.train()
train_losses = []
val_losses = []
avg_gradient_norms = []

# For tracking results
train_results = {}
val_results = {}

# Aggregated predictions and true labels for final evaluation
all_y_true = []
all_y_pred = []
all_y_prob = []

tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
# Grouped time series split
for name, (X, y) in enumerate(train_dataloader):
    # Bring tensor into the right shape for TimeSplit (from: (1, n_timesteps, n_windows, n_features), to: (n_samples, n_features))
    X = torch.nan_to_num(X, nan=0.0).squeeze().to(torch.float32).to(device)
    y = y.squeeze().to(torch.float32).to(device)
    # X = input_streams[name]
    # y = target_streams[name]
    splits = tscv.split(X.cpu().numpy())

    for split_id, (train_index, val_index) in enumerate(splits):
        torch.mps.empty_cache()
        print(f"Training group {name}, split {split_id + 1}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        # y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        # X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
        X_train_tensor = X_train
        y_train_tensor = y_train
        X_val_tensor = X_val
        y_val_tensor = y_val

        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_plateau = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1)
        scheduler_cyclic = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=10, mode='triangular2', cycle_momentum=False)
        early_stopping_counter = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            epoch_gradients = []

            # Training loop
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor.unsqueeze(1))
            loss.backward()

            # Collect gradients and calculate their norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** (1. / 2)
            epoch_gradients.append(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler_cyclic.step()
            epoch_train_loss += loss.item()

            # Validation loop
            model.eval()
            with torch.no_grad():
                output = model(X_val_tensor)
                val_loss = criterion(output, y_val_tensor.unsqueeze(1)).item()
                epoch_val_loss += val_loss

            avg_train_loss = epoch_train_loss / len(train_index)
            avg_val_loss = epoch_val_loss / len(val_index)
            avg_gradient_norm = sum(epoch_gradients) / len(epoch_gradients)

            # Track losses and gradients
            if (name, split_id) not in train_results:
                train_results[(name, split_id)] = []
                val_results[(name, split_id)] = []

            train_results[(name, split_id)].append(avg_train_loss)
            val_results[(name, split_id)].append(avg_val_loss)
            avg_gradient_norms.append(avg_gradient_norm)

            # Step the scheduler based on the validation loss
            scheduler_plateau.step(avg_val_loss)

            # Print epoch summary
            __builtins__['print'](f'Group {name}, Split {split_id + 1}, Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, '
                  f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        # Gradient Boosting
        model.eval()
        with torch.no_grad():
            train_output = model(X_train_tensor).cpu().numpy()
            val_output = model(X_val_tensor).cpu().numpy()
        
        print("\tY Train Labels:", np.unique(y_train_tensor.cpu().numpy(), return_counts=True))
        print("\tY Val Labels:", np.unique(y_val_tensor.cpu().numpy(), return_counts=True))

        # Fit gradient boosting model
        gbm = xgb.XGBRegressor(**gbm_params)
        gbm.fit(train_output, torch.round(y_train_tensor).cpu().numpy().astype(int), eval_set=[(val_output, torch.round(y_val_tensor).cpu().numpy().astype(int))], verbose=True)
        val_pred = gbm.predict(val_output)
        
        # Aggregate predictions and true labels
        all_y_true.extend(y_val_tensor.cpu().numpy())
        all_y_pred.extend(val_pred)
        
        #predict probabilities for ROC AUC
        val_prob = gbm.predict(val_output) # [:, 1]  # Get probabilities for the positive class
        all_y_prob.extend(val_prob)

print("Actual training set: ", num_training_observations, " entries")
print("Actual testing set:  ", num_testing_observations, " entries")
print("Actual training/total ratio: ", num_training_observations / len(total_training_data))


# Print final classification report for all splits and groups
print("Final Classification Report with Gradient Boosting:")

# Set a threshold (you can experiment with different values)
threshold = 0.5

# Convert continuous predictions to binary based on the threshold
all_y_pred_class = [int(pred >= threshold) for pred in all_y_pred]

# Now use classification_report
print(classification_report(all_y_true, all_y_pred_class, target_names=["Abandoned", "Operating"]))
print("Finished execution at '", str(datetime.now()), "'.")
print("Execution took: ", (datetime.now() - script_start).total_seconds())
print("-" * 50)