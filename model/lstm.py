import numpy as np
import pandas as pd
import torch
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import xgboost as xgb

from typing import Final
from datetime import datetime
import json
import os

# import matplotlib.pyplot as plt

from model import SingleInputLSTMClassifier
from focal_loss import FocalLoss
from preparation import create_sequences
from balance import DATA_FILE_NAME, grouped_train, countedOccurrences
script_start = datetime.now()

# def print(*args, end="\n", flush=False):
#     with open('./logs/' + str(script_start) + '.log', "a") as f:
#         for arg in args:
#             f.write(str(arg))
#         f.write(end)
#     __builtins__.print(*args, end=end, flush=flush)


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
daosTrainedOn = checkpoint['daosTrainedOn'] if checkpoint else 0
print("DAOs trained so far:", daosTrainedOn)

# Split total_training_data into training and testing datasets (70/30)
grouped_train = grouped_train[daosTrainedOn:]

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


print("Preparing data for testing...") 
# Check if a GPU is available
device = torch.device('mps')





##4 extract input_dim and initialize model. 
num_features = 3 # devActivity, twitterFollowers, priceUSD # input_streams[0].shape[2] #Input_dim = length of vector that contains unique feature columns.
input_dim=(num_features)

model = SingleInputLSTMClassifier(input_dim, isBidirectional=IS_BIDIRECTIONAL).to(device)

if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])


##5 Run the Model -> Gives output as classification report

# Parameters
print("Parameters:")
num_epochs = 60
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
}
print("Gradient boosting params: \n\t", json.dumps(gbm_params))
# TODO: Explain in thesis: https://stackoverflow.com/a/68368157
# criterion = nn.BCELoss()

percentageMinority = countedOccurrences.most_common()[-1][1] / countedOccurrences.total()
print("Percentage majority class:", percentageMinority)
criterion = FocalLoss(alpha=percentageMinority, gamma=3)
# criterion = nn.BCEWithLogitsLoss()

# Training and validation
model.train()

# For tracking results
train_results = {}
val_results = {}

# Aggregated predictions and true labels for final evaluation
all_y_true = []
all_y_pred = []

# Plotting
# plt.ion()
# fig, ax = plt.subplots(figsize=(10, 6))
# train_line, = ax.plot([], [], label='Training Loss')  # Empty plot for training loss
# val_line, = ax.plot([], [], label='Validation Loss')  # Empty plot for validation loss
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.set_title('Training and Validation Loss Over Time')
# ax.legend()

train_losses = []
val_losses = []

tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
# Grouped time series split
for name, (X, y) in enumerate(trainIterator()):
    if X.size <= 3 or y.size <= 3:
        print(X.size, y.size)
        continue
    # Bring tensor into the right shape for TimeSplit (from: (1, n_timesteps, n_windows, n_features), to: (n_samples, n_features))
    # Move to tensors to gpu if possible
    X = torch.tensor(X.squeeze().squeeze(), dtype=torch.float32, device=device)
    torch.nan_to_num_(X, nan=0.0)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    torch.nan_to_num_(y, nan=0.0)
    splits = tscv.split(X.cpu().numpy())

    for split_id, (train_index, val_index) in enumerate(splits):
        torch.mps.empty_cache()
        print(f"Training group {name}, split {split_id + 1}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_tensor = X_train
        y_train_tensor = y_train
        X_val_tensor = X_val
        y_val_tensor = y_val

        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        if checkpoint:
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

            # Track losses and gradients
            if (name, split_id) not in train_results:
                train_results[(name, split_id)] = []
                val_results[(name, split_id)] = []

            train_results[(name, split_id)].append(avg_train_loss)
            val_results[(name, split_id)].append(avg_val_loss)

            # Step the scheduler based on the validation loss
            scheduler_plateau.step(avg_val_loss)

            # Print epoch summary
            __builtins__.print(f'Group {name}, Split {split_id + 1}, Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, '
                  f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # Plotting: Append the loss values to track them over time
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

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


        # Update the plot data
        # train_line.set_data(range(len(train_losses)), train_losses)
        # val_line.set_data(range(len(val_losses)), val_losses)
        # ax.relim()  # Recalculate limits
        # ax.autoscale_view()  # Autoscale the view to see all data
        # plt.draw()
        # plt.pause(0.1)  # Small pause to update the plot
    
    daosTrainedOn += 1
    torch.save({
        'daosTrainedOn': daosTrainedOn,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'inputFileName': DATA_FILE_NAME,
        'optimizerName': str(optimizer),
        'wasBidirectional': model.isBidirectional,
        'windowSize': WINDOW_SIZE,
        'timeseriesSplits': TIMESERIES_SPLITS,
    }, MODEL_SAVE_PATH)


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


# plt.ioff()
# plt.show(block=False)
