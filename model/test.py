import os
import random
from sklearn.metrics import classification_report
import torch
import numpy as np
from model import SingleInputLSTMClassifier
from preparation import create_sequences
from balance import grouped_test

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CONSTANTS
DATA_FILE_NAME = "../preprocessing/allClassifications.json"
MODEL_SAVE_PATH = "./unidirectional_focal_model_full_correct_a0.29"
WINDOW_SIZE = 64
# Set to true, if you only want to call the final predictions. Not all predictions.
ONLY_LAST = True
print(MODEL_SAVE_PATH)


# Load model checkpoint if available
if not os.path.exists(MODEL_SAVE_PATH):
	raise ValueError("Path to Model file doesn't exist")
checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=True) 

device = torch.device('mps')

# Define the model
num_features = 3  # devActivity, twitterFollowers, priceUSD
input_dim = num_features
model = SingleInputLSTMClassifier(input_dim, isBidirectional="bidirectional" in MODEL_SAVE_PATH).to(device)

if checkpoint:
	model.load_state_dict(checkpoint['model_state_dict'])

# Prepare the test data
num_testing_observations = 0
all_y_true = []
all_y_pred = []

cacheExists = os.path.exists("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy") and os.path.exists("./cache/" + MODEL_SAVE_PATH + "-all_y_pred.npy")
if cacheExists:
	print("Cache exists... Skipping repeated classification.")
	all_y_true = np.asarray(np.load("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy"))
	all_y_pred = np.asarray(np.load("cache/" + MODEL_SAVE_PATH + "-all_y_pred.npy"))
	num_testing_observations = len(all_y_pred)
else:
	for index, observation in grouped_test.iterrows():

		X_seq, y_seq = create_sequences(observation, WINDOW_SIZE)
		if not len(X_seq) or not len(y_seq):
			continue
		X = torch.tensor(X_seq.squeeze().squeeze(), device=device, dtype=torch.float32)
		X = torch.nan_to_num(X, nan=0.0)
		y = torch.tensor(y_seq, device=device, dtype=torch.float32)

		with torch.no_grad():
			try:
				model.eval()
				output = model(X)
				num_testing_observations += 1

				if ONLY_LAST:
					all_y_true.append(y.cpu().numpy()[-1])
					all_y_pred.append(output.cpu().numpy()[-1])
				else:
					all_y_true.extend(y.cpu().numpy())
					all_y_pred.extend(output.cpu().numpy())
				# breakpoint()
			except:
				print("Skipping '" + observation.slug + "' because of mismatching shapes.")

		print("\n",str(np.average(output.cpu().numpy())) + " " + observation.slug, end="")

	np.save("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy", all_y_true)
	np.save("./cache/" + MODEL_SAVE_PATH + "-all_y_pred.npy", all_y_pred)
	print("Wrote results to cache.")

# Set a threshold for classification
threshold = 0.94
all_y_pred_class = [int(pred >= threshold) for pred in all_y_pred]

# Generate the classification report
print("Final Classification Report:")
print("File: ", MODEL_SAVE_PATH)
print("Len Y:", len(all_y_true), "Len PRED Y:", len(all_y_pred_class))
print(classification_report(all_y_true, all_y_pred_class, target_names=["Abandoned", "Operating"]))
print(f"Threshold: {threshold}")

print(f"Evaluated on {num_testing_observations} test observations.")

###################################
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# ROC and AUC Calculation
fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred)
roc_auc = auc(fpr, tpr)

# Print AUC
print(f"Area Under the Curve (AUC): {roc_auc:.2f}")

if __name__ == "__main__":
# Plot the ROC curve
	plt.figure(MODEL_SAVE_PATH + " thresh=" + str(threshold) + " obs=" + str(num_testing_observations))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	plt.show()