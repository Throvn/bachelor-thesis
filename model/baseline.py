import sys
import numpy as np
import pandas as pd
import random
from sklearn.metrics import classification_report
from collections import Counter
from balance import grouped_train
import os
from test import MODEL_SAVE_PATH

# Set seed for reproducibility
random.seed(1337)
np.random.seed(1337)

cacheExists = os.path.exists("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy") and os.path.exists("./cache/" + MODEL_SAVE_PATH + "-all_y_pred.npy")
if cacheExists:
	print("Cache exists... Skipping repeated classification.")
	all_y_true = np.asarray(np.load("./cache/" + MODEL_SAVE_PATH + "-all_y_true.npy"))
	all_y_pred = np.asarray(np.load("cache/" + MODEL_SAVE_PATH + "-all_y_pred.npy"))
	num_testing_observations = len(all_y_pred)

# Convert continuous predictions to binary based on the threshold
all_y_pred_class = [1 for pred in all_y_pred]

# Now use classification_report
print(classification_report(all_y_true, all_y_pred_class, target_names=["Abandoned", "Operating"]))

from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt


# Print AUC

if __name__ == "__main__":
    plt.rcParams["font.size"] = "21"
    plt.figure("Our Model (left) vs. Majority Class Prediction (right)", figsize=(20, 8))

    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred)
    roc_auc = auc(fpr, tpr)
    print(f"NEW: Area Under the Curve (AUC): {roc_auc:.2f}")

    ax = plt.subplot(1, 2, 1)
    ax.set_aspect('equal', adjustable='box')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: Our Model')
    plt.legend(loc="lower right")

    # Calculate mcp
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred_class)
    roc_auc = auc(fpr, tpr)
    print(f"OLD: Area Under the Curve (AUC): {roc_auc:.2f}")

    ax = plt.subplot(1, 2, 2)
    ax.set_aspect('equal', adjustable='box')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: Majority Class Prediction')
    plt.legend(loc="lower right")

    plt.subplots_adjust(wspace=0.5)
    plt.savefig(fname='./roc_auc.pdf')