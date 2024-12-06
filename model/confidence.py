from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from test import all_y_true, all_y_pred
from heuristics import twitter_heur, twitter_true, dev_heur, dev_true, price_heur

from test import MODEL_SAVE_PATH

# Calculate PR curve and AUC for the original data
precision, recall, _ = precision_recall_curve(all_y_true, all_y_pred)
pr_auc = auc(recall, precision)

# Bootstrapping for confidence intervals
n_bootstraps = 1000  # Number of resamples

np.random.seed(1337)
rng_seed = 1337
rng = np.random.default_rng(rng_seed)

bootstrapped_aucs = []
precision_boot = []
recall_interp = np.linspace(0, 1, 100)  # Fixed recall points for interpolation

for i in range(n_bootstraps):
    # Bootstrap resampling with replacement
    indices = rng.choice(len(all_y_true), size=len(all_y_true), replace=True)
    y_true_boot = np.array(all_y_true)[indices]
    y_pred_boot = np.array(all_y_pred)[indices]

    # Ensure there are positive and negative samples in the bootstrap sample
    if len(np.unique(y_true_boot)) < 2:
        continue

    # Calculate PR curve and interpolate precision for fixed recall points
    precision_bootstrap, recall_bootstrap, _ = precision_recall_curve(y_true_boot, y_pred_boot)
    interp_precision = np.interp(recall_interp, recall_bootstrap[::-1], precision_bootstrap[::-1])
    precision_boot.append(interp_precision)

    # Calculate AUC for the bootstrap sample
    auc_bootstrap = auc(recall_bootstrap, precision_bootstrap)
    bootstrapped_aucs.append(auc_bootstrap)

# Convert precision results to numpy array for further analysis
precision_boot = np.array(precision_boot)

# Calculate mean and confidence intervals for precision-recall
mean_precision = np.mean(precision_boot, axis=0)
std_precision = np.std(precision_boot, axis=0)
precision_lower = np.maximum(mean_precision - 1.96 * std_precision, 0)  # 95% CI lower bound
precision_upper = np.minimum(mean_precision + 1.96 * std_precision, 1)  # 95% CI upper bound

# Calculate AUC confidence intervals
auc_lower = np.percentile(bootstrapped_aucs, 2.5)  # 2.5th percentile
auc_upper = np.percentile(bootstrapped_aucs, 97.5)  # 97.5th percentile

# Plot PR Curve with Confidence Intervals
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.fill_between(recall_interp, precision_lower, precision_upper, color='blue', alpha=0.2, 
                 label=f'95% CI (AUC = [{auc_lower:.2f}, {auc_upper:.2f}])')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Bootstrapped Confidence Interval')
plt.legend(loc="lower left")
plt.show()

# Print AUC with confidence intervals
print(f"PR AUC: {pr_auc:.2f}")
print(f"95% Confidence Interval for AUC: [{auc_lower:.2f}, {auc_upper:.2f}]")


###################################################################################################
# Heuristics: PR-AUC Confidence Intervals
###################################################################################################

import itertools
import random
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, classification_report
from sklearn.utils import resample
from balance import grouped_test
from preparation import create_sequences

torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CONSTANTS
WINDOW_SIZE = 64
BOOTSTRAP_SAMPLES = 1000  # Number of resamples for bootstrapping
CONFIDENCE_LEVEL = 0.95
print(MODEL_SAVE_PATH)

#################################
# Bootstrapping for PR AUC
def bootstrap_pr_auc(y_true, y_pred, num_samples=BOOTSTRAP_SAMPLES, conf_level=CONFIDENCE_LEVEL):
    aucs = []
    for _ in range(num_samples):
        # Resample the data with replacement
        indices = resample(range(len(y_true)))
        y_true_resampled = np.array(y_true)[indices]
        y_pred_resampled = np.array(y_pred)[indices]
        
        # Calculate PR AUC for the resampled data
        precision, recall, _ = precision_recall_curve(y_true_resampled, y_pred_resampled)
        aucs.append(auc(recall, precision))
    
    # Compute the confidence interval
    lower = np.percentile(aucs, (1 - conf_level) / 2 * 100)
    upper = np.percentile(aucs, (1 + conf_level) / 2 * 100)
    return np.mean(aucs), (lower, upper)

# Calculate PR AUC and confidence intervals for each heuristic
print("\nPrices USD:")
price_auc, price_ci = bootstrap_pr_auc(all_y_true, price_heur)
print(f"PR AUC: {price_auc:.2f}, 95% CI: {price_ci}")

print("\nDev Activity:")
dev_auc, dev_ci = bootstrap_pr_auc(dev_true, dev_heur)
print(f"PR AUC: {dev_auc:.2f}, 95% CI: {dev_ci}")

print("\nTwitter Followers:")
twitter_auc, twitter_ci = bootstrap_pr_auc(twitter_true, twitter_heur)
print(f"PR AUC: {twitter_auc:.2f}, 95% CI: {twitter_ci}")
