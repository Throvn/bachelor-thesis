from test import all_y_true, all_y_pred
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Precision-Recall and AUC Calculation for Our Model
precision, recall, _ = precision_recall_curve(all_y_true, all_y_pred)
pr_auc = auc(recall, precision)  # PR-AUC using the trapezoidal rule
average_precision = average_precision_score(all_y_true, all_y_pred)

# Plot PRC for Our Model
plt.rcParams["font.size"] = "21"
plt.figure(figsize=(8, 8), dpi=300)
plt.plot(recall, precision, lw=2, color='darkorange', label=f'Model: PRC (PR-AUC = {pr_auc:.2f}, AP = {average_precision:.2f})')

# Precision-Recall and AUC Calculation for Majority Class Prediction
baseline_preds = [1 for _ in all_y_pred]
precision, recall, _ = precision_recall_curve(all_y_true, baseline_preds)
baseline_pr_auc = auc(recall, precision)  # PR-AUC for the baseline
baseline_average_precision = average_precision_score(all_y_true, baseline_preds)

plt.plot(recall, precision, lw=2, color='navy', linestyle='--', label=f'MCP: PRC (PR-AUC = {baseline_pr_auc:.2f}, AP = {baseline_average_precision:.2f})')

# Finalize Plot
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid()
plt.subplots_adjust(wspace=0.5)
plt.savefig(fname="prc.pdf")
