from test import all_y_true, all_y_pred
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Precision-Recall and Average Precision Calculation
precision, recall, prc_thresholds = precision_recall_curve(all_y_true, all_y_pred)
average_precision = average_precision_score(all_y_true, all_y_pred)

# Print Average Precision
print(f"Average Precision (AP): {average_precision:.2f}")

# Plot the PRC curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PRC curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid()
plt.show()