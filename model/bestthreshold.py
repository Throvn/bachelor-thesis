from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from test import all_y_true, all_y_pred, MODEL_SAVE_PATH

# Assuming `all_y_true` contains the true labels and `all_y_pred` contains predicted probabilities.

# Define thresholds
thresholds = np.arange(0.0, 1.01, 0.01)

# Initialize variables to track the best threshold
best_threshold = 0
best_accuracy = 0

# Store accuracy scores for visualization
accuracy_scores = []

# Loop through thresholds to compute accuracy
for threshold in thresholds:
    # Convert probabilities to binary predictions using the threshold
    all_y_pred_class = [int(pred >= threshold) for pred in all_y_pred]
    
    # Calculate accuracy
    accuracy = accuracy_score(all_y_true, all_y_pred_class)
    accuracy_scores.append(accuracy)
    
    # Update best threshold if accuracy improves
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

# Print the best threshold and accuracy
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best Accuracy: {best_accuracy:.2f}")

# Plot accuracy vs. threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracy_scores, label='Accuracy', color='blue')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Threshold')
plt.legend()
plt.grid()
plt.show()

# Generate final classification report using the best threshold
final_y_pred_class = [int(pred >= best_threshold) for pred in all_y_pred]
print("\nFile:", MODEL_SAVE_PATH)
print("Final Classification Report:")
print(classification_report(all_y_true, final_y_pred_class, target_names=["Abandoned", "Operating"]))
