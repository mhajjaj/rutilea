import torch
# from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import confusion_matrix


# Create a MulticlassConfusionMatrix instance
# metric = MulticlassConfusionMatrix(num_classes=5)

# Simulate some example predictions and targets as lists
# Replace these with your actual predictions and targets
predictions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
targets = [0, 1, 2, 3, 4, 1, 2, 3, 4, 0]

# Convert the lists to PyTorch tensors
predictions_tensor = torch.tensor(predictions)
targets_tensor = torch.tensor(targets)

# Update the confusion matrix with tensors
metric.update(predictions_tensor, targets_tensor)

# Compute the confusion matrix and other metrics
confusion_matrix = metric.compute()

correct_predictions = torch.diag(confusion_matrix).sum().item()
total_predictions = confusion_matrix.sum().item()
accuracy = correct_predictions / total_predictions

# Compute the confusion matrix as before
confusion_matrix = metric.compute()

# Calculate precision, recall, and F1-score for each class
num_classes = confusion_matrix.size(0)
precision = torch.zeros(num_classes)
recall = torch.zeros(num_classes)
f1_score = torch.zeros(num_classes)

for i in range(num_classes):
    true_positives = confusion_matrix[i, i]
    false_positives = confusion_matrix[:, i].sum() - true_positives
    false_negatives = confusion_matrix[i, :].sum() - true_positives
    
    # Calculate precision, recall, and F1-score, handling zero denominators
    precision[i] = true_positives / max((true_positives + false_positives), 1e-12)
    recall[i] = true_positives / max((true_positives + false_negatives), 1e-12)
    f1_score[i] = 2 * (precision[i] * recall[i]) / max((precision[i] + recall[i]), 1e-12)

# Print precision, recall, and F1-score for each class
for i in range(num_classes):
    print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}, F1-Score={f1_score[i]}")


# import torch
# from ultralytics import NAS

# print(torch.cuda.is_available())

# #model = NAS('yolo_nas_s.pt')
# #model.info()
# #results = model.val(data='coco8.yaml')
# #results = model('mhajjaj\predictImgs\Scratches\year=2023-month=04-day=01-04_36_34-NG-2_1.png')
# import os
# import cv2

# file_path = 'predictImgs\\Scratches\\year=2023-month=04-day=01-04_36_34-NG-2_1.png'

# if os.path.exists(file_path):
#     # File exists; you can proceed with your operations.
#     print(f"OK")
#     from super_gradients.training import models
#     device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#     model=models.get('yolo_nas_s',pretrained_weights="coco").to(device)
#     #out=model.predict(file_path)
    
#     img = cv2.imread(file_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     out=model.predict(img)
#     out.show()
# else:
#     print(f"File not found: {file_path}")
