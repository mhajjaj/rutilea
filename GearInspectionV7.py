# Developed by m.hajjaj at rutilea
import os
import sys

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import seaborn as sn
import pandas as pd

import torch
from torchvision import transforms

import torch.nn as nn

import torch.optim as optim
from dataset import CustomImageDataset


from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

sys.stdout = sys.__stdout__
from variables import Config as config
from variables import Net


    
def createConfusionMatrix(loader):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in loader:
        output = Net(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    print(f"y_pred: {y_pred}")
    print("\v\v\v")
    print(f"y_true: {y_true}")

    # constant for classes -- config.CLASSES
    classes = config.CLASSES

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    # Create Heatmap
    plt.figure(figsize=(12, 7))
    plt.savefig('output1.png')
    return sn.heatmap(df_cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes).get_figure()
    #sn.heatmap(df_cm, annot=True).get_figure()

    
def createConfusionMatrixForEachClass(data_loader, model):
    # Define a custom function to create a confusion matrix for each class
    # Set the model in evaluation mode
    model.eval()
    
    # Initialize variables to store the confusion matrices
    num_classes = len(config.CLASSES)
    all_confusion_matrices = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Compute the confusion matrix for this batch
            batch_confusion_matrix = confusion_matrix(labels.numpy(), predicted.numpy(), labels=np.arange(num_classes))
            
            # Add the batch confusion matrix to the overall confusion matrix
            all_confusion_matrices += batch_confusion_matrix
    
    return all_confusion_matrices

##################################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if __name__ == '__main__':
 
    # Define your data transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Create custom datasets for train, validation, and test
    train_dataset = CustomImageDataset(
        images_dir=f'{config.DATA_DIR}\\{config.TRAIN_IMAGES_DIR}',
        labels_dir=config.TRAIN_LABELS_DIR,
        transform=transform
    )

    val_dataset = CustomImageDataset(
        images_dir=f'{config.DATA_DIR}\\{config.VAL_IMAGES_DIR}',
        labels_dir=config.VAL_LABELS_DIR,
        transform=transform
    )

    test_dataset = CustomImageDataset(
        images_dir=f'{config.DATA_DIR}\\{config.TEST_IMAGES_DIR}',
        labels_dir=config.TEST_LABELS_DIR,
        transform=transform
    )

# Define a DataLoader for your train dataset
# batch_size = 64  # Choose an appropriate batch size
    trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.DATALOADER_PARAMS['batch_size'], 
        shuffle=True, 
        num_workers=config.DATALOADER_PARAMS['num_workers'])

    valloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.DATALOADER_PARAMS['batch_size'], 
        shuffle=True, 
        num_workers=config.DATALOADER_PARAMS['num_workers'])

    testloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.DATALOADER_PARAMS['batch_size'], 
        shuffle=True, 
        num_workers=config.DATALOADER_PARAMS['num_workers'])

    createConfusionMatrix(train_dataset)
    sys.exit()
# exit()
# datasets
# Load your custom testing dataset
# testset = CustomDataset(
#     root_dir=config.DATA_DIR,
#     train=False,
#     transform=transform
# )

# dataloaders
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.DATALOADER_PARAMS['batch_size'],
#                                         shuffle=True, num_workers=config.DATALOADER_PARAMS['num_workers'])


# testloader = torch.utils.data.DataLoader(testset, batch_size=config.DATALOADER_PARAMS['batch_size'],
#                                         shuffle=False, num_workers=config.DATALOADER_PARAMS['num_workers'])

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# def createConfusionMatrixForEachClass(loader):
#     y_pred = []
#     y_true = []

#     # iterate over test data
#     for inputs, labels in loader:
#         output = net(inputs) # Feed Network

#         output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#         y_pred.extend(output) # Save Prediction
        
#         labels = labels.data.cpu().numpy()
#         y_true.extend(labels) # Save Truth

#     # constant for classes -- config.CLASSES
#     # classes = config.CLASSES

#     # Example dataset
#     true_labels = y_true # [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7]  # True class labels
#     predicted_labels = y_pred # [0, 1, 2, 2, 0, 1, 3, 3, 4, 5, 5, 6, 6, 7, 7]  # Predicted class labels
#     class_names = config.CLASSES

#     # Initialize dictionaries to store metrics for each class
#     confusion_matrices = {}
#     accuracies = {}
#     precisions = {}
#     recalls = {}
#     f1_scores = {}

#     # Loop through each class
#     for class_idx, class_name in enumerate(class_names):
#         # Create binary labels for the current class vs. all other classes
#         binary_true_labels = [1 if label == class_idx else 0 for label in true_labels]
#         binary_predicted_labels = [1 if label == class_idx else 0 for label in predicted_labels]

#         # Calculate confusion matrix for the current class
#         conf_matrix = confusion_matrix(binary_true_labels, binary_predicted_labels)
        
#         # Calculate performance metrics for the current class with zero_division=1.0
#         accuracy = accuracy_score(binary_true_labels, binary_predicted_labels)
#         precision = precision_score(binary_true_labels, binary_predicted_labels, zero_division=1.0)
#         recall = recall_score(binary_true_labels, binary_predicted_labels, zero_division=1.0)
#         f1 = f1_score(binary_true_labels, binary_predicted_labels, zero_division=1.0)
        
#         # Store the metrics in their respective dictionaries with the class name as the key
#         confusion_matrices[class_name] = conf_matrix
#         accuracies[class_name] = accuracy
#         precisions[class_name] = precision
#         recalls[class_name] = recall
#         f1_scores[class_name] = f1

#     # Plot confusion matrices for all classes
#     fig, axes = plt.subplots(2, 4, figsize=(16, 8))
#     for i, class_name in enumerate(class_names):
#         row, col = divmod(i, 4)
#         ax = axes[row, col]
        
#         im = ax.imshow(confusion_matrices[class_name], cmap='Blues', interpolation='nearest')
#         ax.set_title(f"Class {class_name} Confusion Matrix")
        
#         # Customize tick labels based on your class names
#         tick_marks = np.arange(2)  # Two classes: 0 and 1
#         ax.set_xticks(tick_marks)
#         ax.set_yticks(tick_marks)
#         ax.set_xticklabels([f'Predicted {class_name}', f'Predicted Other'])
#         ax.set_yticklabels([f'True {class_name}', f'True Other'])

#         for i in range(2):
#             for j in range(2):
#                 ax.text(j, i, str(confusion_matrices[class_name][i, j]), ha="center", va="center", color="black")

#     plt.tight_layout()
#     plt.show()
#     plt.savefig('output2.png')

#     # Print metrics for each class
#     for class_name in class_names:
#         print(f"Metrics for Class {class_name}:")
#         print("Accuracy:", accuracies[class_name])
#         print("Precision:", precisions[class_name])
#         print("Recall:", recalls[class_name])
#         print("F1 Score:", f1_scores[class_name])
#         print()

#     return sn.heatmap(conf_matrix, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names).get_figure()

    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

    # Create a custom dataset using your data directory and the defined transformations
    train_dataset = CustomImageDataset(
        images_dir=f'{config.DATA_DIR}\\{config.TRAIN_IMAGES_DIR}',
        labels_dir=config.TRAIN_LABELS_DIR,
        transform=transform
    )

    # Create a data loader using the dataset
    trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.DATALOADER_PARAMS['batch_size'], 
        shuffle=True, 
        num_workers=config.DATALOADER_PARAMS['num_workers'])
    
    # Define your data transformations
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 640)),
        transforms.ToTensor()
    ])

    #########################################
    # Get a single batch from the trainloader
    for batch_idx, (images, labels) in enumerate(trainloader):
        # Check data type of images and labels in the batch
        image_data_type = images.dtype
        label_data_type = labels.dtype
        
        # Print data types
        print(f"Batch {batch_idx + 1}:")
        print(f"Image data type: {image_data_type}")
        print(f"Label data type: {label_data_type}")
        
        # Break after checking the first batch (you can check more if needed)
        break

    os.makedirs(config.LOGS, exist_ok=True)
    writer = SummaryWriter(config.LOGS)

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TRAIN_IMAGES_DIR,
            'labels_dir': config.TRAIN_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.VAL_IMAGES_DIR,
            'labels_dir': config.VAL_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    # print(f"DATA_DIR: {config.DATA_DIR} | \v TEST_IMAGES_DIR: {config.TEST_IMAGES_DIR} | \v TEST_LABELS_DIR: {config.TEST_LABELS_DIR}")
    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TEST_IMAGES_DIR,
            'labels_dir': config.TEST_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    train_data.dataset.plot()

    model = models.get(config.MODEL_NAME,
                       num_classes=config.NUM_CLASSES,
                       pretrained_weights=config.PRETRAINED_WEIGHTS
                       )

    train_params = {
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.001,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": config.EPOCHS, # Change No of epochs you want, more is better until a level
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=config.NUM_CLASSES,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=config.NUM_CLASSES,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    running_loss = config.RUNNING_LOSS
    accuracy = config.ACCURACY
    epochs = config.EPOCHS
    batch_size = config.DATALOADER_PARAMS['batch_size']

    for epoch in range(epochs):

    # for epoch in range(epochs):  # loop over the dataset multiple times
    #     print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        # for i, data in enumerate(trainloader, 0):
        #     inputs, labels = data # get the inputs; data is a list of [inputs, labels]
        #     optimizer.zero_grad() # zero the parameter gradients
            
        #     outputs = net(inputs) # forward
        #     loss = criterion(outputs, labels) # calculate loss
        #     loss.backward() # backward loss
        #     optimizer.step() # optimize gradients

        #     running_loss += loss.item() # save loss
        #     _, preds = torch.max(outputs, 1) # save prediction
        #     accuracy += torch.sum(preds == labels.data) # save accuracy
            
        #     if i % 1000 == 999:    # every 1000 mini-batches...           
        #         steps = epoch * len(trainloader) + i # calculate steps 
        #         batch = i*batch_size # calculate batch 
        #         print("Training loss {:.3} Accuracy {:.3} Steps: {}".format(running_loss / batch, accuracy/batch, steps))
                
        #         # Save accuracy and loss to Tensorboard
        #         writer.add_scalar('Training loss by steps', running_loss / batch, steps)
        #         writer.add_scalar('Training accuracy by steps', accuracy / batch, steps)

       
        # writer.add_figure("Confusion matrix For Each class", createConfusionMatrixForEachClass(trainloader), epoch)    
            # Train your model here
        trainer.train(model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data)
        
        # Calculate the confusion matrix for each class
        try:
            confusion_matrices = createConfusionMatrixForEachClass(trainloader, model)
        except TypeError:
            print("Error: Unable to create confusion matrices. Check data type compatibility.")
            continue

        # Add the confusion matrices to TensorBoard
        for class_idx, confusion_matrix in enumerate(confusion_matrices):
            plt.figure()
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - Class {class_idx}')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            writer.add_figure(f"Confusion Matrix - Class {class_idx}", plt.gcf(), epoch)
            plt.close()

        # Add the confusion matrices to TensorBoard
        for class_idx, confusion_matrix in enumerate(confusion_matrices):
            # Convert the confusion matrix (NumPy ndarray) to a PyTorch tensor
            confusion_matrix_tensor = torch.tensor(confusion_matrix, dtype=torch.float32)
            
            # Add the confusion matrix tensor to TensorBoard
            writer.add_image(f"Confusion Matrix - Class {class_idx}", confusion_matrix_tensor, epoch)

    best_model = models.get(config.MODEL_NAME,
                        num_classes=config.NUM_CLASSES,
                        checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'average_model.pth'))

    trainer.test(model=best_model,
            test_loader=test_data, 
            test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                top_k_predictions=300,
                                                num_cls=config.NUM_CLASSES,
                                                normalize_targets=True,
                                                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                                        nms_top_k=1000,
                                                                                                        max_predictions=300,
                                                                                                        nms_threshold=0.7)
                                                ))
                
    print("Accuracy: {}/{} ({:.3} %) Loss: {:.3}".format(accuracy, len(trainloader), 100. * accuracy / len(trainloader.dataset), running_loss / len(trainloader.dataset)))
    
    running_loss = 0.0
    accuracy = 0
    
    print("Finished Training")

    writer.close()
    PATH = f"{config.HOME}\AGIExperiment\AGIModel.pt"
    torch.save(model.state_dict(), PATH)
    
    # For predication, please change conf.
    img_path = 'GearInspection-Dataset/predict/year=2023-month=06-day=20-03_54_04-NG-2_0.png'
    best_model.predict(img_path, conf=0.25).show()
    best_model.predict(img_path, conf=0.50).show()
    best_model.predict(img_path, conf=0.75).show()