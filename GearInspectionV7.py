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
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class config:
    #trainer params
    HOME = os.getcwd()

    CHECKPOINT_DIR = f'{HOME}\checkpoint\AGI-Dataset' #specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'AGIExperiment' 

    ##dataset params
    DATA_DIR = f'{HOME}\GearInspection-Dataset3\CategoryNG\ClassAll' 
    LOGS = f'{CHECKPOINT_DIR}\AGILogs'

    CATEGORY = 'CategoryNG\ClassAll'

    TRAIN_IMAGES_DIR = 'train\images' 
    TRAIN_LABELS_DIR = 'train\labels' 

    VAL_IMAGES_DIR = 'valid\images'
    VAL_LABELS_DIR = 'valid\labels' 

    # if you have a test set
    TEST_IMAGES_DIR = 'test\images' 
    TEST_LABELS_DIR = 'test\labels'

    #what class names do you have
    CLASSES = ['akkon', 'dakon', 'kizu', 'hakkon', 'kuromoyou', 'mizunokori', 'senkizu', 'yogore']

    NUM_CLASSES = len(CLASSES)

    DATALOADER_PARAMS={
    'batch_size':8,
    'num_workers':2
    }

    EPOCHS = 1
    RUNNING_LOSS = 0.0
    ACCURACY = 0.0

    # model params
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco' 

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST(config.CHECKPOINT_DIR,
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST(config.CHECKPOINT_DIR,
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.DATALOADER_PARAMS['batch_size'],
                                        shuffle=True, num_workers=config.DATALOADER_PARAMS['num_workers'])


testloader = torch.utils.data.DataLoader(testset, batch_size=config.DATALOADER_PARAMS['batch_size'],
                                        shuffle=False, num_workers=config.DATALOADER_PARAMS['num_workers'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def createConfusionMatrix(loader):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
        output = net(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

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

def createConfusionMatrixForEachClass(loader):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
        output = net(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # constant for classes -- config.CLASSES
    # classes = config.CLASSES

    # Example dataset
    true_labels = y_true # [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7]  # True class labels
    predicted_labels = y_pred # [0, 1, 2, 2, 0, 1, 3, 3, 4, 5, 5, 6, 6, 7, 7]  # Predicted class labels
    class_names = config.CLASSES

    # Initialize dictionaries to store metrics for each class
    confusion_matrices = {}
    accuracies = {}
    precisions = {}
    recalls = {}
    f1_scores = {}

    # Loop through each class
    for class_idx, class_name in enumerate(class_names):
        # Create binary labels for the current class vs. all other classes
        binary_true_labels = [1 if label == class_idx else 0 for label in true_labels]
        binary_predicted_labels = [1 if label == class_idx else 0 for label in predicted_labels]

        # Calculate confusion matrix for the current class
        conf_matrix = confusion_matrix(binary_true_labels, binary_predicted_labels)
        
        # Calculate performance metrics for the current class with zero_division=1.0
        accuracy = accuracy_score(binary_true_labels, binary_predicted_labels)
        precision = precision_score(binary_true_labels, binary_predicted_labels, zero_division=1.0)
        recall = recall_score(binary_true_labels, binary_predicted_labels, zero_division=1.0)
        f1 = f1_score(binary_true_labels, binary_predicted_labels, zero_division=1.0)
        
        # Store the metrics in their respective dictionaries with the class name as the key
        confusion_matrices[class_name] = conf_matrix
        accuracies[class_name] = accuracy
        precisions[class_name] = precision
        recalls[class_name] = recall
        f1_scores[class_name] = f1

    # Plot confusion matrices for all classes
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, class_name in enumerate(class_names):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        
        im = ax.imshow(confusion_matrices[class_name], cmap='Blues', interpolation='nearest')
        ax.set_title(f"Class {class_name} Confusion Matrix")
        
        # Customize tick labels based on your class names
        tick_marks = np.arange(2)  # Two classes: 0 and 1
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([f'Predicted {class_name}', f'Predicted Other'])
        ax.set_yticklabels([f'True {class_name}', f'True Other'])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(confusion_matrices[class_name][i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()
    plt.savefig('output2.png')

    # Print metrics for each class
    for class_name in class_names:
        print(f"Metrics for Class {class_name}:")
        print("Accuracy:", accuracies[class_name])
        print("Precision:", precisions[class_name])
        print("Recall:", recalls[class_name])
        print("F1 Score:", f1_scores[class_name])
        print()

    return sn.heatmap(conf_matrix, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names).get_figure()

if __name__ == '__main__':
    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

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
        "cosine_final_lr_ratio": 0.1,
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

    if trainer.train(model=model,
                    training_params=train_params,
                    train_loader=train_data,
                    valid_loader=val_data):
        
        running_loss = config.RUNNING_LOSS
        accuracy = config.ACCURACY
        epochs = config.EPOCHS
        batch_size = config.DATALOADER_PARAMS['batch_size'] 
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data # get the inputs; data is a list of [inputs, labels]
                optimizer.zero_grad() # zero the parameter gradients
                
                outputs = net(inputs) # forward
                loss = criterion(outputs, labels) # calculate loss
                loss.backward() # backward loss
                optimizer.step() # optimize gradients

                running_loss += loss.item() # save loss
                _, preds = torch.max(outputs, 1) # save prediction
                accuracy += torch.sum(preds == labels.data) # save accuracy
                
                if i % 1000 == 999:    # every 1000 mini-batches...           
                    steps = epoch * len(trainloader) + i # calculate steps 
                    batch = i*batch_size # calculate batch 
                    print("Training loss {:.3} Accuracy {:.3} Steps: {}".format(running_loss / batch, accuracy/batch, steps))
                    
                    # Save accuracy and loss to Tensorboard
                    writer.add_scalar('Training loss by steps', running_loss / batch, steps)
                    writer.add_scalar('Training accuracy by steps', accuracy / batch, steps)

        # trainer.train(model=model,
        #             training_params=train_params,
        #             train_loader=train_data,
        #             valid_loader=val_data)

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
        
        # Save confusion matrix to Tensorboard
        writer.add_figure("Confusion matrix", createConfusionMatrix(trainloader), epoch)
        
        running_loss = 0.0
        accuracy = 0
    
    writer.add_figure("Confusion matrix For Each class", createConfusionMatrixForEachClass(trainloader), config.EPOCHS)    
    print('Finished Training')

    # trainer.train(model=model,
    #             training_params=train_params,
    #             train_loader=train_data,
    #             valid_loader=val_data)

    # best_model = models.get(config.MODEL_NAME,
    #                     num_classes=config.NUM_CLASSES,
    #                     checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'average_model.pth'))

    # trainer.test(model=best_model,
    #         test_loader=test_data, 
    #         test_metrics_list=DetectionMetrics_050(score_thres=0.1,
    #                                             top_k_predictions=300,
    #                                             num_cls=config.NUM_CLASSES,
    #                                             normalize_targets=True,
    #                                             post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
    #                                                                                                     nms_top_k=1000,
    #                                                                                                     max_predictions=300,
    #                                                                                                     nms_threshold=0.7)
    #                                             ))

    writer.close()
    PATH = f"{config.HOME}\AGIExperiment\AGIModel.pt"
    torch.save(model.state_dict(), PATH)
    
    # For predication, please change conf.
    img_path = 'GearInspection-Dataset/predict/year=2023-month=06-day=20-03_54_04-NG-2_0.png'
    best_model.predict(img_path, conf=0.25).show()
    best_model.predict(img_path, conf=0.50).show()
    best_model.predict(img_path, conf=0.75).show()