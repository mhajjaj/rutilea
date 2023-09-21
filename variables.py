import os
import torch.nn as nn
import torch.nn.functional as F

class Config:  
#trainer params
    HOME = os.getcwd()

    #dataset params
    CHECKPOINT_DIR = os.path.join(HOME, 'checkpoint', 'AGI-Dataset')  # Specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'AGIExperiment2'

    # Dataset params
    DATA_DIR = os.path.join(HOME, 'GearInspection-Dataset3', 'CategoryNG', 'ClassAll')
    LOGS = os.path.join(CHECKPOINT_DIR, 'AGILogs2')
    # CATEGORY = 'CategoryNG\ClassAll'

    TRAIN_IMAGES_DIR = 'train\images' 
    TRAIN_LABELS_DIR = 'train\labels' 

    VAL_IMAGES_DIR = 'valid\images'
    VAL_LABELS_DIR = 'valid\labels' 

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 128 * 160, 128, bias=True)  
        self.fc2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 227 * 347, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 1)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
        
    #     # Calculate the correct size for x before the linear layer
    #     batch_size = x.size(0)  # Get the batch size dynamically
    #     x = x.view(batch_size, -1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
if __name__ == '__main__':
    pass