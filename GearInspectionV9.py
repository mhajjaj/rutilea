# Developed by m.hajjaj at rutilea
import os
import sys
import subprocess

import requests
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
import tensorflow as tf

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

sys.stdout = sys.__stdout__

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class config:
    #trainer params
    HOME = os.getcwd()

    CHECKPOINT_DIR = f'{HOME}\checkpoint\AGI-Dataset3' #specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'AGIExperiment3' 

    #dataset params
    DATA_DIR = f'{HOME}\GearInspection-Dataset3\CategoryNG\ClassAll' 
    LOGS = f'{CHECKPOINT_DIR}\AGILogs3'

    # CATEGORY = 'CategoryNG\ClassAll'

    TRAIN_IMAGES_DIR = 'train\image_split' 
    TRAIN_LABELS_DIR = 'train\labels_adjusted' 

    VAL_IMAGES_DIR = 'valid\images'
    VAL_LABELS_DIR = 'valid\labels' 

    TEST_IMAGES_DIR = 'test\images' 
    TEST_LABELS_DIR = 'test\labels'

    #what class names do you have
    CLASSES = ['akkon', 'dakon', 'kizu', 'hakkon', 'kuromoyou', 'mizunokori', 'senkizu', 'yogore']

    NUM_CLASSES = len(CLASSES)

    DATALOADER_PARAMS={
    'batch_size':8,
    'num_workers':2,
    'worker_init_fn':seed_worker,
    'generator':torch.Generator().manual_seed(0),
    }

    EPOCHS = 10
    RUNNING_LOSS = 0.0
    ACCURACY = 0.0

    # model params
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco'

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
    
    trainer.train(model=model,
                training_params=train_params,
                train_loader=train_data,
                valid_loader=val_data)

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

    # Run the nvidia-smi command
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)

    # Get the output
    nvidia_smi_output = result.stdout
    # Print the output
    print(result.stdout)

    # Create a TensorFlow summary
    summary = tf.summary.text("NVIDIA SMI Output", tf.constant(nvidia_smi_output))

    # Create a summary writer
    log_dir = config.LOGS  # Change this to your desired log directory
    writer = tf.summary.create_file_writer(log_dir)

    # Log the summary to TensorBoard
    # with writer.as_default():
    #     tf.summary.write("NVIDIA SMI Output", tf.constant(nvidia_smi_output), step=0)  # Change step as needed

    # Close the summary writer
    writer.close()
    writer = SummaryWriter(config.LOGS)

    PATH = f"{config.HOME}\AGIExperiment\AGIModel3.pt"
    torch.save(model.state_dict(), PATH)
