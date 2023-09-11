import os
import torch

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

HOME = os.getcwd()
print(f"{HOME}\GearInspection-Dataset")
# print (HOME)

SOURCE_IMAGE_PATH = f"{HOME}\GearInspection-Dataset\test\images\year=2022-month=07-day=05-12_53_16-NG-2_2.png"

import cv2

class config:
    #trainer params
    CHECKPOINT_DIR = f'{HOME}\checkpoint\GearInspection-Dataset' #specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'AGIExperiment' 

    #dataset params
    DATA_DIR = f'{HOME}\GearInspection-Dataset' 

    TRAIN_IMAGES_DIR = 'train\images' 
    TRAIN_LABELS_DIR = 'train\labels' 

    VAL_IMAGES_DIR = 'valid\images'
    VAL_LABELS_DIR = 'valid\labels' 

    # if you have a test set
    TEST_IMAGES_DIR = 'test\images' 
    TEST_LABELS_DIR = 'test\labels' 

    CLASSES = ['akkon', 'dakon', 'kizu', 'hakkon', 'kuromoyou', 'mizunokori', 'senkizu', 'yogore'] #what class names do you have

    NUM_CLASSES = len(CLASSES)

    BATCH_SIZE = 8
    NUM_WORKERS = 2
    MAX_EPOCHS = 500

    DATALOADER_PARAMS={
    'batch_size':8,
    'num_workers':2
    }

    # model params
    DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
    MODEL_ARCH = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l 
    PRETRAINED_WEIGHTS = 'coco' 

dataset_params = {
    'data_dir': config.DATA_DIR,
    'train_images_dir':config.TRAIN_IMAGES_DIR,
    'train_labels_dir':config.TRAIN_LABELS_DIR,
    'val_images_dir':config.VAL_IMAGES_DIR,
    'val_labels_dir':config.VAL_LABELS_DIR,
    'test_images_dir':config.TEST_IMAGES_DIR,
    'test_labels_dir':config.TEST_LABELS_DIR,
    'classes': config.CLASSES
}

model = models.get(config.MODEL_ARCH, pretrained_weights="coco").to(config.DEVICE)

from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)

if __name__ == '__main__':
    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TRAIN_IMAGES_DIR,
            'labels_dir': config.TRAIN_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params={
            'batch_size': config.BATCH_SIZE,
            'num_workers': config.NUM_WORKERS
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.VAL_IMAGES_DIR,
            'labels_dir': config.VAL_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params={
            'batch_size': config.BATCH_SIZE,
            'num_workers': config.NUM_WORKERS
        }
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TEST_IMAGES_DIR,
            'labels_dir': config.TEST_LABELS_DIR,
            'classes': config.CLASSES
        },
        dataloader_params={
            'batch_size': config.BATCH_SIZE,
            'num_workers': config.NUM_WORKERS
        }
    )

    train_data.dataset.transforms

    model = models.get(
        config.MODEL_ARCH, 
        num_classes=len(dataset_params['classes']), 
        pretrained_weights="coco"
    )

    train_params = {
        'silent_mode': False,
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
        "max_epochs": config.MAX_EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
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

    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_data, 
        valid_loader=val_data
    )

PATH = f"{HOME}\AGIExperiment\AGIModel.pt"
torch.save(model.state_dict(), PATH)