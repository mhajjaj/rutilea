import os
import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt  # Import Matplotlib

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

class config:
    #trainer params
    CHECKPOINT_DIR = 'checkpoint/GearInspection-Dataset' #specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'Aisin-Gear-Inspection-Experiment' #specify the experiment name

    #dataset params
    DATA_DIR = 'GearInspection-Dataset' #parent directory to where data lives

    TRAIN_IMAGES_DIR = 'train/images' #child dir of DATA_DIR where train images are
    TRAIN_LABELS_DIR = 'train/labels' #child dir of DATA_DIR where train labels are

    VAL_IMAGES_DIR = 'valid/images' #child dir of DATA_DIR where validation images are
    VAL_LABELS_DIR = 'valid/labels' #child dir of DATA_DIR where validation labels are

    # if you have a test set
    TEST_IMAGES_DIR = 'test/images' #child dir of DATA_DIR where test images are
    TEST_LABELS_DIR = 'test/labels' #child dir of DATA_DIR where test labels are

    CLASSES = ['akkon', 'dakon', 'kizu'] #what class names do you have

    NUM_CLASSES = len(CLASSES)

    #dataloader params - you can add whatever PyTorch dataloader params you have
    #could be different across train, val, and test
    DATALOADER_PARAMS={
    'batch_size':8,
    'num_workers':2
    }

    # model params
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco' #only one option here: coco

if __name__ == '__main__':
    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

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
        # ENABLING SILENT MODE
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
        "max_epochs": 1,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            # NOTE: num_classes needs to be defined here
            num_classes=config.NUM_CLASSES,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                # NOTE: num_classes needs to be defined here
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

    training_history = trainer.train(model=model,
                                      training_params=train_params,
                                      train_loader=train_data,
                                      valid_loader=val_data)

    # Extract the training history for the desired metrics
    loss_values = training_history['train']['loss']
    loss_cls_values = training_history['train']['loss_cls']
    loss_dfl_values = training_history['train']['loss_dfl']
    loss_iou_values = training_history['train']['loss_iou']
    gpu_mem_values = training_history['train']['gpu_mem']

    # Create x-axis values (epochs)
    epochs = range(1, len(loss_values) + 1)

    # Plot the metrics
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss_values, label='PPYoloELoss/loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss_cls_values, label='PPYoloELoss/loss_cls')
    plt.xlabel('Epochs')
    plt.ylabel('Loss_cls')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, loss_dfl_values, label='PPYoloELoss/loss_dfl')
    plt.xlabel('Epochs')
    plt.ylabel('Loss_dfl')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, loss_iou_values, label='PPYoloELoss/loss_iou')
    plt.xlabel('Epochs')
    plt.ylabel('Loss_iou')
    plt.legend()

    plt.tight_layout()

    # Show GPU memory usage
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, gpu_mem_values, label='GPU Memory Usage', marker='o', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('GPU Memory (MB)')
    plt.legend()

    plt.show()

    best_model = models.get(config.MODEL_NAME,
                            num_classes=config.NUM_CLASSES,
                            checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'average_model.pth'))

    trainer.test(model=best_model,
                 test_loader=test_data,
                 test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                        top_k_predictions=300,
                                                        num_cls=config.NUM_CLASSES,
                                                        normalize_targets=True,
                                                        post_prediction_callback=PPYoloEPostPredictionCallback(
                                                            score_threshold=0.01,
                                                            nms_top_k=1000,
                                                            max_predictions=300,
                                                            nms_threshold=0.7)
                                                        ))

    img_path = 'GearInspection-Dataset/predict/year=2023-month=06-day=20-03_54_04-NG-2_0.png'
    best_model.predict(img_path, conf=0.25).show()
    best_model.predict(img_path, conf=0.50).show()
    best_model.predict(img_path, conf=0.75).show()
