import os
from torchvision import transforms

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)
from torchvision import datasets  # Import the datasets module from torchvision

class config:
    #trainer params
    CHECKPOINT_DIR = 'checkpoint/Aisin-Gear-Inspection-Dataset'
    EXPERIMENT_NAME = 'Aisin-Gear-Inspection Experiment'

    #dataset params
    DATA_DIR = 'Aisin-Gear-Inspection-Dataset'  # Replace with your actual dataset directory

    # Folders for each class
    TRAIN_IMAGES_DIR_DAKON = 'dataset1_train_NG_dakon/images'
    TRAIN_LABELS_DIR_DAKON = 'dataset1_train_NG_dakon/labels'

    VALID_IMAGES_DIR_DAKON = 'dataset1_valid_NG_dakon/images'
    VALID_LABELS_DIR_DAKON = 'dataset1_valid_NG_dakon/labels'

    TRAIN_IMAGES_DIR_KIZU = 'dataset1_train_NG_kizu/images'
    TRAIN_LABELS_DIR_KIZU = 'dataset1_train_NG_kizu/labels'

    TRAIN_IMAGES_DIR_AKKON = 'dataset1_train_NG_akkon/images'
    TRAIN_LABELS_DIR_AKKON = 'dataset1_train_NG_akkon/labels'

    # Folders for prediction
    PREDICT_DAKON_DIR = 'dataset1_valid_NG_dakon/images'
    PREDICT_KIZU_DIR = 'dataset1_valid_NG_kizu/labels'
    PREDICT_AKKON_DIR = None

    CLASSES = ['dakon', 'kizu', 'akkon']  # Update with your actual classes
    NUM_CLASSES = len(CLASSES)

    DATALOADER_PARAMS = {
        'batch_size': 8,
        'num_workers': 2
    }

    MODEL_NAME = 'yolo_nas_l'
    PRETRAINED_WEIGHTS = 'coco'

if __name__ == '__main__':
    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

    train_data_dakon = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.TRAIN_IMAGES_DIR_DAKON,
            'labels_dir': config.TRAIN_LABELS_DIR_DAKON,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    val_data_dakon = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': config.DATA_DIR,
            'images_dir': config.VALID_IMAGES_DIR_DAKON,
            'labels_dir': config.VALID_LABELS_DIR_DAKON,
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    # Define image transforms
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to a specific size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image values
    ])

    # Use datasets.ImageFolder for prediction (no labels)
    predict_dakon_data = datasets.ImageFolder(
        root=os.path.join(config.DATA_DIR, "dataset1_valid_NG_dakon"),
        transform=image_transforms,  # Use the defined transforms here
    )

    predict_kizu_data = datasets.ImageFolder(
        root=os.path.join(config.DATA_DIR, config.PREDICT_KIZU_DIR),
        transform=image_transforms,  # Use the defined transforms here
    )

    predict_akkon_data = datasets.ImageFolder(
        root=os.path.join(config.DATA_DIR, config.PREDICT_AKKON_DIR),
        transform=image_transforms,  # Use the defined transforms here
    )

    train_data_dakon.dataset.plot()

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
        "optimizer": "kizu",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
        "max_epochs": 10,
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

    trainer.train(model=model,
                training_params=train_params,
                train_loader=train_data_dakon,  # Use the appropriate dataset here
                valid_loader=val_data_dakon)  # Use the appropriate dataset here
