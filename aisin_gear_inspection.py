import os
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
    CHECKPOINT_DIR = 'checkpoint/Aisin-Gear-Inspection-Dataset'
    EXPERIMENT_NAME = 'Aisin-Gear-Inspection Experiment'

    #dataset params
    DATA_DIR = 'Aisin-Gear-Inspection-Dataset'  # Replace with your actual dataset directory

    # Folders for each class
    TRAIN_IMAGES_DIR = 'dataset1_train_NG_dakon/images'
    TRAIN_LABELS_DIR = 'dataset1_train_NG_dakon/labels'

    VAL_IMAGES_DIR = 'validation/images'
    VAL_LABELS_DIR = 'validation/labels'

    # Folders for prediction
    PREDICT_DAKON_DIR = 'dataset1_predict_NG_dakon'
    PREDICT_KIZU_DIR = 'dataset1_predict_NG_kizu'
    PREDICT_AKKON_DIR = 'dataset1_train_NG_akkon'

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

    # Use these variables for prediction
    predict_dakon_data = dataloaders.ImageFolderDetection(
        dataset_params={
            'data_dir': os.path.join(config.DATA_DIR, config.PREDICT_DAKON_DIR),
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    predict_kizu_data = dataloaders.ImageFolderDetection(
        dataset_params={
            'data_dir': os.path.join(config.DATA_DIR, config.PREDICT_KIZU_DIR),
            'classes': config.CLASSES
        },
        dataloader_params=config.DATALOADER_PARAMS
    )

    predict_akkon_data = dataloaders.ImageFolderDetection(
        dataset_params={
            'data_dir': os.path.join(config.DATA_DIR, config.PREDICT_AKKON_DIR),
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
                train_loader=train_data,
                valid_loader=val_data)

    best_model = models.get(config.MODEL_NAME,
                        num_classes=config.NUM_CLASSES,
                        checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'average_model.pth'))

    trainer.test(model=best_model,
            test_loader=test_loader,
            test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                   top_k_predictions=300,
                                                   num_cls=config.NUM_CLASSES,
                                                   normalize_targets=True,
                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                                          nms_top_k=1000,
                                                                                                          max_predictions=300,
                                                                                                          nms_threshold=0.7)
                                                  ))
                      
    #best_model.predict( "path/to/your/asset",  conf=0.25).show()