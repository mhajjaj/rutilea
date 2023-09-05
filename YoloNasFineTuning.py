from roboflow import Roboflow
rf = Roboflow(api_key="rwIxOQU4nvItixfrHFIg")
project = rf.workspace("atathamuscoinsdataset").project("u.s.-coins-dataset-a.tatham")
dataset = project.version(5).download("yolov5")

from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

dataset_params = {
    'data_dir':'U.S.-Coins-Dataset---A.Tatham-5',
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    'classes': ['Dime', 'Nickel', 'Penny', 'Quarter']
}

from IPython.display import clear_output

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

clear_output()

train_data.dataset.transforms
train_data.dataset.plot()

from super_gradients.training import models
model = models.get('yolo_nas_l',
                    num_classes=len(dataset_params['classes']),
                    pretrained_weights="coco"
                    )

from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

train_params = {
    # ENABLING SILENT MODE
    'silent_mode': True,
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
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": 10,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
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

#from super_gradients.training import Trainer

#CHECKPOINT_DIR = 'checkpoints/my_first_yolonas_run/average_model.pthcheckpoints/my_first_yolonas_run/average_model.pth' # "checkpoints/my_first_yolonas_run/average_model.pth"
#trainer = Trainer(experiment_name='Coins', ckpt_root_dir="checkpoints/my_first_yolonas_run/average_model.pth")

#trainer.train(model=model,
#              training_params=train_params,
#              train_loader=train_data,
#              valid_loader=val_data)

#best_model = models.get('yolo_nas_l',
#                        num_classes=len(dataset_params['classes']),
#                        checkpoint_path="checkpoints/my_first_yolonas_run/average_model.pth")

#trainer.test(model=best_model,
#            test_loader=test_data,
#            test_metrics_list=DetectionMetrics_050(score_thres=0.1,
#                                                   top_k_predictions=300,
#                                                   num_cls=len(dataset_params['classes']),
#                                                   normalize_targets=True,
#                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
#                                                                                                          nms_top_k=1000,
#                                                                                                          max_predictions=300,
#                                                                                                          nms_threshold=0.7)
#                                                  ))

#img_url = 'https://www.mynumi.net/media/catalog/product/cache/2/image/9df78eab33525d08d6e5fb8d27136e95/s/e/serietta_usa_2_1/www.mynumi.net-USASE5AD160-31.jpg'
#best_model.predict(img_url).show()