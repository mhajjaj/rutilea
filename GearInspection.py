import os
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter  

from torch.utils.data import DataLoader

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
    CHECKPOINT_DIR = 'checkpoint/GearInspection-Dataset' 
    EXPERIMENT_NAME = 'Aisin-Gear-Inspection-Experiment' 

    DATA_DIR = 'GearInspection-Dataset' 

    TRAIN_IMAGES_DIR = 'train/images' 
    TRAIN_LABELS_DIR = 'train/labels' 

    VAL_IMAGES_DIR = 'valid/images' 
    VAL_LABELS_DIR = 'valid/labels' 

    TEST_IMAGES_DIR = 'test/images' 
    TEST_LABELS_DIR = 'test/labels' 

    CLASSES = ['akkon', 'dakon', 'kizu', 'hakkon', 'kuromoyou', 'mizunokori', 'senkizu', 'yogore'] 

    NUM_CLASSES = len(CLASSES)

    DATALOADER_PARAMS={
    'batch_size':8,
    'num_workers':2
    }

    MODEL_NAME = 'yolo_nas_l' 
    PRETRAINED_WEIGHTS = 'coco'

if __name__ == '__main__':
    trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

    tensorboard_writer = SummaryWriter()

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

    train_loader = torch.utils.data.DataLoader(train_data, **config.DATALOADER_PARAMS)  # Create a DataLoader for training data
#    first_batch = train_loader[0] 

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
        "max_epochs": 1,
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

    for epoch in range(train_params['max_epochs']):
        model.train()

        total_loss = 0.0
        total_loss_cls = 0.0
        total_loss_dfl = 0.0
        total_loss_iou = 0.0

        # train_iterator = iter(train_loader)
        # first_batch = next(train_iterator)

        for batch_idx, batch in enumerate(train_loader.dataset):
            data, target = batch 

            # optimizer.zero_grad()
            output = model(data)
            
            loss, loss_cls, loss_dfl, loss_iou = nn.CrossEntropyLoss(output, target) 
            #compute_loss(output, target)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_dfl += loss_dfl.item()
            total_loss_iou += loss_iou.item()

        avg_loss = total_loss / len(train_loader)
        avg_loss_cls = total_loss_cls / len(train_loader)
        avg_loss_dfl = total_loss_dfl / len(train_loader)
        avg_loss_iou = total_loss_iou / len(train_loader)

        tensorboard_writer.add_scalar('Train/PPYoloELoss/loss', avg_loss, global_step=epoch)
        tensorboard_writer.add_scalar('Train/PPYoloELoss/loss_cls', avg_loss_cls, global_step=epoch)
        tensorboard_writer.add_scalar('Train/PPYoloELoss/loss_dfl', avg_loss_dfl, global_step=epoch)
        tensorboard_writer.add_scalar('Train/PPYoloELoss/loss_iou', avg_loss_iou, global_step=epoch)
        tensorboard_writer.add_scalar('Train/gpu_mem', gpu_mem, global_step=epoch)

        print(f'Epoch [{epoch+1}/{train_params["max_epochs"]}], '
            f'Loss: {avg_loss:.4f}, '
            f'Loss_cls: {avg_loss_cls:.4f}, '
            f'Loss_dfl: {avg_loss_dfl:.4f}, '
            f'Loss_iou: {avg_loss_iou:.4f}')

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
                                                  
    img_path = 'GearInspection-Dataset/predict/year=2023-month=06-day=20-03_54_04-NG-2_0.png'
    best_model.predict(img_path, conf=0.25).show()
    best_model.predict(img_path, conf=0.50).show()
    best_model.predict(img_path, conf=0.75).show()



# import os
# import sys

# import requests
# import torch
# from PIL import Image
# from tensorboardX import SummaryWriter  # Import TensorBoardX

# import torch.optim as optim

# from super_gradients.training import Trainer, dataloaders, models
# from super_gradients.training.dataloaders.dataloaders import (
#     coco_detection_yolo_format_train, coco_detection_yolo_format_val
# )
# from super_gradients.training.losses import PPYoloELoss
# from super_gradients.training.metrics import DetectionMetrics_050
# from super_gradients.training.models.detection_models.pp_yolo_e import (
#     PPYoloEPostPredictionCallback
# )

# class config:
#     #trainer params
#     CHECKPOINT_DIR = 'checkpoint/GearInspection-Dataset' #specify the path you want to save checkpoints to
#     EXPERIMENT_NAME = 'Aisin-Gear-Inspection-Experiment' #specify the experiment name

#     #dataset params
#     DATA_DIR = 'GearInspection-Dataset' #parent directory to where data lives

#     TRAIN_IMAGES_DIR = 'train/images' #child dir of DATA_DIR where train images are
#     TRAIN_LABELS_DIR = 'train/labels' #child dir of DATA_DIR where train labels are

#     VAL_IMAGES_DIR = 'valid/images' #child dir of DATA_DIR where validation images are
#     VAL_LABELS_DIR = 'valid/labels' #child dir of DATA_DIR where validation labels are

#     # if you have a test set
#     TEST_IMAGES_DIR = 'test/images' #child dir of DATA_DIR where test images are
#     TEST_LABELS_DIR = 'test/labels' #child dir of DATA_DIR where test labels are

#     CLASSES = ['akkon', 'dakon', 'kizu'] #what class names do you have

#     NUM_CLASSES = len(CLASSES)

#     #dataloader params - you can add whatever PyTorch dataloader params you have
#     #could be different across train, val, and test
#     DATALOADER_PARAMS={
#     'batch_size':8,
#     'num_workers':2
#     }

#     # model params
#     MODEL_NAME = 'yolo_nas_m' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
#     PRETRAINED_WEIGHTS = 'coco' #only one option here: coco

# if __name__ == '__main__':
#     # Create a TensorBoard SummaryWriter to log metrics
#     log_dir = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'logs')
#     os.makedirs(log_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=log_dir)

#     trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)

#     train_data = coco_detection_yolo_format_train(
#         dataset_params={
#             'data_dir': config.DATA_DIR,
#             'images_dir': config.TRAIN_IMAGES_DIR,
#             'labels_dir': config.TRAIN_LABELS_DIR,
#             'classes': config.CLASSES
#         },
#         dataloader_params=config.DATALOADER_PARAMS
#     )

#     val_data = coco_detection_yolo_format_val(
#         dataset_params={
#             'data_dir': config.DATA_DIR,
#             'images_dir': config.VAL_IMAGES_DIR,
#             'labels_dir': config.VAL_LABELS_DIR,
#             'classes': config.CLASSES
#         },
#         dataloader_params=config.DATALOADER_PARAMS
#     )

#     test_data = coco_detection_yolo_format_val(
#         dataset_params={
#             'data_dir': config.DATA_DIR,
#             'images_dir': config.TEST_IMAGES_DIR,
#             'labels_dir': config.TEST_LABELS_DIR,
#             'classes': config.CLASSES
#         },
#         dataloader_params=config.DATALOADER_PARAMS
#     )

#     train_data.dataset.plot()

#     model = models.get(config.MODEL_NAME,
#                        num_classes=config.NUM_CLASSES,
#                        pretrained_weights=config.PRETRAINED_WEIGHTS
#                        )

#     train_params = {
#         # ENABLING SILENT MODE
#         "average_best_models": True,
#         "warmup_mode": "linear_epoch_step",
#         "warmup_initial_lr": 1e-6,
#         "lr_warmup_epochs": 3,
#         "initial_lr": 5e-4,
#         "lr_mode": "cosine",
#         "cosine_final_lr_ratio": 0.1,
#         "optimizer": "Adam",
#         "optimizer_params": {"weight_decay": 0.0001},
#         "zero_weight_decay_on_bias_and_bn": True,
#         "ema": True,
#         "ema_params": {"decay": 0.9, "decay_type": "threshold"},
#         "max_epochs": 1,  # Increase the number of epochs for training
#         "mixed_precision": True,
#         "loss": PPYoloELoss(
#             use_static_assigner=False,
#             # NOTE: num_classes needs to be defined here
#             num_classes=config.NUM_CLASSES,
#             reg_max=16
#         ),
#         "valid_metrics_list": [
#             DetectionMetrics_050(
#                 score_thres=0.1,
#                 top_k_predictions=300,
#                 # NOTE: num_classes needs to be defined here
#                 num_cls=config.NUM_CLASSES,
#                 normalize_targets=True,
#                 post_prediction_callback=PPYoloEPostPredictionCallback(
#                     score_threshold=0.01,
#                     nms_top_k=1000,
#                     max_predictions=300,
#                     nms_threshold=0.7
#                 )
#             )
#         ],
#         "metric_to_watch": 'mAP@0.50'
#     }

#     global_step = 0  # Initialize global step
#     optimizer = optim.Adam(model.parameters(), lr=train_params["initial_lr"], weight_decay=train_params["optimizer_params"]["weight_decay"])

#     for epoch in range(train_params["max_epochs"]):
#         # Training loop
#         for batch_idx, data in enumerate(train_data):
#             # The training code here

#             trainer.train(model=model,
#                         training_params=train_params,
#                         train_loader=train_data,
#                         valid_loader=val_data)
            
#             optimizer.zero_grad()

#             #print(f"-----------------")
#             #print(type(data))
            
#             input_tensor = torch.stack(data, dim=0)
#             # data = data.to(device)  # Replace 'device' with your actual device (e.g., 'cuda' or 'cpu')
#             output = model(data)  # Replace with your model forward pass

#             desired_size = (8, 3, 640, 640)
#             processed_data = [torch.nn.functional.interpolate(tensor, size=desired_size[2:]) for tensor in data]

#             print(f"Tensor {batch_idx} size: {tensor.size()}")
#             #print(type(output))

#             #sys.exit()

#             loss = compute_loss(output, target)  # Replace with your loss computation
#             loss.backward()
#             optimizer.step()
                                                  
#             # Log the loss and other metrics to TensorBoard
#             writer.add_scalar('Train/PPYoloELoss/loss', loss.item(), global_step)
#             writer.add_scalar('Train/PPYoloELoss/loss_cls', loss_cls.item(), global_step)
#             writer.add_scalar('Train/PPYoloELoss/loss_dfl', loss_dfl.item(), global_step)
#             writer.add_scalar('Train/PPYoloELoss/loss_iou', loss_iou.item(), global_step)
#             # Log GPU memory usage
#             writer.add_scalar('Train/GPU_Memory', torch.cuda.memory_allocated(0) / (1024 * 1024), global_step)

#             global_step += 1  # Increment the global step for each batch

#         # Validation loop
#         for batch_idx, data in enumerate(val_data):
#             # The validation code here
#             best_model = models.get(config.MODEL_NAME,
#                                     num_classes=config.NUM_CLASSES,
#                                     checkpoint_path=os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, 'average_model.pth'))

#             trainer.test(model=best_model,
#                     test_loader=test_data,  # Update variable name here
#                     test_metrics_list=DetectionMetrics_050(score_thres=0.1,
#                                                         top_k_predictions=300,
#                                                         num_cls=config.NUM_CLASSES,
#                                                         normalize_targets=True,
#                                                         post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
#                                                                                                                 nms_top_k=1000,
#                                                                                                                 max_predictions=300,
#                                                                                                                 nms_threshold=0.7)
#                                                         ))
#             pass

#         # Save the model checkpoint
#         checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, f'checkpoint_epoch_{epoch}.pth')
#         torch.save(model.state_dict(), checkpoint_path)

#     writer.close()  # Close the TensorBoard SummaryWriter when done
                                                  
#     img_path = 'GearInspection-Dataset/predict/year=2023-month=06-day=20-03_54_04-NG-2_0.png'
#     best_model.predict(img_path, conf=0.25).show()
#     best_model.predict(img_path, conf=0.50).show()
#     best_model.predict(img_path, conf=0.75).show()