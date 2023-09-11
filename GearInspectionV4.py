# Developed by m.hajjaj at rutilea
import os
import sys

import requests
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch.nn.functional as F

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
    EXPERIMENT_NAME = 'Aisin-Gear-Inspection-Experiment' 

    #dataset params
    DATA_DIR = 'GearInspection-Dataset' 

    TRAIN_IMAGES_DIR = 'train/images' 
    TRAIN_LABELS_DIR = 'train/labels' 

    VAL_IMAGES_DIR = 'valid/images'
    VAL_LABELS_DIR = 'valid/labels' 

    # if you have a test set
    TEST_IMAGES_DIR = 'test/images' 
    TEST_LABELS_DIR = 'test/labels' 

    CLASSES = ['akkon', 'dakon', 'kizu', 'hakkon', 'kuromoyou', 'mizunokori', 'senkizu', 'yogore'] #what class names do you have

    NUM_CLASSES = len(CLASSES)

    DATALOADER_PARAMS={
    'batch_size':8,
    'num_workers':2
    }

    # model params
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
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

    max_epochs = 1
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
        "max_epochs": max_epochs, # Change No of epochs you want, more is better until a level
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

    ################
    train_loader = torch.utils.data.DataLoader(train_data, **config.DATALOADER_PARAMS)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # x = ([-5, 5, 0.1])
    # y = -5 + 9 * 3

    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    # def train_model(iter):
    #     for epoch in range(iter):
    #         y1 = model(x)
    #         loss = criterion(y1, y)
    #         writer.add_scalar("Loss/train", loss, epoch)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    # train_model(max_epochs)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["initial_lr"], weight_decay=train_params["optimizer_params"]["weight_decay"])

    # Training loop with logging
    global_step = 0

    ################################# Model Checkpoint
    checkpoint_dir = 'checkpoint/GearInspection-Dataset' # CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

    def load_checkpoint(model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    checkpoint_path = 'checkpoint/GearInspection-Dataset/checkpoint_epoch_1.pt'  # Replace with the path to your checkpoint file

    #################################

    for epoch in range(train_params["max_epochs"]):
        for batch_idx, (data, target) in enumerate(train_data):
            optimizer.zero_grad()

            output_tuple = model(data)  # Ensure that the model returns a tuple

            if not isinstance(output_tuple, tuple):
                raise ValueError(f"Model output should be a tuple, but got {type(output_tuple)}")

            # Unpack the relevant tensors and ignore the rest
            output = output_tuple[0]  # Replace [0] with the correct index if needed

            save_checkpoint(model, optimizer, max_epochs, 0, checkpoint_dir)

            print(output.shape)
            model, optimizer, epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)

            # raise ValueError(output.shape)

            # Reshape the output to separate class scores from other components
            output_shape = output.shape
            num_anchors = 3  # Adjust based on your model's configuration
            num_classes = config.NUM_CLASSES

            output_reshaped = output.view(output_shape[0], output_shape[1], output_shape[2], num_anchors, num_classes + 5)
            output_logits = output_reshaped[..., 5:(5 + num_classes)]

            # Calculate the classification loss
            target_labels = target[..., 0].long()  # Assuming the class label is in the first position
            classification_loss = torch.nn.functional.cross_entropy(output_logits.view(-1, num_classes), target_labels.view(-1))

            # Calculate loss_dfl and loss_iou based on your implementation
            loss_dfl = calculate_loss_dfl(output, bounding_boxes, grid_sizes, image_sizes)  # Replace with your implementation
            loss_iou = calculate_loss_iou(output, bounding_boxes, grid_sizes, image_sizes)  # Replace with your implementation

            # Calculate the total loss
            total_loss = classification_loss + loss_dfl + loss_iou

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Log the total loss to TensorBoard
            writer.add_scalar('Loss/total_loss', total_loss.item(), global_step)

            global_step += 1
    
    # Close the TensorBoard SummaryWriter
    writer.close()
   
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
    writer.flush()
    writer.close()

    # For predication, please change conf.
    img_path = 'GearInspection-Dataset/predict/year=2023-month=06-day=20-03_54_04-NG-2_0.png'
    best_model.predict(img_path, conf=0.25).show()
    best_model.predict(img_path, conf=0.50).show()
    best_model.predict(img_path, conf=0.75).show()