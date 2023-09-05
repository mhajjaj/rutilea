import super_gradients

yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()
model_predictions  = yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg").show()

prediction = model_predictions[0].prediction        # One prediction per image - Here we work with 1 image so we get the first.

bboxes = prediction.bboxes_xyxy                     # [[Xmin,Ymin,Xmax,Ymax],..] list of all annotation(s) for detected object(s) 
bboxes = prediction.bboxes_xyxy                     # [[Xmin,Ymin,Xmax,Ymax],..] list of all annotation(s) for detected object(s) 
class_names = prediction.class_names                # ['Class1', 'Class2', ...] List of the class names
class_name_indexes = prediction.labels.astype(int)  # [2, 3, 1, 1, 2, ....] Index of each detected object in class_names(corresponding to each bounding box)
confidences =  prediction.confidence.astype(float)  # [0.3, 0.1, 0.9, ...] Confidence value(s) in float for each bounding boxes