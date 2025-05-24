import yaml
import torch
from ultralytics import YOLO


with open('/configs/train.yaml', 'r') as file:
    cfg = yaml.safe_load(file)


# Load pretrained model
model = YOLO(cfg['Model']['pretrained'])

# Fine-tune the model
model.train(
    data=cfg['Arguments']['data'],
    batch=cfg['Arguments']['num_batchs'],
    epochs=cfg['Arguments']['epochs'],
    imgsz=cfg['Arguments']['imgsz'],
    device=cfg['Arguments']['device'] # 0 for GPU, 'cpu' for CPU
)


metrics = model.val()  # Validate on validation set
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75