import yaml
import torch
from ultralytics import YOLO


with open('/configs/detec_model.yaml', 'r') as file:
    cfg = yaml.safe_load(file)


# Load the trained model
model = YOLO(cfg['Model']['trained'])

# Export the model to ONNX
model.export(
    format="onnx",
    imgsz=cfg['Arguments']['imgsz'],
    nms=cfg['Arguments']['nms'],
    conf=cfg['Arguments']['conf'],
    iou=cfg['Arguments']['iou']
)
