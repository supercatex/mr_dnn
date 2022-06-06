#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import keypointrcnn_resnet50_fpn


class TorchModel(object):
    def __init__(self, torch_home) -> None:
        os.environ["TORCH_HOME"] = torch_home
        self.device = "cpu"
    
    def BGR2Tensor(self, frame):
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        img = np.array(img, dtype=np.float32) / 255
        img = torch.from_numpy(img)
        img = img.to(self.device)
        return img
    

class FasterRCNN(TorchModel):
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, torch_home) -> None:
        super().__init__(torch_home)
        self.net = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        self.net.eval().to(self.device)
        self.labels = self.COCO_CLASSES
    
    def forward(self, frame):
        img = self.BGR2Tensor(frame)
        out = self.net(img)

        res = []
        boxes = out[0]["boxes"]
        labels = out[0]["labels"]
        scores = out[0]["scores"]
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.7: continue
            x1, y1, x2, y2 = map(int, box)
            res.append([0, label, score, x1, y1, x2, y2])
        return res


class Yolov5(TorchModel):
    def __init__(self, torch_home) -> None:
        super().__init__(torch_home)
        self.net = torch.hub.load("ultralytics/yolov5", "yolov5s", device="cpu")
        self.net.eval().to(self.device)
        self.labels = self.net.names

    def forward(self, frame):
        img = frame.copy()
        out = self.net(img)

        res = []
        for x1, y1, x2, y2, pred, index in out.xyxy[0]:
            if pred < 0.7: continue
            x1, y1, x2, y2, index = map(int, (x1, y1, x2, y2, index))
            res.append([0, index, pred, x1, y1, x2, y2])
        return res
        