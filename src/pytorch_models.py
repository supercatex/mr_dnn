#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn


class TorchModel(object):
    def __init__(self, torch_home) -> None:
        os.environ["TORCH_HOME"] = torch_home
        self.device = "cpu"
    

class FasterRCNN(TorchModel):
    def __init__(self, torch_home) -> None:
        super().__init__(torch_home)
        self.net = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        self.net.eval().to(self.device)
    
    def forward(self, frame):
        img = frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        img = np.array(img, dtype=np.float32) / 255
        img = torch.from_numpy(img)
        img = img.to(self.device)
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
