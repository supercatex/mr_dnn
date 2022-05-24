#!/usr/bin/env python3
import cv2
import numpy as np
from openvino.runtime import Core


class IntelPreTrainedModel(object):
    def __init__(self, models_dir: str, model_name: str) -> None:
        ie = Core()
        name = model_name
        path = "%s/intel/%s/FP16/%s.xml" % (models_dir, name, name)
        net = ie.read_model(model=path)
        self.model = ie.compile_model(model=net, device_name="CPU")
        self.input_layer = next(iter(self.model.inputs))
        self.output_layers = []
        for layer in self.model.outputs:
            self.output_layers.insert(0, layer)
    
    def forward(self, inputs):
        return self.model(inputs=[inputs])


class FaceDetection(IntelPreTrainedModel):
    def __init__(self, models_dir: str) -> None:
        super().__init__(models_dir, "face-detection-adas-0001")

    def forward(self, frame): 
        # (B, C, H, W) => (1, 3, 384, 672) RGB
        img = frame.copy()
        h, w, c = img.shape
        img = cv2.resize(img, (672, 384))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.transpose(2, 0, 1), 0)

        # (1, 1, 200, 7) [image_id, label, conf, x_min, y_min, x_max, y_max]
        boxes = super().forward(img)[self.output_layers[0]][0][0]

        # (N, 4) (x1, y1, x2, y2)
        res = []
        for id, label, conf, x1, y1, x2, y2 in boxes:
            if conf < 0.95: continue
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            if x1 < 0 or y1 < 0: continue
            if x2 >= w or y2 >= h: continue
            if x1 >= x2 or y1 >= y2: continue
            res.append([x1, y1, x2, y2])
        return res


class AgeGenderRecognition(IntelPreTrainedModel):
    def __init__(self, models_dir: str) -> None:
        super().__init__(models_dir, "age-gender-recognition-retail-0013")
        self.genders_label = ("female", "male")

    def forward(self, frame): 
        # (B, C, H, W) => (1, 3, 62, 62) BGR
        img = frame.copy()
        img = cv2.resize(img, (62, 62))
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        out = super().forward(img)

        # (1, 1, 1, 1) divided by 100
        age = out[self.output_layers[0]][0][0][0][0]
        # (1, 2, 1, 1) [0 - female, 1 - male]
        gender = out[self.output_layers[1]][0]
        return age * 100, np.argmax(gender)


class EmotionsRecognition(IntelPreTrainedModel):
    def __init__(self, models_dir: str) -> None:
        super().__init__(models_dir, "emotions-recognition-retail-0003")
        self.emotions_label = ("neutral", "happy", "sad", "surprise", "anger")

    def forward(self, frame):
        # (B, C, H, W) => (1, 3, 64, 64) BGR
        img = frame.copy()
        img = cv2.resize(img, (64, 64))
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        out = super().forward(img)

        # (1, 5, 1, 1) [0 - neutral, 1 - happy, 2 - sad, 3 - surprise, 4 - anger]
        out = out[self.output_layers[0]][0]
        return np.argmax(out)
