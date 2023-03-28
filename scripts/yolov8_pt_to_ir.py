from ultralytics import YOLO
from pathlib import Path


DET_MODEL_NAME = "yolov8s"
det_model = YOLO("%s.pt" % DET_MODEL_NAME)
label_map = det_model.model.names

SEG_MODEL_NAME = "yolov8s-seg"
seg_model = YOLO("%s.pt" % SEG_MODEL_NAME)

det_model_path = Path("%s_openvino_model/%s.xml" % (DET_MODEL_NAME, DET_MODEL_NAME))
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)

seg_model_path = Path("%s_openvino_model/%s.xml" % (SEG_MODEL_NAME, SEG_MODEL_NAME))
if not seg_model_path.exists():
    seg_model.export(format="openvino", dynamic=True, half=False)

    
