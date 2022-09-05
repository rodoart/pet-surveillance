from types import NoneType
import tensorflow as tf

class object_model:
    detection_graph = NoneType
    image_tensor = NoneType
    detection_boxes = NoneType
    detection_scores = NoneType
    num_detections = NoneType
    category_index = NoneType
    sess = NoneType
    parser = NoneType
