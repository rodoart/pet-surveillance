import tensorflow as tf

class object_model:
    detection_graph = None
    image_tensor = None
    detection_boxes = None
    detection_scores = None
    num_detections = None
    category_index = None
    sess = None
    parser = None
