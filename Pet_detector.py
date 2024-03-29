# Add submodule folder to path.
## Modifications to the library were required, therefore is necessary to load directly from the local directory.

import sys
sys.path.append('tensorflow/models/research/')
sys.path.append('tensorflow/models/research/object_detection/')

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from Pet_detector_object import object_model as model

## Segformer
from pet_surveillance.models import segformer
from pet_surveillance.utils.video_layers import boolean_mask_overlay

def setupCamera(camera_type, file_path=''):
    # Set up camera constants
    IM_WIDTH = 1280
    IM_HEIGHT = 720
    #### Initialize other parameters ####

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX
    # The camera has to be set up and used differently depending on if it's a
    # Picamera or USB webcam.
    ### USB webcam ###
    if camera_type == 'usb_camera':
        # Initialize USB webcam feed
        camera = cv2.VideoCapture(0)
    elif camera_type == 'file':
        camera =cv2.VideoCapture(file_path) 

    if (camera.isOpened()== False):
        print("Error opening video stream or file")
    
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)
    
    # Model loads
    objectModel = model_load()
    objectSegformer = segformer.Segformer()

        # Continuously capture frames and perform object detection on them
    
    while(camera.isOpened()):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        if(ret == True):

            # Pass frame into pet detection function
            global floor
            
            try:
                floor
            except:
                floor = floor_detection(frame, objectSegformer)
            
            frame = pet_detector(frame, floor, IM_WIDTH, IM_HEIGHT, font, objectModel)

            # Draw FPS
            cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # FPS calculation
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                    break
        else:
            break
    
    camera.release()
    cv2.destroyAllWindows()

 #### Pet detection function ####

    # This function contains the code to detect a pet, determine if it's
    # inside or outside, and send a text to the user's phone.
    #### Initialize camera and perform object detection ####

def pet_detector(frame, floor, im_width, im_height, font, objectModel):
    
    # Use globals for the control variables so they retain their value after function exits

    global detected_inside, detected_outside
    global inside_counter, outside_counter
    global pause, pause_counter
    global limitxMax, limitxMin, limityMax, limityMin

    try:
        pause
    except:
        pause = 0

    try:
        inside_counter
    except:
        inside_counter = 0
     
    
    # Initialize control variables used for pet detector
    detected_inside = False
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = objectModel.sess.run(
        [objectModel.detection_boxes, objectModel.detection_scores, objectModel.detection_classes, objectModel.num_detections],
        feed_dict={objectModel.image_tensor: frame_expanded})

    # Draw floor
    frame = boolean_mask_overlay(frame, mask=floor)

    # Draw the results of the detection (aka 'visualize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        objectModel.category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

    if(((int(classes[0][0]) == 63) or (int(classes[0][0] == 62) or (int(classes[0][0]) == 1))) and (pause == 0)):
        limitxMax = int(((boxes[0][0][1]+boxes[0][0][3])/2)*im_width)
        limityMax = int(((boxes[0][0][1]+boxes[0][0][2])/2)*im_height)
        limitxMin = int(((boxes[0][0][1]+boxes[0][0][1])/2)*im_width)
        limityMin = int(((boxes[0][0][1]+boxes[0][0][0])/2)*im_height)

    



    # Check the class of the top detected object by looking at classes[0][0].
    # If the top detected object is a cat (17) or a dog (18) (or a teddy bear (88) for test purposes),
    # find its center coordinates by looking at the boxes[0][0] variable.
    # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax
    x = 0
    y = 0

    if (((int(classes[0][0]) == 17) or (int(classes[0][0] == 1) or (int(classes[0][0]) == 88))) and (pause == 0)):

        
        x_no_rescaled = int(((boxes[0][0][1]+boxes[0][0][1])/2)*frame.shape[1])
        y_no_rescaled = int(((boxes[0][0][0]+boxes[0][0][2])/2)*frame.shape[0])
        x = int(((boxes[0][0][1]+boxes[0][0][1])/2)*im_width)
        y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*im_height)


        

        # Draw a circle at center of object
        cv2.circle(frame,(x,y), 5, (75,13,180), -1)

        # Probability of the animal not in the floor.        
        y_start = int(((boxes[0][0][1]+boxes[0][0][0])/2)*frame.shape[1])
        y_stop = int(((boxes[0][0][1]+boxes[0][0][2])/2)*frame.shape[1])
        x_start = int(((boxes[0][0][1]+boxes[0][0][1])/2)*frame.shape[0])
        x_stop = int(((boxes[0][0][1]+boxes[0][0][3])/2)*frame.shape[0]) 
        
        floor_arround_center = floor[y_start:y_stop, x_start:x_stop]  

        sum_floor = np.sum(floor_arround_center)
        box_area = (y_stop-y_start)*(x_stop-x_start)
        p_not_in_floor = 1-sum_floor/box_area


        # If object is in inside box, increment inside counter variable
        if p_not_in_floor >= 0.85:
            inside_counter = inside_counter + 1

    # If pet has been detected inside for more than 10 frames, set detected_inside flag
    if inside_counter >= 5: 
        cv2.putText(
            img = frame, 
            text='Pet has been on the couch!',
            org =  (int(im_width*.1),int(im_height*.5)),
            fontFace=font,
            fontScale=1.5,
            color=(0,0,255),
            thickness=7,
            lineType=cv2.LINE_AA
        )

        #inside_counter = 0
        # Pause pet detection by setting "pause" flag
        pause = 1

    # If paussee flag is set, draw message on screen.
    # Draw counter info
    cv2.putText(frame, f'Detection counter: {str(inside_counter)}', (10,100),
                font, 0.5, (255,255,0), 1, cv2.LINE_AA)
    
    return frame

def floor_detection(frame, objectSegformer):    
    labels = objectSegformer.predict_labels(frame)
    return objectSegformer.detect_floor(labels)   


def model_load():

    objectModel = model
    sys.path.append('tensorflow/models/research/')
    sys.path.append('tensorflow/models/research/object_detection/')
    objectModel.parser = argparse.ArgumentParser()

    #### Initialize TensorFlow model ####

    # This is needed since the working directory is the object_detection folder.
    sys.path.append('..')
    sys.path.append('/content/models/research/object_detection')

    # Import utilites

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'models/ssdlite_mobilenet_v2_coco_2018_05_09'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 90

    ## Load the label map.
    # Label maps map indices to category names, so that when the convolution
    # network predicts `5`, we know that this corresponds to `airplane`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    objectModel.category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        objectModel.sess = tf.compat.v1.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    objectModel.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    objectModel.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    objectModel.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    objectModel.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    objectModel.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return objectModel
