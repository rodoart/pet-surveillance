from pickletools import uint8
import cv2
import numpy as np
from Pet_detector import setupCamera
from pet_surveillance.models import segformer
from pet_surveillance.utils.video_layers import boolean_mask_overlay
from PIL import Image

def motion_detector(camera_type, testurl):
  
  frame_count = 0
  previous_frame = None
  if(camera_type == 'usb_camera'):
    cap = cv2.VideoCapture(0)
  else:
    cap = cv2.VideoCapture(testurl)
  
  while True:
    frame_count += 1
    ret, frame = cap.read()

    # 1. Load image; convert to RGB
    img_brg = np.array(frame)
    img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

    if ((frame_count % 2) == 0):

      # 2. Prepare image; grayscale and blur
      prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
      prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)
      # 3. Set previous frame and continue if there is None
    if (previous_frame is None):
    # First frame; there is no previous one yet
        previous_frame = prepared_frame
        continue
    
    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    return (bool(diff_frame = cv2.dilate(diff_frame, kernel, 1)))


def test_floor_detection_overlays():
    image_path = 'data/processed/semantic_segmentation/unity_residential_interiors/train_images/7.png'
    
    img = cv2.imread(image_path)
    
    # Segformer
    obe = segformer.Segformer()    
    labels = obe.predict_labels(img)

    # Detect floors
    mask = obe.detect_floor(labels)

    # Create overlays.
    img = boolean_mask_overlay(frame=img,mask=mask)

    
    # Plot
    for image in [labels, img]:
        #transpose(1, 2, 0)
        cv2.imshow('alo', image)
        #waits for user to press any key 
        #(this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0) 
        
        #closing all open windows 
        cv2.destroyAllWindows()


def test_pet_detector():
  #if(motion_detector('test_video', 'data/VideoTests/test.mp4')):
  setupCamera('file', r'data/VideoTests/test.mp4')


def test_numpy_arrays():
  mat = np.ones(shape=(200,150), dtype=np.uint8)
  mat*=255
  mat = mat/3

  
  old_not_labeled_frame = mat
  mean_frame = mat+1


  for step in range(30):
    mat = mat+1
    if step > 3:
      divider = 3
    else:
      divider = step +1
      
    mean_frame = mean_frame*((divider-1)/divider) + old_not_labeled_frame*((1/divider))

    old_not_labeled_frame = mat.copy()


  print(np.uint8(np.round(mean_frame)))


  



def run():
  test_pet_detector()



if __name__ == '__main__':
    run()



