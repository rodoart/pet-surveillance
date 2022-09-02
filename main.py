import cv2
import numpy as np
from Pet_detector import setupCamera
from pet_surveillance.models import segformer
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

def run():

  #if(motion_detector('test_video', 'data/VideoTests/test.mp4')):
  setupCamera('test_video', 'data/VideoTests/test.mp4')
  image_path = 'data/processed/semantic_segmentation/unity_residential_interiors/train_images/7.png'
  


  obe = segformer.Segformer()

  image = Image.open(image_path)
  image_array = np.asarray(image)
  img = obe.predict_labels(image_array)

  print(np.sum(obe.detect_floor(img)))

if __name__ == '__main__':
    run()



