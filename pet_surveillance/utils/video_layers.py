import numpy as np
import cv2

def boolean_mask_overlay(frame, mask):  

    # Create a red image.    
    uniform_color = np.ones_like(frame)    
    uniform_color[:, :] = [0, 0, 255] 

    
    # Creating semi transparent mask
    mask = mask.astype(np.uint8)  #convert to an unsigned byte
    mask *= 128
    
    # Black-out the area behind the floor
    frame_bg = cv2.bitwise_and(frame.copy(),frame.copy(),mask = cv2.bitwise_not(mask))

    
    # Mask out the floor from the color image.
    uniform_color_bg = cv2.bitwise_and(uniform_color,uniform_color,mask = mask)


    # Update the original image
    return cv2.add(frame_bg, uniform_color_bg)