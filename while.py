import cv2
import numpy as np
from Pet_detector import setupCamera
from pet_surveillance.models import segformer
from pet_surveillance.utils.video_layers import boolean_mask_overlay 

def test(k):
    if k==0:
        global count
        count = k
    
    if k%5:
        return count

    count += 1

    return count




def test_2():
    image_path = 'data/processed/semantic_segmentation/unity_residential_interiors/train_images/7.png'
    image_path_2 = 'data\PNG_transparency_demonstration_1.png'
    
    img = cv2.imread(image_path)
    
    # Segformer
    obe = segformer.Segformer()    
    labels = obe.predict_labels(img)

    # Detect floors
    mask = obe.detect_floor(labels)

    img = boolean_mask_overlay(frame=img,mask=mask)

    for image in [labels, img]:
        #transpose(1, 2, 0)
        cv2.imshow('alo', image)
        #waits for user to press any key 
        #(this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0) 
        
        #closing all open windows 
        cv2.destroyAllWindows()


def test_3():

    image_path = 'data/processed/semantic_segmentation/unity_residential_interiors/train_images/7.png'
    
    img = cv2.imread(image_path)
    
    # Segformer
    obe = segformer.Segformer()    
    labels = obe.predict_labels(img)

    img = np.asarray(labels)
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)    
    
    max_num_of_pixels_same_color = 0
    
    # Find pixels in border with same color:    
    for color in colors:
        sum_border = np.sum(np.all(img[-1,:,:] == color, axis=-1))

        # Find the sum of all pixels with the same color
        if sum_border > 0:
            sum_all = np.sum(np.all(img == color, axis=-1))

            if sum_all > max_num_of_pixels_same_color:
                max_num_of_pixels_same_color = sum_all
                floor_color = color

    print(floor_color)



    #print(np.all(img == color, axis=-1))


    
    

        





if __name__=='__main__':
    test_2()







