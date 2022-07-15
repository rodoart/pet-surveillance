import cv2
from numpy import unique, ndarray


def correct_labels(
    img:ndarray
) -> ndarray :
    """Simplifies the colors of an image array of labels or few colors, leaving contiguous colors, in the format required by the `keras_segmentation.` library functions.

    Args:
        img (ndarray): The matrix with the image to transform. Using `cv2.imread` is recommended. It should not have more than 255 different colors.

    Returns:
        img (ndarray): Returns the supplied image with the adjacent colors. Example: if the provided image had three colors [18,20,35], [15,40,155], [230,230,230] it will change them to [0,0,0], [1,1,1], [2,2,2].
    """


    img_reshaped = img.reshape(-1, img.shape[-1])
    unique_colors = unique(img_reshaped, axis=0)

    for k,color in enumerate(unique_colors):
        img_reshaped[img_reshaped == color] = k
    
    return img_reshaped.reshape(img.shape)
            
    

