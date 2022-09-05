
from ..utils.paths import make_dir_function, is_valid
from git import Repo
import gdown
import torch
from torchvision import io
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
import cv2

import subprocess
import sys

from typing import Union


local_dir = make_dir_function()
repo_dir = local_dir('tmp', 'semantic_segmentation')
git_url = 'https://github.com/sithu31296/semantic-segmentation'



def _download_library() -> None:
    """
    For some mysterious reason, the Semsec library, available on GitHub, cannot be loaded if installed directly from requirements.txt, it has to be downloaded locally in order to be used.
    """
    repo_dir.parent.mkdir(exist_ok=True, parents=True)
    Repo.clone_from(git_url, repo_dir)



def _install_library() -> None :
    assert is_valid(repo_dir)

    python = sys.executable
    subprocess.check_call([python, 
        "-m",  "pip", "install",  "-e",  str(repo_dir)
        ], stdout=subprocess.DEVNULL)

if not is_valid(repo_dir):
    _download_library()
    _install_library()

if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))

from semseg.models import *
from semseg.datasets import *
palette = eval('ADE20K').PALETTE

class Segformer():

    model_dir = local_dir('models','segformer')
    model_url = 'https://drive.google.com/uc?id=1-OmW3xRD3WAbJTzktPC-VMOF5WMsN8XT'   
    segformer_path = model_dir.joinpath('segformer.b3.ade.pth')


    def __init__(self):
        self.model =  None

        if not self.segformer_path.is_file():
            self._download_model()

        self._load_model()


    def _download_model(self)-> None:
        self.model_dir.mkdir(exist_ok=True, parents=True)
        gdown.download(self.model_url, str(self.segformer_path), quiet=False)


    def _load_model(self) -> None:

        model = eval('SegFormer')(
            backbone='MiT-B3',
            num_classes=150
        )

        try:
            model.load_state_dict(torch.load(self.segformer_path, map_location='cpu'))
        except:
            print("Download a pretrained model's weights from the result table.")
        
        model.eval()

        self.model = model

        print('Loaded SegFormer Model.')


    def _load_image_from_file(self, image_path:Union[str,Path])-> torch.Tensor:
        image = io.read_image(str(image_path))

        if str(image_path).endswith('png'):
            image = image[:3,:,:]      
        
        return image

    def _load_image_from_np_array(self, image_array:np.ndarray)-> torch.Tensor:
        

        #(720, 1280, 3)
        
        if image_array.shape[2] == 4:
            image_array = image_array[:,:,:3]
       
        image_array = np.rollaxis(image_array, 2)

        image = torch.from_numpy(image_array)    
        return image
    
    
    def _preprocess_image(self, image:torch.Tensor) -> torch.Tensor:
        # resize
        image_ = T.Resize((512, 512))(image)
        # scale to [0.0, 1.0]
        image_ = image_.float() / 255
        # normalize
        image_ = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image_)
        # add batch size
        image_ = image_.unsqueeze(0)
        return image_

    def _inference(self, image:torch.Tensor) -> torch.Tensor:  

        # Forward
        with torch.inference_mode():
            seg = self.model(image)

        # Prediction
        seg = seg.softmax(1).argmax(1).to(int) 
        # recolor
        seg_map = palette[seg].squeeze().to(torch.uint8)

        return seg_map


    def _show_image(self, image: torch.Tensor) -> np.ndarray:
        return image
        

    def _interpolate(self, image:np.ndarray, size:tuple) -> np.ndarray:
        return cv2.resize(image, dsize=size, interpolation=cv2.INTER_NEAREST)

    def predict_labels(
        self, image:Union[str,Path, np.ndarray], labels_path:Union[str,Path]=''
    ) -> Image.Image:


        # Str
        if (type(labels_path) == str) and (labels_path):
            labels_path = Path(labels_path)                    
        
        # Image
        if type(image) in [str,Path]:
            img = self._load_image_from_file(image)
        else:
            img = self._load_image_from_np_array(image)


        # Original_size
        original_size = tuple(img.shape[1:])  
        original_size = original_size[::-1]   


        img = self._preprocess_image(img)
        labels = self._inference(img)
        labels = labels.numpy()
        labels = self._interpolate(labels, original_size)

        return labels

    def detect_floor(self, labels:Union[str,Path, Image.Image])->np.array:
        
        if type(labels) in [str, Path]:
            img  = Image.open(labels)
        else:
            img = labels


        # Convert to array
        img = np.asarray(img)

        # Find Unique Colors
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

        # Find the floor
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

        # Finding coordinates of floor pixels
        indices = np.where(np.all(img == floor_color, axis=-1))

        # Creating a mask.
        mask = np.zeros(img.shape[:-1], bool)

        mask[indices] = True

        return mask
        

    


  

    




