
from ..utils.paths import make_dir_function, is_valid
from git import Repo
import gdown
import torch
from torchvision import io
from torchvision import transforms as T
from PIL import Image
from pathlib import Path

import subprocess
import sys


local_dir = make_dir_function()
repo_dir = local_dir('tmp', 'semantic_segmentation')
git_url = 'https://github.com/sithu31296/semantic-segmentation'



def _download_library():
    """
    For some mysterious reason, the Semsec library, available on GitHub, cannot be loaded if installed directly from requirements.txt, it has to be downloaded in order to be used.
    """
    repo_dir.parent.mkdir(exist_ok=True, parents=True)
    Repo.clone_from(git_url, repo_dir)



def _install_library():
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


    def _download_model(self):
        self.model_dir.mkdir(exist_ok=True, parents=True)
        gdown.download(self.model_url, str(self.segformer_path), quiet=False)


    def _load_model(self):

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

        print('Loaded Model')


    def _load_image(self, image_path):
        image = io.read_image(str(image_path))
        if str(image_path).endswith('png'):
            image = image[:3,:,:]
        
        return image

    def _preprocess_image(self, image):
        # resize
        image_ = T.Resize((512, 512))(image)
        # scale to [0.0, 1.0]
        image_ = image_.float() / 255
        # normalize
        image_ = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image_)
        # add batch size
        image_ = image_.unsqueeze(0)

        return image_

    def _inference(self, image):  
        with torch.inference_mode():
            seg = self.model(image)

        seg = seg.softmax(1).argmax(1).to(int) 
        seg_map = palette[seg].squeeze().to(torch.uint8)

        return seg_map


    def _show_image(self, image):
        if image.shape[2] != 3: image = image.permute(1, 2, 0)
        image = Image.fromarray(image.numpy())
        return image


    def predict_labels(self, image_path):
        image = self._load_image(image_path)
        image = self._preprocess_image(image)
        labels = self._inference(image)
        img_labels = self._show_image(labels)
        return img_labels


  

    




