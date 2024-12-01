import torch 
import torchvision.transforms as transforms
from PIL import Image 

class StyleTransferModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path).to(self.device)

    def apply_style(self, content_image, style_image):
        '''Style transfer logic'''
        ...

