import torch 
import torchvision.transforms as transforms
from PIL import Image 
from torchvision.models import vgg19


class StyleTransferModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_pretrained_model()

    def _load_pretrained_model(self):
        '''Using VGG-19 for feature extraction'''
        model = vgg19(pretrained=True).features
        for param in model.parameters():
            param.requires_grad = False 
        return model.to(self.device)

    def apply_style(self, content_image, style_image):
        '''Style transfer logic'''
        ...

