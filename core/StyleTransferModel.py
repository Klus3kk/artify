import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image

class StyleTransferModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_pretrained_model()

    def _load_pretrained_model(self):
        # Load VGG-19 pretrained model for feature extraction
        model = vgg19(pretrained=True).features
        for param in model.parameters():
            param.requires_grad = False
        return model.to(self.device)

    def apply_style(self, content_image, style_image, iterations=300, style_weight=1e6, content_weight=1):
        # Preprocess images
        content_tensor = self._image_to_tensor(content_image).to(self.device)
        style_tensor = self._image_to_tensor(style_image).to(self.device)

        # Initialize target image (clone of content image)
        target = content_tensor.clone().requires_grad_(True)

        # Define optimizer
        optimizer = optim.Adam([target], lr=0.003)

        # Feature maps for content and style
        style_features = self._extract_features(style_tensor)
        content_features = self._extract_features(content_tensor)

        # Compute style and content losses
        for i in range(iterations):
            target_features = self._extract_features(target)
            content_loss = self._calculate_content_loss(content_features, target_features)
            style_loss = self._calculate_style_loss(style_features, target_features)

            total_loss = style_weight * style_loss + content_weight * content_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return self._tensor_to_image(target)

    def _image_to_tensor(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)

    def _tensor_to_image(self, tensor):
        unnormalize = transforms.Normalize(
            mean=[-2.12, -2.04, -1.8],
            std=[4.37, 4.46, 4.44],
        )
        tensor = unnormalize(tensor.squeeze(0))
        return transforms.ToPILImage()(tensor)

    def _extract_features(self, tensor):
        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # Content representation
            "28": "conv5_1",
        }
        features = {}
        x = tensor
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def _calculate_content_loss(self, content_features, target_features):
        return torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

    def _calculate_style_loss(self, style_features, target_features):
        style_loss = 0
        for layer in style_features:
            target_gram = self._gram_matrix(target_features[layer])
            style_gram = self._gram_matrix(style_features[layer])
            _, d, h, w = target_features[layer].size()
            style_loss += torch.mean((target_gram - style_gram) ** 2) / (d * h * w)
        return style_loss

    def _gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        return torch.mm(tensor, tensor.t())
