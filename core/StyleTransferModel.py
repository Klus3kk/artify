import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import logging
import os


class StyleTransferModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device set to: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        self.model = self._load_pretrained_model()

    def _load_pretrained_model(self):
        try:
            model = vgg19(pretrained=True).features
            for param in model.parameters():
                param.requires_grad = False
            logging.info("Pretrained VGG-19 model loaded successfully.")
            return model.to(self.device)
        except Exception as e:
            logging.error(f"Failed to load pretrained model: {e}")
            raise

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

    def _image_to_tensor(self, image, target_size=(512, 512)):
        """
        Convert a PIL image to a normalized tensor and resize it.
        :param image: PIL Image object.
        :param target_size: Target size to resize the image (default: 512x512).
        :return: Normalized tensor.
        """
        transform = transforms.Compose([
            transforms.Resize(target_size),  # Resize to target size
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
    
    def _calculate_tv_loss(self, tensor):
        diff_x = torch.sum(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]))
        diff_y = torch.sum(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))
        return diff_x + diff_y


    def _gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        return torch.mm(tensor, tensor.t())

    def train_model(self, content_image, style_image, output_path, iterations=300, style_weight=1e6, content_weight=5):
        """
        Train a style transfer model and save it to the specified path.
        :param content_image: PIL Image object for content.
        :param style_image: PIL Image object for style.
        :param output_path: Path to save the trained model.
        :param iterations: Number of training iterations.
        :param style_weight: Weight for style loss.
        :param content_weight: Weight for content loss.
        """
        # Preprocess images
        content_tensor = self._image_to_tensor(content_image).to(self.device)
        style_tensor = self._image_to_tensor(style_image).to(self.device)

        # Initialize target image (clone of content image)
        target = content_tensor.clone().requires_grad_(True)

        # Define optimizer
        optimizer = optim.Adam([target], lr=0.005)

        # Extract features
        style_features = self._extract_features(style_tensor)
        content_features = self._extract_features(content_tensor)

        # Training loop
        print(f"Starting training for {iterations} iterations...")
        for i in range(iterations):
            target_features = self._extract_features(target)
            content_loss = self._calculate_content_loss(content_features, target_features)
            style_loss = self._calculate_style_loss(style_features, target_features)
            tv_loss = self._calculate_tv_loss(target)

            total_loss = style_weight * style_loss + content_weight * content_loss + 1e-5 * tv_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0 or i == 0:
                logging.info(f"\nIteration {i + 1}/{iterations}: \n"
                    f"Content Loss = {content_loss.item():.4f} \n"
                    f"Style Loss = {style_loss.item():.4f} \n"
                    f"TV Loss = {tv_loss.item():.4f} \n"
                    f"Total Loss = {total_loss.item():.4f}")

        torch.save(target, output_path)
        print(f"Model saved to {output_path}")



    def load_model(self, model_path):
        """
        Load a pretrained model from the local filesystem.
        :param model_path: Path to the model file.
        """
        self.model = torch.load(model_path, map_location=self.device)
        print(f"Model loaded from {model_path}")

    def load_model_from_gcloud(self, bucket_name, model_name):
        """
        Load a model dynamically from Google Cloud Storage.
        :param bucket_name: Name of the GCS bucket.
        :param model_name: Name of the model file in GCS.
        """
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_name)

        local_path = f"/tmp/{model_name}"
        blob.download_to_filename(local_path)
        self.load_model(local_path)
        print(f"Model loaded from GCS bucket {bucket_name}, file {model_name}")
