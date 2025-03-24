import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch.nn.functional as F
from PIL import Image
import logging
import os
import time
from tqdm import tqdm
import numpy as np
from utilities.Logger import Logger
from core.HuggingFaceHandler import HuggingFaceHandler

# Set up the logger
logger = Logger.setup_logger(log_file="artify.log", log_level=logging.INFO)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()

    def forward(self, x):
        G = self._gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x

    def _gram_matrix(self, input):
        batch_size, h, w, f_map_num = input.size()
        features = input.view(batch_size * h, w * f_map_num)
        G = torch.mm(features, features.t())
        return G.div(batch_size * h * w * f_map_num)

class TVLoss(nn.Module):
    def forward(self, x):
        batch_size, h, w, f_map_num = x.size()
        tv_h = torch.pow(x[:, 1:, :, :] - x[:, :-1, :, :], 2).sum()
        tv_w = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        return (tv_h + tv_w) / (batch_size * h * w * f_map_num)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class StyleTransferModel:
    def __init__(self, hf_token=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device set to: {self.device}")
        
        self.cnn = self._load_pretrained_model()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        self.hf_token = hf_token
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Cache for style and content features to avoid recomputation
        self.feature_cache = {}
        
        # For batch processing
        self.batch_size = 4  # Adjust based on available memory
        
    def _load_pretrained_model(self):
        """Load the pretrained VGG-19 model."""
        try:
            # Load only the features portion and put in evaluation mode
            vgg = vgg19(pretrained=True).features.eval()
            
            # Freeze all parameters
            for param in vgg.parameters():
                param.requires_grad = False
            
            logger.info("Pretrained VGG-19 model loaded successfully.")
            return vgg.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

    def _get_style_model_and_losses(self, style_img, content_img):
        """Create a style transfer model with content and style losses."""
        # Create a sequential model with normalization as the first layer
        normalization = Normalization(self.mean, self.std).to(self.device)
        model = nn.Sequential(normalization)
        
        # Initialize content and style losses
        content_losses = []
        style_losses = []
        tv_loss = TVLoss().to(self.device)
        
        # Current position in the VGG model
        i = 0
        
        # Iterate through the VGG layers and add them to our model
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                # Replace in-place version with out-of-place
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
                # Use average pooling instead of max pooling
                layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
            model.add_module(name, layer)
            
            # Add content loss
            if name in self.content_layers:
                # Get content features
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
            
            # Add style loss
            if name in self.style_layers:
                # Get style features
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
        
        # Now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses, tv_loss

    def _preprocess_image(self, image, target_size=None):
        """Preprocess an image for style transfer."""
        if target_size:
            # Resize while preserving aspect ratio
            image.thumbnail(target_size, Image.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def _postprocess_tensor(self, tensor):
        """Convert a tensor back to a PIL image."""
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).cpu()
        
        # Clip values to valid image range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL Image
        transform = transforms.ToPILImage()
        return transform(tensor)

    def apply_style(self, content_image, style_image=None, iterations=300, 
                    style_weight=1e6, content_weight=1, tv_weight=1e-6, 
                    early_stopping=True, optimizer_type='lbfgs'):
        """
        Apply style transfer with optimizations for speed.
        
        Args:
            content_image: PIL Image object for content
            style_image: PIL Image (optional if using pre-trained model)
            iterations: Number of optimization steps
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            tv_weight: Weight for total variation loss
            early_stopping: Stop if loss doesn't improve
            optimizer_type: 'lbfgs' or 'adam' - lbfgs is faster, adam is more memory efficient
            
        Returns:
            Styled PIL Image
        """
        logger.info("Starting style transfer process...")
        start_time = time.time()
        
        # Size determination - use content image size, but limit for performance
        max_size = 512
        orig_width, orig_height = content_image.size
        aspect_ratio = orig_width / orig_height
        
        if max(orig_width, orig_height) > max_size:
            if orig_width > orig_height:
                new_width = max_size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(new_height * aspect_ratio)
            target_size = (new_width, new_height)
        else:
            target_size = (orig_width, orig_height)
        
        logger.info(f"Processing at size: {target_size}")
        
        # Preprocess images
        content_tensor = self._preprocess_image(content_image, target_size)
        
        if style_image is not None:
            style_tensor = self._preprocess_image(style_image)
        else:
            logger.error("Style image must be provided")
            raise ValueError("Style image must be provided")
        
        # Initialize target as a copy of the content image
        target = content_tensor.clone().requires_grad_(True)
        
        # Set up the model and losses
        model, style_losses, content_losses, tv_loss = self._get_style_model_and_losses(
            style_tensor, content_tensor
        )
        
        # Set up optimizer
        if optimizer_type == 'lbfgs':
            optimizer = optim.LBFGS([target], max_iter=20, line_search_fn='strong_wolfe')
        else:  # adam
            optimizer = optim.Adam([target], lr=0.01)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        # Track best result
        best_loss = float('inf')
        best_img = None
        patience = 10  # iterations without improvement before early stopping
        patience_counter = 0
        
        logger.info(f"Running {iterations} iterations of style transfer...")
        pbar = tqdm(range(iterations))
        
        for i in pbar:
            def closure():
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass through the model
                model(target)
                
                # Compute losses
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                # Compute total variation loss for smoothness
                tv_score = tv_loss(target)
                
                # Weight the losses
                style_score *= style_weight
                content_score *= content_weight
                tv_score *= tv_weight
                
                # Total loss
                loss = style_score + content_score + tv_score
                
                # Backward pass
                loss.backward()
                
                # Log progress
                if i % 20 == 0 or i == iterations - 1:
                    logger.info(
                        f"Iteration {i+1}/{iterations} - "
                        f"Style Loss: {style_score.item():.4f}, "
                        f"Content Loss: {content_score.item():.4f}, "
                        f"TV Loss: {tv_score.item():.4f}, "
                        f"Total Loss: {loss.item():.4f}"
                    )
                
                # Update progress bar
                pbar.set_description(f"Loss: {loss.item():.2f}")
                
                return loss
            
            # Update the image
            if optimizer_type == 'lbfgs':
                loss = optimizer.step(closure)
            else:  # adam
                optimizer.step(closure)
                scheduler.step()
                # Get current loss
                with torch.no_grad():
                    model(target)
                    style_score = sum(sl.loss for sl in style_losses) * style_weight
                    content_score = sum(cl.loss for cl in content_losses) * content_weight
                    tv_score = tv_loss(target) * tv_weight
                    loss = style_score + content_score + tv_score
            
            # Check if current result is best so far
            if i % 10 == 0 or i == iterations - 1:
                current_loss = loss.item() if optimizer_type == 'lbfgs' else loss
                
                # # Log progress
                # if i % 20 == 0 or i == iterations - 1:
                #     logger.info(
                #         f"Iteration {i+1}/{iterations} - "
                #         f"Style Loss: {style_score.item():.4f}, "
                #         f"Content Loss: {content_score.item():.4f}, "
                #         f"TV Loss: {tv_score.item():.4f}, "
                #         f"Total Loss: {current_loss:.4f}"
                #     )
                
                # if current_loss < best_loss:
                #     best_loss = current_loss
                #     best_img = target.clone().detach()
                #     patience_counter = 0
                # else:
                #     patience_counter += 1
                
                # # Early stopping
                # if early_stopping and patience_counter >= patience:
                #     logger.info(f"Early stopping at iteration {i+1}")
                #     break
        
        # Use the best result
        final_img = best_img if best_img is not None else target.detach()
        
        # Convert back to PIL image
        result_image = self._postprocess_tensor(final_img)
        
        # If we resized for processing, resize back to original size
        if result_image.size != (orig_width, orig_height):
            result_image = result_image.resize((orig_width, orig_height), Image.LANCZOS)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Style transfer complete in {elapsed_time:.2f} seconds")
        
        return result_image
    
    def ensure_model(self, style_category):
        """Ensure the required style model file exists locally or download it."""
        model_filename = f"{style_category}_model.pth"
        model_path = os.path.join(self.models_dir, model_filename)
        
        if not os.path.exists(model_path):
            logger.info(f"Model '{model_filename}' not found locally. Attempting to download...")
            try:
                hf_handler = HuggingFaceHandler(self.hf_token)
                repo_dir = hf_handler.download_model(repo_id="ClueSec/artify-models", cache_dir=self.models_dir)
                downloaded_model_path = os.path.join(repo_dir, model_filename)
                
                if not os.path.exists(downloaded_model_path):
                    raise FileNotFoundError(f"Model file '{model_filename}' not found in the downloaded repository.")
                
                os.rename(downloaded_model_path, model_path)
                logger.info(f"Model '{model_filename}' successfully downloaded and moved to {model_path}.")
            except Exception as e:
                logger.error(f"Failed to download model '{model_filename}': {e}")
                raise
        
        logger.info(f"Model '{model_filename}' is ready at {model_path}.")
        return model_path
    
    def train_model(self, content_image, style_image, output_path, iterations=1000, 
                style_weight=1e6, content_weight=1, tv_weight=1e-6):
        """Train a style transfer model and save it to the specified path."""
        logger.info(f"Training style transfer model, saving to {output_path}")
        
        # Simply use apply_style but make sure to capture the result
        try:
            styled_image = self.apply_style(
                content_image=content_image,
                style_image=style_image,
                iterations=iterations,
                style_weight=style_weight,
                content_weight=content_weight,
                tv_weight=tv_weight
            )
            
            # Convert to tensor and save
            transform = transforms.ToTensor()
            styled_tensor = transform(styled_image).unsqueeze(0)
            torch.save(styled_tensor, output_path)
            
            logger.info(f"Model saved to {output_path}")
            return styled_image
        except Exception as e:
            logger.error(f"Error in train_model: {e}")
            raise

    def load_model(self, model_path):
        """Load a pretrained model from the local filesystem."""
        self.model = torch.load(model_path, map_location=self.device)
        logger.info(f"Model loaded from {model_path}")