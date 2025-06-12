"""
Updated StyleTransferModel for Artify
Supports multiple storage backends and optimized small models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Union

from utilities.Logger import Logger

logger = Logger.setup_logger(log_file="artify.log", log_level=logging.INFO)

class ResidualBlock(nn.Module):
    """Residual block for style transfer network"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + residual

class FastStyleNetwork(nn.Module):
    """Optimized fast style transfer network - Small and efficient"""
    
    def __init__(self, style_dim=128):
        super(FastStyleNetwork, self).__init__()
        self.style_dim = style_dim
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 9, 1, 4),
            nn.Tanh()
        )
        
        # Style encoder for AdaIN
        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, style_dim, 1, 1, 0)
        )
    
    def forward(self, content, style):
        # Encode content
        content_features = self.encoder(content)
        
        # Encode style
        style_features = self.style_encoder(style)
        style_features = style_features.expand_as(content_features)
        
        # Apply simple style modulation (lightweight AdaIN)
        styled_features = self.adaptive_instance_norm(content_features, style_features)
        
        # Process through residual blocks
        styled_features = self.residual_blocks(styled_features)
        
        # Decode to final image
        output = self.decoder(styled_features)
        
        return output
    
    def adaptive_instance_norm(self, content_features, style_features):
        """Lightweight AdaIN implementation"""
        
        # Calculate content statistics
        content_mean = content_features.mean(dim=[2, 3], keepdim=True)
        content_std = content_features.std(dim=[2, 3], keepdim=True) + 1e-8
        
        # Calculate style statistics
        style_mean = style_features.mean(dim=[2, 3], keepdim=True)
        style_std = style_features.std(dim=[2, 3], keepdim=True) + 1e-8
        
        # Normalize content and apply style statistics
        normalized_content = (content_features - content_mean) / content_std
        styled_features = normalized_content * style_std + style_mean
        
        return styled_features

class ModelLoader:
    """Handles loading models from multiple storage backends"""
    
    def __init__(self, storage_backend: str = "local"):
        self.storage_backend = storage_backend
        self.cache_dir = Path("models/cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize storage clients
        self._init_storage_clients()
    
    def _init_storage_clients(self):
        """Initialize storage backend clients"""
        
        self.clients = {}
        
        # S3 client
        try:
            import boto3
            self.clients['s3'] = boto3.client('s3')
            self.s3_bucket = os.getenv('ARTIFY_S3_BUCKET', 'artify-models')
        except ImportError:
            logger.warning("boto3 not available - S3 storage disabled")
        
        # HuggingFace client
        try:
            from huggingface_hub import hf_hub_download
            self.clients['huggingface'] = hf_hub_download
            self.hf_repo = os.getenv('ARTIFY_HF_REPO', 'artify-community/style-models')
        except ImportError:
            logger.warning("huggingface_hub not available - HF storage disabled")
    
    def load_model(self, model_name: str) -> str:
        """Load model from configured storage backend"""
        
        if self.storage_backend == "local":
            return self._load_local_model(model_name)
        elif self.storage_backend == "s3":
            return self._load_s3_model(model_name)
        elif self.storage_backend == "huggingface":
            return self._load_hf_model(model_name)
        else:
            raise ValueError(f"Unknown storage backend: {self.storage_backend}")
    
    def _load_local_model(self, model_name: str) -> str:
        """Load model from local storage"""
        
        # Try different local paths
        local_paths = [
            Path(f"models/{model_name}"),
            Path(f"models/{model_name}_best.pth"),
            Path(f"models/{model_name}_optimized.pth"),
            Path(f"models/{model_name}_fast_network_best.pth"),
            Path(f"models/{model_name}_fast_network_optimized.pth")
        ]
        
        for path in local_paths:
            if path.exists():
                logger.info(f"Loaded local model: {path}")
                return str(path)
        
        raise FileNotFoundError(f"Local model not found: {model_name}")
    
    def _load_s3_model(self, model_name: str) -> str:
        """Load model from S3"""
        
        if 's3' not in self.clients:
            raise RuntimeError("S3 client not available")
        
        # Check cache first
        cache_path = self.cache_dir / f"s3_{model_name}"
        if cache_path.exists():
            logger.info(f"Using cached S3 model: {cache_path}")
            return str(cache_path)
        
        # Download from S3
        s3_key = f"models/{model_name}"
        try:
            self.clients['s3'].download_file(self.s3_bucket, s3_key, str(cache_path))
            logger.info(f"Downloaded S3 model: {s3_key}")
            return str(cache_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download from S3: {e}")
    
    def _load_hf_model(self, model_name: str) -> str:
        """Load model from HuggingFace Hub"""
        
        if 'huggingface' not in self.clients:
            raise RuntimeError("HuggingFace client not available")
        
        try:
            local_path = self.clients['huggingface'](
                repo_id=self.hf_repo,
                filename=model_name,
                cache_dir=str(self.cache_dir / "hf_cache")
            )
            logger.info(f"Downloaded HF model: {model_name}")
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download from HuggingFace: {e}")

class StyleTransferModel:
    """Production-ready StyleTransferModel with multiple storage backends"""
    
    def __init__(self, 
                 storage_backend: str = "local",
                 model_cache_size: int = 3,
                 device: str = "auto"):
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"StyleTransferModel initialized on {self.device}")
        logger.info(f"Storage backend: {storage_backend}")
        
        # Initialize model loader
        self.model_loader = ModelLoader(storage_backend)
        
        # Model cache for fast switching between styles
        self.model_cache = {}
        self.cache_size = model_cache_size
        self.cache_order = []
        
        # Current loaded model
        self.current_model = None
        self.current_style = None
        
        # Performance mode
        self.mode = "fast"  # "fast" or "quality"
    
    def set_fast_mode(self):
        """Switch to fast mode (optimized for speed)"""
        self.mode = "fast"
        logger.info("Switched to fast mode (~0.05s per image)")
    
    def set_quality_mode(self):
        """Switch to quality mode (optimized for quality)"""
        self.mode = "quality"
        logger.info("Switched to quality mode (~0.3s per image)")
    
    def load_style_model(self, style_category: str) -> nn.Module:
        """Load model for specific style category with caching"""
        
        # Check cache first
        if style_category in self.model_cache:
            logger.info(f"Using cached model for {style_category}")
            return self.model_cache[style_category]
        
        # Determine model filename based on mode
        if self.mode == "fast":
            model_filename = f"{style_category}_fast_network_best.pth"
        else:
            model_filename = f"{style_category}_quality_network_best.pth"
        
        # Try optimized version first
        optimized_filename = model_filename.replace("_best.pth", "_optimized.pth")
        
        try:
            model_path = self.model_loader.load_model(optimized_filename)
            logger.info(f"Loaded optimized model: {optimized_filename}")
        except (FileNotFoundError, RuntimeError):
            try:
                model_path = self.model_loader.load_model(model_filename)
                logger.info(f"Loaded standard model: {model_filename}")
            except (FileNotFoundError, RuntimeError) as e:
                raise RuntimeError(f"Model not found for style '{style_category}': {e}")
        
        # Load model
        model = self._load_model_from_path(model_path)
        
        # Cache management
        self._update_cache(style_category, model)
        
        return model
    
    def _load_model_from_path(self, model_path: str) -> nn.Module:
        """Load PyTorch model from file path"""
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            model = FastStyleNetwork()
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the file contains only the state dict
                model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Get model size
            model_size = Path(model_path).stat().st_size / (1024 * 1024)
            logger.info(f"Model loaded: {model_size:.2f}MB")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def _update_cache(self, style_category: str, model: nn.Module):
        """Update model cache with LRU eviction"""
        
        # Remove from cache if already exists
        if style_category in self.model_cache:
            del self.model_cache[style_category]
            self.cache_order.remove(style_category)
        
        # Add to cache
        self.model_cache[style_category] = model
        self.cache_order.append(style_category)
        
        # Evict oldest if cache is full
        while len(self.model_cache) > self.cache_size:
            oldest_style = self.cache_order.pop(0)
            del self.model_cache[oldest_style]
            logger.info(f"Evicted {oldest_style} from model cache")
    
    def apply_style(self, 
                   content_image: Union[Image.Image, torch.Tensor], 
                   style_category: str = "impressionism",
                   style_image: Image.Image = None) -> Image.Image:
        """
        Apply style transfer to content image
        
        Args:
            content_image: Input content image (PIL Image or tensor)
            style_category: Style category name (e.g., "impressionism") 
            style_image: Optional specific style image (uses random from category if None)
        
        Returns:
            Stylized PIL Image
        """
        
        start_time = time.time()
        
        # Load appropriate model
        if self.current_style != style_category:
            self.current_model = self.load_style_model(style_category)
            self.current_style = style_category
        
        # Prepare content tensor
        content_tensor = self._prepare_input(content_image)
        
        # Prepare style tensor
        if style_image is not None:
            style_tensor = self._prepare_input(style_image)
        else:
            # Use a dummy style tensor (the model is trained for this specific style)
            style_tensor = content_tensor  # Models are style-specific
        
        # Apply style transfer
        with torch.no_grad():
            stylized_tensor = self.current_model(content_tensor, style_tensor)
            
            # Clamp values to valid range
            stylized_tensor = torch.clamp(stylized_tensor, 0, 1)
        
        # Convert back to PIL Image
        stylized_image = self._tensor_to_pil(stylized_tensor.squeeze(0))
        
        inference_time = time.time() - start_time
        logger.info(f"Style transfer completed in {inference_time:.3f}s ({self.mode} mode)")
        
        return stylized_image
    
    def _prepare_input(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Prepare input image as tensor"""
        
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            tensor = transform(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("Input must be PIL Image or torch.Tensor")
        
        return tensor.to(self.device)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        
        # Move to CPU and convert to PIL
        tensor = tensor.cpu()
        transform = transforms.ToPILImage()
        return transform(tensor)
    
    def get_available_styles(self) -> list:
        """Get list of available style categories"""
        
        # This would ideally check the storage backend for available models
        # For now, return common styles
        return [
            "impressionism", "cubism", "abstract", "expressionism", 
            "baroque", "post-impressionism", "surrealism", "pop-art"
        ]
    
    def get_model_info(self, style_category: str) -> dict:
        """Get information about a specific model"""
        
        try:
            model = self.load_style_model(style_category)
            
            # Calculate model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "style_category": style_category,
                "mode": self.mode,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device),
                "storage_backend": self.model_loader.storage_backend,
                "cached": style_category in self.model_cache
            }
            
        except Exception as e:
            return {
                "style_category": style_category,
                "error": str(e),
                "available": False
            }
    
    def warm_up_cache(self, style_categories: list):
        """Pre-load models into cache for faster inference"""
        
        logger.info(f"Warming up cache with {len(style_categories)} models...")
        
        for style_category in style_categories[:self.cache_size]:
            try:
                self.load_style_model(style_category)
                logger.info(f"✓ Cached {style_category}")
            except Exception as e:
                logger.warning(f"✗ Failed to cache {style_category}: {e}")
        
        logger.info("Cache warm-up complete")
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        
        self.model_cache.clear()
        self.cache_order.clear()
        self.current_model = None
        self.current_style = None
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
    
    # Legacy compatibility methods
    def ensure_model(self, style_category: str):
        """Legacy compatibility - ensure model is available"""
        try:
            self.load_style_model(style_category)
            return True
        except Exception as e:
            logger.error(f"Failed to ensure model {style_category}: {e}")
            return False
    
    def load_model(self, model_path: str):
        """Legacy compatibility - load model from path"""
        logger.info(f"Using new architecture - ignoring legacy load_model call")
        return True
    
    def train_model(self, content_image, style_image, output_path, **kwargs):
        """Legacy compatibility - training not supported in inference model"""
        logger.warning("Training not supported in production model - use training/train_fast_networks.py")
        return self.apply_style(content_image, "impressionism")

# Example usage and testing
if __name__ == "__main__":
    # Test the updated model
    print("Testing StyleTransferModel with multiple storage backends...")
    
    # Test local storage
    model_local = StyleTransferModel(storage_backend="local")
    print(f"Local model initialized on {model_local.device}")
    
    # Test available styles
    styles = model_local.get_available_styles()
    print(f"Available styles: {styles[:3]}...")
    
    # Test model info
    if styles:
        info = model_local.get_model_info(styles[0])
        print(f"Model info for {styles[0]}: {info}")
    
    print("✓ StyleTransferModel test complete!")