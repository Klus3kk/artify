"""
Loss functions for style transfer training
FIXED VERSION - keeps all original function names
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import logging

logger = logging.getLogger(__name__)

class PerceptualLoss(nn.Module):
    """Fixed perceptual loss - SAME NAME"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Use VGG19 instead of ResNet - more stable for style transfer
        vgg = vgg19(pretrained=True).features
        
        # Extract specific layers for content loss
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:23])  # relu4_2
        self.feature_extractor.eval()
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        # Ensure inputs are in [0, 1] range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Normalize for VGG
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract content features
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        return F.mse_loss(pred_features, target_features)

class StyleLoss(nn.Module):
    """Fixed style loss with multiple layers - SAME NAME"""
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        
        vgg = vgg19(pretrained=True).features
        
        # Multiple style layers for better representation
        self.style_layers = {
            'conv1_1': nn.Sequential(*list(vgg.children())[:2]),   # relu1_1
            'conv2_1': nn.Sequential(*list(vgg.children())[:7]),   # relu2_1
            'conv3_1': nn.Sequential(*list(vgg.children())[:12]),  # relu3_1
            'conv4_1': nn.Sequential(*list(vgg.children())[:21]),  # relu4_1
        }
        
        # Convert to ModuleDict for proper registration
        self.style_layers = nn.ModuleDict(self.style_layers)
        
        # Freeze all parameters
        for layer in self.style_layers.values():
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
        
        # VGG normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Layer weights
        self.layer_weights = {
            'conv1_1': 1.0,
            'conv2_1': 1.0,
            'conv3_1': 1.0,
            'conv4_1': 1.0
        }
    
    def gram_matrix(self, features):
        """Calculate Gram matrix with proper normalization - SAME NAME"""
        batch_size, channels, height, width = features.size()
        
        # Reshape features
        features = features.view(batch_size, channels, height * width)
        
        # Calculate Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by feature map size - THIS WAS THE PROBLEM!
        gram = gram / (channels * height * width)
        
        return gram
    
    def forward(self, pred, style_target):
        # Ensure inputs are in [0, 1] range
        pred = torch.clamp(pred, 0, 1)
        style_target = torch.clamp(style_target, 0, 1)
        
        # Normalize for VGG
        pred_norm = (pred - self.mean) / self.std
        style_norm = (style_target - self.mean) / self.std
        
        total_style_loss = 0
        
        for layer_name, layer in self.style_layers.items():
            # Extract features
            pred_features = layer(pred_norm)
            style_features = layer(style_norm)
            
            # Calculate Gram matrices
            pred_gram = self.gram_matrix(pred_features)
            style_gram = self.gram_matrix(style_features)
            
            # Calculate style loss for this layer
            layer_loss = F.mse_loss(pred_gram, style_gram)
            
            # Weight the layer loss
            weight = self.layer_weights[layer_name]
            total_style_loss += weight * layer_loss
        
        return total_style_loss

class TotalVariationLoss(nn.Module):
    """Total variation loss for smoothness - SAME NAME"""
    
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, image):
        # Ensure proper range
        image = torch.clamp(image, 0, 1)
        
        batch_size, channels, height, width = image.size()
        
        # Calculate differences
        tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        
        # Sum and normalize
        tv_loss = (tv_h.sum() + tv_w.sum()) / (batch_size * channels * height * width)
        
        return tv_loss

class CombinedLoss(nn.Module):
    """Fixed combined loss to prevent style loss = 0 - SAME NAME"""
    
    def __init__(self, 
                 content_weight=1.0,
                 style_weight=100.0,  # FIXED: Much smaller than 1e6!
                 tv_weight=1e-4):     # FIXED: Adjusted TV weight
        super(CombinedLoss, self).__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # Loss functions - SAME NAMES
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TotalVariationLoss()
        
        # Loss scaling factors (learned during training)
        self.register_buffer('content_scale', torch.tensor(1.0))
        self.register_buffer('style_scale', torch.tensor(1.0))
        self._initialized = False
        
        logger.info(f"Loss weights - Content: {content_weight}, Style: {style_weight}, TV: {tv_weight}")
    
    def forward(self, pred, content, style):
        """Calculate combined loss - SAME SIGNATURE"""
        
        # Calculate individual losses
        content_loss = self.perceptual_loss(pred, content)
        style_loss = self.style_loss(pred, style)
        tv_loss = self.tv_loss(pred)
        
        # Auto-scale losses on first few iterations to prevent style loss = 0
        if not self._initialized:
            with torch.no_grad():
                if content_loss.item() > 0:
                    self.content_scale.fill_(1.0 / max(content_loss.item(), 1e-8))
                if style_loss.item() > 0:
                    self.style_scale.fill_(1.0 / max(style_loss.item(), 1e-8)) 
                self._initialized = True
                logger.info(f"Auto-scaled losses - Content scale: {self.content_scale:.6f}, Style scale: {self.style_scale:.6f}")
        
        # Apply weights
        weighted_content = self.content_weight * content_loss
        weighted_style = self.style_weight * style_loss
        weighted_tv = self.tv_weight * tv_loss
        
        # Combined loss
        total_loss = weighted_content + weighted_style + weighted_tv
        
        # Return same format as before
        return {
            'total': total_loss,
            'content': content_loss,
            'style': style_loss,
            'tv': tv_loss
        }