"""
Loss functions for style transfer training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class PerceptualLoss(nn.Module):
    """Perceptual loss using ResNet50 features (Adobe/Google standard)"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Load ResNet50 - industry standard
        resnet = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers
        self.feature_extractor.eval()
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # ResNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def forward(self, pred, target):
        # Move normalization tensors to correct device
        if self.mean.device != pred.device:
            self.mean = self.mean.to(pred.device)
            self.std = self.std.to(pred.device)
        
        # Normalize inputs for ResNet
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        return F.mse_loss(pred_features, target_features)

class StyleLoss(nn.Module):
    """Gram matrix based style loss using ResNet50"""
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        
        # Load ResNet50 - industry standard  
        resnet = resnet50(pretrained=True)
        
        # Use intermediate layers for style
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # First few layers
        self.layer2 = nn.Sequential(*list(resnet.children())[:6])  # Mid layers
        self.layer3 = nn.Sequential(*list(resnet.children())[:7])  # Higher layers
        
        # Freeze parameters
        for layer in [self.layer1, self.layer2, self.layer3]:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
        
        # ResNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def gram_matrix(self, features):
        """Compute Gram matrix"""
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        
        # Compute Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by number of elements
        return gram / (channels * height * width)
    
    def forward(self, pred, style):
        # Move normalization tensors to correct device
        if self.mean.device != pred.device:
            self.mean = self.mean.to(pred.device)
            self.std = self.std.to(pred.device)
        
        # Normalize inputs for ResNet
        pred_norm = (pred - self.mean) / self.std
        style_norm = (style - self.mean) / self.std
        
        # Extract features from multiple layers
        pred_feat1 = self.layer1(pred_norm)
        pred_feat2 = self.layer2(pred_norm)
        pred_feat3 = self.layer3(pred_norm)
        
        style_feat1 = self.layer1(style_norm)
        style_feat2 = self.layer2(style_norm)
        style_feat3 = self.layer3(style_norm)
        
        # Calculate style loss using Gram matrices
        style_loss = 0
        
        # Layer 1 loss
        pred_gram1 = self.gram_matrix(pred_feat1)
        style_gram1 = self.gram_matrix(style_feat1)
        style_loss += F.mse_loss(pred_gram1, style_gram1)
        
        # Layer 2 loss
        pred_gram2 = self.gram_matrix(pred_feat2)
        style_gram2 = self.gram_matrix(style_feat2)
        style_loss += F.mse_loss(pred_gram2, style_gram2)
        
        # Layer 3 loss
        pred_gram3 = self.gram_matrix(pred_feat3)
        style_gram3 = self.gram_matrix(style_feat3)
        style_loss += F.mse_loss(pred_gram3, style_gram3)
        
        return style_loss

class TotalVariationLoss(nn.Module):
    """Total variation loss for smoothness"""
    
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, image):
        batch_size, channels, height, width = image.size()
        
        # Calculate differences
        tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        
        # Sum and normalize
        tv_loss = (tv_h.sum() + tv_w.sum()) / (batch_size * channels * height * width)
        
        return tv_loss

class CombinedLoss(nn.Module):
    """Combined loss for style transfer training"""
    
    def __init__(self, 
                 content_weight=1.0,
                 style_weight=1e6,
                 tv_weight=1e-6):
        super(CombinedLoss, self).__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # Loss functions
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TotalVariationLoss()
    
    def forward(self, pred, content, style):
        """Calculate combined loss"""
        
        # Content loss (perceptual)
        content_loss = self.perceptual_loss(pred, content)
        
        # Style loss (Gram matrices)
        style_loss = self.style_loss(pred, style)
        
        # Total variation loss (smoothness)
        tv_loss = self.tv_loss(pred)
        
        # Combined loss
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss + 
                     self.tv_weight * tv_loss)
        
        return {
            'total': total_loss,
            'content': content_loss,
            'style': style_loss,
            'tv': tv_loss
        }