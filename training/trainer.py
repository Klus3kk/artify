"""
Training logic for style transfer networks
FIXED VERSION - keeps all original function names
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .losses import CombinedLoss

logger = logging.getLogger(__name__)

class StyleTransferTrainer:
    """Trainer for style transfer networks - SAME NAME"""
    
    def __init__(self, 
                 model, 
                 device='cuda',
                 learning_rate=1e-3,
                 content_weight=1.0,
                 style_weight=100.0,  # FIXED: Much more reasonable than 1e6
                 tv_weight=1e-4):     # FIXED: Better TV weight
        
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function - SAME NAME
        self.criterion = CombinedLoss(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        ).to(device)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # ADDED: Loss monitoring to debug style loss = 0
        self.loss_monitor = {
            'content_values': [],
            'style_values': [],
            'tv_values': [],
            'gradient_norms': []
        }
        
        logger.info(f"Trainer initialized with style_weight={style_weight} (much better than 1e6!)")
    
    def debug_tensors(self, pred, content, style, step_name=""):
        """ADDED: Debug tensor properties to find issues"""
        logger.debug(f"{step_name} - Pred: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
        logger.debug(f"{step_name} - Content: {content.shape}, range: [{content.min():.3f}, {content.max():.3f}]")
        logger.debug(f"{step_name} - Style: {style.shape}, range: [{style.min():.3f}, {style.max():.3f}]")
        
        # Check for NaN or inf
        if torch.isnan(pred).any():
            logger.warning(f"{step_name} - NaN detected in prediction!")
        if torch.isinf(pred).any():
            logger.warning(f"{step_name} - Inf detected in prediction!")
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch - SAME NAME, ENHANCED"""
        self.model.train()
        epoch_losses = {'total': 0, 'content': 0, 'style': 0, 'tv': 0}
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (content, style) in enumerate(pbar):
            content = content.to(self.device)
            style = style.to(self.device)
            
            # FIXED: Ensure proper input range [0, 1]
            content = torch.clamp(content, 0, 1)
            style = torch.clamp(style, 0, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(content, style)
            
            # FIXED: Ensure output is in proper range
            output = torch.clamp(output, 0, 1)
            
            # ADDED: Debug on first few batches to catch issues early
            if batch_idx < 3 and epoch == 1:
                self.debug_tensors(output, content, style, f"Epoch {epoch}, Batch {batch_idx}")
            
            # Calculate losses
            losses = self.criterion(output, content, style)
            
            # ADDED: Check for problematic losses
            if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                logger.error(f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                continue
            
            # Backward pass
            losses['total'].backward()
            
            # ADDED: Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            # ADDED: Monitor gradient norms
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.loss_monitor['gradient_norms'].append(total_norm)
            
            self.optimizer.step()
            
            # Update running losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # ADDED: Monitor individual loss components
            self.loss_monitor['content_values'].append(losses['content'].item())
            self.loss_monitor['style_values'].append(losses['style'].item())
            self.loss_monitor['tv_values'].append(losses['tv'].item())
            
            # ENHANCED: Update progress bar with more detailed info
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Content': f"{losses['content'].item():.6f}",
                'Style': f"{losses['style'].item():.6f}",  # Now you'll see if this is 0!
                'TV': f"{losses['tv'].item():.8f}",
                'GradNorm': f"{total_norm:.4f}"
            })
            
            # ADDED: Log warnings for problematic losses
            if losses['style'].item() < 1e-8:
                logger.warning(f"⚠️  Very low style loss: {losses['style'].item():.2e} at epoch {epoch}, batch {batch_idx}")
            
            if losses['content'].item() > 100:
                logger.warning(f"⚠️  Very high content loss: {losses['content'].item():.2f} at epoch {epoch}, batch {batch_idx}")
        
        # Step scheduler with average loss
        avg_loss = epoch_losses['total'] / num_batches
        self.scheduler.step(avg_loss)
        
        # Calculate average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update training state
        self.current_epoch = epoch
        self.training_history.append(epoch_losses)
        
        # ENHANCED: More detailed logging
        logger.info(f"Epoch {epoch} Results:")
        logger.info(f"  Total Loss: {epoch_losses['total']:.6f}")
        logger.info(f"  Content Loss: {epoch_losses['content']:.6f}")
        logger.info(f"  Style Loss: {epoch_losses['style']:.6f}")  # This should NOT be 0!
        logger.info(f"  TV Loss: {epoch_losses['tv']:.8f}")
        logger.info(f"  Average Gradient Norm: {np.mean(self.loss_monitor['gradient_norms'][-num_batches:]):.4f}")
        
        return epoch_losses
    
    def validate(self, dataloader):
        """Validate model on validation set - SAME NAME"""
        self.model.eval()
        val_losses = {'total': 0, 'content': 0, 'style': 0, 'tv': 0}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for content, style in dataloader:
                content = content.to(self.device)
                style = style.to(self.device)
                
                # FIXED: Ensure proper input range
                content = torch.clamp(content, 0, 1)
                style = torch.clamp(style, 0, 1)
                
                # Forward pass
                output = self.model(content, style)
                output = torch.clamp(output, 0, 1)
                
                # Calculate losses
                losses = self.criterion(output, content, style)
                
                # Update running losses
                for key in val_losses:
                    val_losses[key] += losses[key].item()
        
        # Calculate average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        logger.info(f"Validation - Total: {val_losses['total']:.4f}, Style: {val_losses['style']:.6f}")
        
        return val_losses
    
    def train(self, 
              train_dataloader, 
              epochs, 
              val_dataloader=None,
              save_dir='models',
              save_name='model'):
        """Complete training loop - SAME NAME"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Style weight: {self.criterion.style_weight} (should be ~100, not 1e6!)")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_losses = self.train_epoch(train_dataloader, epoch)
            
            # Validate
            if val_dataloader is not None:
                val_losses = self.validate(val_dataloader)
                current_loss = val_losses['total']
            else:
                current_loss = train_losses['total']
            
            # Save best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_model(save_dir / f"{save_name}_best.pth")
                logger.info(f"New best model saved (loss: {current_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(save_dir / f"{save_name}_epoch_{epoch}.pth")
        
        # Save final model
        self.save_model(save_dir / f"{save_name}_final.pth")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def save_model(self, filepath):
        """Save model state - SAME NAME"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }, filepath)
        logger.info(f"Model saved: {filepath}")
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint - SAME NAME"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint - SAME NAME"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded: {filepath}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def generate_sample(self, content_image, style_image, output_path):
        """Generate sample image during training - SAME NAME"""
        self.model.eval()
        
        with torch.no_grad():
            content = content_image.unsqueeze(0).to(self.device)
            style = style_image.unsqueeze(0).to(self.device)
            
            # FIXED: Ensure proper range
            content = torch.clamp(content, 0, 1)
            style = torch.clamp(style, 0, 1)
            
            output = self.model(content, style)
            output = torch.clamp(output, 0, 1)
            
            # Convert to PIL and save
            output_img = output.squeeze(0).cpu()
            
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            result_img = to_pil(output_img)
            result_img.save(output_path)
        
        self.model.train()
        logger.info(f"Sample saved: {output_path}")