"""
Training logic for style transfer networks
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
from pathlib import Path
from tqdm import tqdm

from .losses import CombinedLoss

logger = logging.getLogger(__name__)

class StyleTransferTrainer:
    """Trainer for style transfer networks"""
    
    def __init__(self, 
                 model, 
                 device='cuda',
                 learning_rate=1e-3,
                 content_weight=1.0,
                 style_weight=1e6,
                 tv_weight=1e-6):
        
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        # Loss function
        self.criterion = CombinedLoss(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        ).to(device)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'content': 0, 'style': 0, 'tv': 0}
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (content, style) in enumerate(pbar):
            content = content.to(self.device)
            style = style.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(content, style)
            
            # Calculate losses
            losses = self.criterion(output, content, style)
            
            # Backward pass
            losses['total'].backward()
            self.optimizer.step()
            
            # Update running losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Content': f"{losses['content'].item():.4f}",
                'Style': f"{losses['style'].item():.4f}"
            })
        
        # Step scheduler
        self.scheduler.step()
        
        # Calculate average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update training state
        self.current_epoch = epoch
        self.training_history.append(epoch_losses)
        
        # Log results
        logger.info(f"Epoch {epoch} - Loss: {epoch_losses['total']:.4f}, "
                   f"Content: {epoch_losses['content']:.4f}, "
                   f"Style: {epoch_losses['style']:.4f}, "
                   f"TV: {epoch_losses['tv']:.4f}")
        
        return epoch_losses
    
    def validate(self, dataloader):
        """Validate model on validation set"""
        self.model.eval()
        val_losses = {'total': 0, 'content': 0, 'style': 0, 'tv': 0}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for content, style in dataloader:
                content = content.to(self.device)
                style = style.to(self.device)
                
                # Forward pass
                output = self.model(content, style)
                
                # Calculate losses
                losses = self.criterion(output, content, style)
                
                # Update running losses
                for key in val_losses:
                    val_losses[key] += losses[key].item()
        
        # Calculate average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        logger.info(f"Validation - Loss: {val_losses['total']:.4f}")
        
        return val_losses
    
    def train(self, 
              train_dataloader, 
              epochs, 
              val_dataloader=None,
              save_dir='models',
              save_name='model'):
        """Complete training loop"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }, filepath)
        logger.info(f"Model saved: {filepath}")
    
    def save_checkpoint(self, filepath):
        """Save training checkpoint"""
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
        """Load training checkpoint"""
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
        """Generate sample image during training"""
        self.model.eval()
        
        with torch.no_grad():
            content = content_image.unsqueeze(0).to(self.device)
            style = style_image.unsqueeze(0).to(self.device)
            
            output = self.model(content, style)
            
            # Convert to PIL and save
            output_img = output.squeeze(0).cpu().clamp(0, 1)
            
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            result_img = to_pil(output_img)
            result_img.save(output_path)
        
        self.model.train()
        logger.info(f"Sample saved: {output_path}")