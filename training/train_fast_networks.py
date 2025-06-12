"""
Main training script for fast style transfer networks
Clean implementation that can run locally or on AWS
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset import StyleTransferDataset, COCODownloader, get_train_transforms
from training.trainer import StyleTransferTrainer
from core.StyleTransferModel import FastStyleNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train fast style transfer networks')
    
    # Data arguments
    parser.add_argument('--content-dir', default='data/coco/subset',
                       help='Directory containing content images')
    parser.add_argument('--style-dir', default='images/style',
                       help='Directory containing style images')
    parser.add_argument('--download-coco', action='store_true',
                       help='Download COCO dataset before training')
    parser.add_argument('--coco-size', type=int, default=5000,
                       help='Size of COCO subset to download')
    
    # Training arguments
    parser.add_argument('--styles', nargs='+', 
                       default=['abstract', 'baroque', 'cubism', 'expressionism', 'impressionism', 'post-impressionism'],
                       help='Style categories to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loader workers')
    
    # Loss weights
    parser.add_argument('--content-weight', type=float, default=1.0,
                       help='Content loss weight')
    parser.add_argument('--style-weight', type=float, default=1e6,
                       help='Style loss weight')
    parser.add_argument('--tv-weight', type=float, default=1e-6,
                       help='Total variation loss weight')
    
    # Output arguments
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save trained models')
    parser.add_argument('--save-samples', action='store_true',
                       help='Save sample images during training')
    parser.add_argument('--sample-dir', default='samples',
                       help='Directory to save sample images')
    
    # Device arguments
    parser.add_argument('--device', default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup compute device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    return device

def prepare_data(args):
    """Prepare training data"""
    
    # Download COCO if requested
    if args.download_coco:
        logger.info("Downloading COCO dataset...")
        downloader = COCODownloader(output_dir='data/coco')
        content_dir = downloader.download_subset(args.coco_size)
        args.content_dir = str(content_dir)
    
    # Verify data directories exist
    content_path = Path(args.content_dir)
    style_path = Path(args.style_dir)
    
    if not content_path.exists():
        raise FileNotFoundError(f"Content directory not found: {content_path}")
    if not style_path.exists():
        raise FileNotFoundError(f"Style directory not found: {style_path}")
    
    logger.info(f"Content images: {args.content_dir}")
    logger.info(f"Style images: {args.style_dir}")
    
    return args.content_dir, args.style_dir

def train_style_network(style_category, args, device):
    """Train network for a specific style category"""
    
    logger.info(f"Starting training for style: {style_category}")
    
    # Create dataset
    dataset = StyleTransferDataset(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        style_category=style_category,
        content_transform=get_train_transforms(),
        style_transform=get_train_transforms()
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    model = FastStyleNetwork()
    
    # Create trainer
    trainer = StyleTransferTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Train
    history = trainer.train(
        train_dataloader=dataloader,
        epochs=args.epochs,
        save_dir=output_dir,
        save_name=f"{style_category}_fast_network"
    )
    
    logger.info(f"Training completed for {style_category}")
    
    return history

def main():
    args = parse_args()
    
    logger.info("Starting fast style transfer training")
    logger.info(f"Training styles: {args.styles}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Setup device
    device = setup_device(args.device)
    
    # Prepare data
    content_dir, style_dir = prepare_data(args)
    
    # Create output directories
    Path(args.output_dir).mkdir(exist_ok=True)
    if args.save_samples:
        Path(args.sample_dir).mkdir(exist_ok=True)
    
    # Train networks for each style
    training_results = {}
    
    for style_category in args.styles:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING STYLE: {style_category.upper()}")
            logger.info(f"{'='*60}")
            
            history = train_style_network(style_category, args, device)
            training_results[style_category] = history
            
            logger.info(f"{style_category} training complete")
            
        except Exception as e:
            logger.error(f"Failed to train {style_category}: {e}")
            training_results[style_category] = None
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = []
    failed = []
    
    for style, result in training_results.items():
        if result is not None:
            successful.append(style)
            final_loss = result[-1]['total'] if result else "Unknown"
            logger.info(f"{style}: Final loss = {final_loss:.4f}")
        else:
            failed.append(style)
            logger.info(f"{style}: Training failed")
    
    logger.info(f"\nSuccessful: {len(successful)}/{len(args.styles)}")
    logger.info(f"Models saved in: {args.output_dir}")
    
    if successful:
        logger.info("\nTraining completed! Your fast networks are ready!")
    
    if failed:
        logger.warning(f"\nSome styles failed: {failed}")

if __name__ == "__main__":
    main()