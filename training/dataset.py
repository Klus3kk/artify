"""
Dataset classes for style transfer training
Clean, professional implementation
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class StyleTransferDataset(Dataset):
    """Dataset for training style transfer networks"""
    
    def __init__(self, 
                 content_dir: str, 
                 style_dir: str, 
                 style_category: str,
                 content_transform: Optional[Callable] = None,
                 style_transform: Optional[Callable] = None):
        
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir) / style_category
        
        # Set up transforms
        self.content_transform = content_transform or self._default_transform()
        self.style_transform = style_transform or self._default_transform()
        
        # Load image paths
        self.content_images = self._load_images(self.content_dir)
        self.style_images = self._load_images(self.style_dir)
        
        if not self.content_images:
            raise ValueError(f"No content images found in {self.content_dir}")
        if not self.style_images:
            raise ValueError(f"No style images found in {self.style_dir}")
        
        logger.info(f"Dataset loaded: {len(self.content_images)} content, {len(self.style_images)} style images")
    
    def _load_images(self, directory: Path):
        """Load all image paths from directory"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = []
        
        if directory.exists():
            for ext in extensions:
                images.extend(directory.glob(f"*{ext}"))
                images.extend(directory.glob(f"*{ext.upper()}"))
        
        return images
    
    def _default_transform(self):
        """Default image transforms"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.content_images)
    
    def __getitem__(self, idx):
        # Load content image
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert('RGB')
        
        # Load random style image
        style_idx = random.randint(0, len(self.style_images) - 1)
        style_path = self.style_images[style_idx]
        style_img = Image.open(style_path).convert('RGB')
        
        # Apply transforms
        content_tensor = self.content_transform(content_img)
        style_tensor = self.style_transform(style_img)
        
        return content_tensor, style_tensor

class COCODownloader:
    """Downloads COCO dataset subset"""
    
    def __init__(self, output_dir: str = "data/coco"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_subset(self, subset_size: int = 5000):
        """Download COCO validation set and create subset"""
        import requests
        import zipfile
        import shutil
        from tqdm import tqdm
        
        logger.info(f"Downloading COCO subset ({subset_size} images)...")
        
        # Download validation set (smaller than train)
        val_url = "http://images.cocodataset.org/zips/val2017.zip"
        val_zip = self.output_dir / "val2017.zip"
        
        if not val_zip.exists():
            logger.info("Downloading COCO val2017...")
            response = requests.get(val_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(val_zip, 'wb') as f, tqdm(
                desc="val2017.zip",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        
        # Extract
        val_dir = self.output_dir / "val2017"
        if not val_dir.exists():
            logger.info("Extracting COCO images...")
            with zipfile.ZipFile(val_zip, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
        
        # Create subset
        subset_dir = self.output_dir / "subset"
        if not subset_dir.exists() or len(list(subset_dir.glob("*.jpg"))) < subset_size:
            subset_dir.mkdir(exist_ok=True)
            
            all_images = list(val_dir.glob("*.jpg"))
            
            if len(all_images) > subset_size:
                selected_images = random.sample(all_images, subset_size)
            else:
                selected_images = all_images
            
            logger.info(f"Creating subset with {len(selected_images)} images...")
            for img in tqdm(selected_images, desc="Copying images"):
                shutil.copy2(img, subset_dir / img.name)
        
        # Cleanup
        if val_zip.exists():
            val_zip.unlink()
        if val_dir.exists():
            shutil.rmtree(val_dir)
        
        logger.info(f"COCO subset ready: {len(list(subset_dir.glob('*.jpg')))} images")
        return subset_dir

def get_train_transforms():
    """Get training transforms with augmentation"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor()
    ])

def get_val_transforms():
    """Get validation transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])