"""
Industry-Grade Data Download & Curation System for Artify
Based on Adobe NeAT, Google Meta Networks, and standard benchmarks

This module implements:
1. COCO dataset download and preprocessing 
2. Style image curation from public APIs
3. Adobe-style quality classification
4. Industry-standard evaluation benchmarks
"""

import os
import requests
import zipfile
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import cv2
from sklearn.metrics import classification_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Industry-grade data download and curation system"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Industry-standard directory structure
        self.coco_dir = self.base_dir / "coco"
        self.style_dir = self.base_dir / "style"
        self.eval_dir = self.base_dir / "evaluation"
        
        for directory in [self.coco_dir, self.style_dir, self.eval_dir]:
            directory.mkdir(exist_ok=True)
            
        # Style categories based on research
        self.style_categories = {
            'abstract': {
                'description': 'Abstract art with non-representational forms',
                'target_count': 30,
                'keywords': ['abstract', 'kandinsky', 'mondrian', 'non-representational']
            },
            'baroque': {
                'description': 'Baroque art with dramatic lighting and rich detail',
                'target_count': 30,
                'keywords': ['baroque', 'caravaggio', 'rubens', 'dramatic']
            },
            'cubism': {
                'description': 'Cubist artworks with geometric fragmentation',
                'target_count': 30,
                'keywords': ['cubism', 'picasso', 'braque', 'geometric']
            },
            'expressionism': {
                'description': 'Expressionist art with emotional intensity',
                'target_count': 30,
                'keywords': ['expressionism', 'van gogh', 'munch', 'emotional']
            },
            'impressionism': {
                'description': 'Impressionist paintings with light and color emphasis',
                'target_count': 30,
                'keywords': ['impressionism', 'monet', 'renoir', 'degas']
            },
            'post-impressionism': {
                'description': 'Post-impressionist works with symbolic content',
                'target_count': 30,
                'keywords': ['post-impressionism', 'cezanne', 'gauguin', 'seurat']
            }
        }
        
        # Initialize style classifier (Adobe approach)
        self.style_classifier = self._load_style_classifier()

    def _load_style_classifier(self) -> nn.Module:
        """Load or create Adobe-style quality classifier"""
        try:
            # Try to load pre-trained classifier
            classifier_path = self.base_dir / "models" / "style_classifier.pth"
            if classifier_path.exists():
                classifier = torch.load(classifier_path)
                logger.info("Loaded pre-trained style classifier")
                return classifier
        except Exception as e:
            logger.warning(f"Could not load pre-trained classifier: {e}")
        
        # Create new classifier based on ResNet18
        classifier = StyleClassifier()
        logger.info("Created new style classifier")
        return classifier

    def download_coco_subset(self, subset_size: int = 10000) -> None:
        """
        Download COCO dataset subset for content images
        Based on industry standard: diverse, high-quality content
        """
        logger.info(f"Downloading COCO subset ({subset_size} images)...")
        
        # COCO 2017 URLs
        urls = {
            'train': 'http://images.cocodataset.org/zips/train2017.zip',
            'val': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        
        # Download and extract
        for split, url in urls.items():
            self._download_and_extract(url, self.coco_dir)
        
        # Create balanced subset
        self._create_balanced_coco_subset(subset_size)
        
    def _download_and_extract(self, url: str, extract_dir: Path) -> None:
        """Download and extract zip file with progress bar"""
        filename = url.split('/')[-1]
        filepath = extract_dir / filename
        
        if filepath.exists():
            logger.info(f"{filename} already exists, skipping download")
            return
            
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        # Extract
        logger.info(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Clean up zip file
        filepath.unlink()

    def _create_balanced_coco_subset(self, subset_size: int) -> None:
        """Create balanced subset with diverse content categories"""
        annotations_file = self.coco_dir / "annotations" / "instances_train2017.json"
        
        if not annotations_file.exists():
            logger.error("COCO annotations not found")
            return
            
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Categorize images for balanced sampling
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        image_categories = {}
        
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            category = categories[annotation['category_id']]
            
            if image_id not in image_categories:
                image_categories[image_id] = set()
            image_categories[image_id].add(category)
        
        # Sample balanced subset
        target_per_category = subset_size // len(categories)
        selected_images = set()
        
        for category in categories.values():
            count = 0
            for image_id, cats in image_categories.items():
                if category in cats and count < target_per_category:
                    selected_images.add(image_id)
                    count += 1
        
        # Copy selected images
        train_dir = self.coco_dir / "train2017"
        subset_dir = self.coco_dir / "subset"
        subset_dir.mkdir(exist_ok=True)
        
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        logger.info(f"Creating COCO subset with {len(selected_images)} images...")
        for image_id in tqdm(selected_images):
            if image_id in id_to_filename:
                src = train_dir / id_to_filename[image_id]
                dst = subset_dir / id_to_filename[image_id]
                if src.exists():
                    shutil.copy2(src, dst)

    def curate_style_images(self) -> None:
        """
        Curate high-quality style images from public APIs
        Based on Adobe BBST-4M approach and Google WikiArt standards
        """
        logger.info("Starting style image curation...")
        
        # Public art APIs
        apis = [
            self._fetch_met_museum_art,
            self._fetch_rijksmuseum_art,
            self._fetch_wikiart_public_domain
        ]
        
        for category, config in self.style_categories.items():
            logger.info(f"Curating {category} images...")
            category_dir = self.style_dir / category
            category_dir.mkdir(exist_ok=True)
            
            collected_images = []
            
            # Try each API
            for api_func in apis:
                try:
                    images = api_func(config['keywords'], config['target_count'] // len(apis))
                    collected_images.extend(images)
                except Exception as e:
                    logger.warning(f"API error for {category}: {e}")
            
            # Filter and save high-quality images
            self._filter_and_save_style_images(collected_images, category_dir, config['target_count'])

    def _fetch_met_museum_art(self, keywords: List[str], count: int) -> List[Dict]:
        """Fetch art from Metropolitan Museum API"""
        base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
        images = []
        
        for keyword in keywords:
            try:
                # Search for objects
                search_url = f"{base_url}/search?hasImages=true&q={keyword}"
                response = requests.get(search_url)
                data = response.json()
                
                if 'objectIDs' in data:
                    object_ids = data['objectIDs'][:count // len(keywords)]
                    
                    for obj_id in object_ids:
                        # Get object details
                        obj_url = f"{base_url}/objects/{obj_id}"
                        obj_response = requests.get(obj_url)
                        obj_data = obj_response.json()
                        
                        if obj_data.get('primaryImage'):
                            images.append({
                                'url': obj_data['primaryImage'],
                                'title': obj_data.get('title', ''),
                                'artist': obj_data.get('artistDisplayName', ''),
                                'source': 'met_museum'
                            })
                            
                        if len(images) >= count:
                            break
                            
            except Exception as e:
                logger.warning(f"Met Museum API error for {keyword}: {e}")
                continue
                
        return images

    def _fetch_rijksmuseum_art(self, keywords: List[str], count: int) -> List[Dict]:
        """Fetch art from Rijksmuseum API"""
        api_key = os.getenv('RIJKSMUSEUM_API_KEY', '0fiuZFh4')  # Public key
        base_url = "https://www.rijksmuseum.nl/api/en/collection"
        images = []
        
        for keyword in keywords:
            try:
                params = {
                    'key': api_key,
                    'q': keyword,
                    'imgonly': True,
                    'ps': count // len(keywords)
                }
                
                response = requests.get(base_url, params=params)
                data = response.json()
                
                for item in data.get('artObjects', []):
                    if item.get('webImage'):
                        images.append({
                            'url': item['webImage']['url'],
                            'title': item.get('title', ''),
                            'artist': item.get('principalOrFirstMaker', ''),
                            'source': 'rijksmuseum'
                        })
                        
            except Exception as e:
                logger.warning(f"Rijksmuseum API error for {keyword}: {e}")
                continue
                
        return images

    def _fetch_wikiart_public_domain(self, keywords: List[str], count: int) -> List[Dict]:
        """
        Fetch public domain art from WikiArt
        Note: This is a simplified implementation. 
        In production, use proper WikiArt API or dataset
        """
        # Placeholder for WikiArt integration
        # In real implementation, this would use:
        # 1. WikiArt API (if available)
        # 2. Public domain dataset from Papers with Code
        # 3. Kaggle WikiArt dataset
        
        logger.info("WikiArt integration placeholder - implement proper API access")
        return []

    def _filter_and_save_style_images(self, images: List[Dict], output_dir: Path, target_count: int) -> None:
        """Filter images using Adobe-style quality classifier and save"""
        
        def download_and_process_image(image_info: Dict) -> Optional[Tuple[str, float]]:
            try:
                # Download image
                response = requests.get(image_info['url'], timeout=30)
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                
                # Check minimum quality requirements
                if img.size[0] < 256 or img.size[1] < 256:
                    return None
                
                # Apply style classifier (Adobe approach)
                quality_score = self._evaluate_style_quality(img)
                
                if quality_score > 0.7:  # Quality threshold
                    # Generate unique filename
                    hash_str = hashlib.md5(image_info['url'].encode()).hexdigest()[:8]
                    filename = f"{hash_str}_{image_info['source']}.jpg"
                    filepath = output_dir / filename
                    
                    # Save with metadata
                    img.save(filepath, 'JPEG', quality=95)
                    
                    # Save metadata
                    metadata = {
                        'url': image_info['url'],
                        'title': image_info['title'],
                        'artist': image_info['artist'],
                        'source': image_info['source'],
                        'quality_score': quality_score
                    }
                    
                    metadata_file = filepath.with_suffix('.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    return filename, quality_score
                
            except Exception as e:
                logger.warning(f"Error processing image {image_info['url']}: {e}")
                return None
        
        # Parallel processing
        logger.info(f"Processing {len(images)} images...")
        successful_downloads = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_image = {executor.submit(download_and_process_image, img): img for img in images}
            
            for future in tqdm(as_completed(future_to_image), total=len(images)):
                result = future.result()
                if result:
                    successful_downloads.append(result)
                    
                if len(successful_downloads) >= target_count:
                    break
        
        logger.info(f"Successfully downloaded {len(successful_downloads)} style images")

    def _evaluate_style_quality(self, image: Image.Image) -> float:
        """
        Evaluate image style quality using Adobe-inspired approach
        Returns quality score between 0 and 1
        """
        try:
            # Convert to tensor
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0)
            
            # Run through classifier
            with torch.no_grad():
                score = self.style_classifier(img_tensor)
                return float(torch.sigmoid(score).item())
                
        except Exception as e:
            logger.warning(f"Style quality evaluation error: {e}")
            return 0.5  # Default neutral score

    def create_evaluation_benchmarks(self) -> None:
        """Create industry-standard evaluation benchmarks"""
        
        logger.info("Creating evaluation benchmarks...")
        
        # Create benchmark directories
        benchmark_dir = self.eval_dir / "benchmarks"
        benchmark_dir.mkdir(exist_ok=True)
        
        # Standard test images for reproducible evaluation
        standard_content = [
            "portrait", "landscape", "architecture", "animal", "object", "scene"
        ]
        
        # Create test combinations
        test_combinations = []
        
        for content_type in standard_content:
            for style_category in self.style_categories.keys():
                test_combinations.append({
                    'content_type': content_type,
                    'style_category': style_category,
                    'expected_metrics': {
                        'min_lpips': 0.3,  # Minimum perceptual similarity
                        'min_ssim': 0.6,   # Minimum structural similarity
                        'max_inference_time': 2.0  # Maximum seconds
                    }
                })
        
        # Save benchmark configuration
        benchmark_config = {
            'version': '1.0',
            'description': 'Industry-standard style transfer evaluation benchmarks',
            'test_combinations': test_combinations,
            'evaluation_metrics': [
                'LPIPS', 'SSIM', 'FID', 'Style Deception Rate', 
                'Content Preservation', 'Inference Time'
            ]
        }
        
        with open(benchmark_dir / "config.json", 'w') as f:
            json.dump(benchmark_config, f, indent=2)
        
        logger.info(f"Created {len(test_combinations)} benchmark test cases")

    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            'coco_subset': self._count_images(self.coco_dir / "subset"),
            'style_categories': {},
            'total_style_images': 0
        }
        
        for category in self.style_categories.keys():
            category_dir = self.style_dir / category
            count = self._count_images(category_dir)
            stats['style_categories'][category] = count
            stats['total_style_images'] += count
        
        return stats

    def _count_images(self, directory: Path) -> int:
        """Count images in directory"""
        if not directory.exists():
            return 0
        return len([f for f in directory.glob("*.jpg") if f.is_file()])


class StyleClassifier(nn.Module):
    """
    Adobe-inspired style quality classifier
    Determines if an image has artistic/stylistic qualities
    """
    
    def __init__(self):
        super(StyleClassifier, self).__init__()
        
        # Use pre-trained ResNet18 as backbone
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Binary: stylistic vs non-stylistic
        )
        
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


# Example usage and testing
if __name__ == "__main__":
    import io
    
    # Initialize downloader
    downloader = DataDownloader()
    
    # Download COCO subset (small test)
    downloader.download_coco_subset(subset_size=1000)
    
    # Curate style images
    downloader.curate_style_images()
    
    # Create evaluation benchmarks
    downloader.create_evaluation_benchmarks()
    
    # Print statistics
    stats = downloader.get_dataset_statistics()
    print("\nDataset Statistics:")
    print(f"COCO subset: {stats['coco_subset']} images")
    print(f"Total style images: {stats['total_style_images']}")
    for category, count in stats['style_categories'].items():
        print(f"  {category}: {count} images")