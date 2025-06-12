"""
Complete Model Training & Storage System for Artify
Supports AWS training, local training, and multiple storage backends
"""

import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.StyleTransferModel import FastStyleNetwork
from training.trainer import StyleTransferTrainer
from training.dataset import StyleTransferDataset, COCODownloader, get_train_transforms
from utilities.Logger import Logger
import logging

logger = Logger.setup_logger(log_file="training.log", log_level=logging.INFO)

class ModelStorageManager:
    """Manages model storage across multiple backends"""
    
    def __init__(self):
        self.storage_backends = {
            'local': LocalStorage(),
            's3': S3Storage(),
            'huggingface': HuggingFaceStorage()
        }
    
    def save_model(self, model_path: str, backends: List[str], metadata: Dict = None):
        """Save model to multiple storage backends"""
        
        results = {}
        
        for backend_name in backends:
            if backend_name in self.storage_backends:
                try:
                    backend = self.storage_backends[backend_name]
                    result = backend.upload_model(model_path, metadata)
                    results[backend_name] = {'status': 'success', 'location': result}
                    logger.info(f"Model uploaded to {backend_name}: {result}")
                except Exception as e:
                    results[backend_name] = {'status': 'failed', 'error': str(e)}
                    logger.error(f"Failed to upload to {backend_name}: {e}")
            else:
                results[backend_name] = {'status': 'failed', 'error': 'Unknown backend'}
        
        return results
    
    def load_model(self, model_name: str, backend: str = 'local') -> str:
        """Load model from specified backend"""
        
        if backend not in self.storage_backends:
            raise ValueError(f"Unknown backend: {backend}")
        
        storage = self.storage_backends[backend]
        return storage.download_model(model_name)

class LocalStorage:
    """Local file system storage"""
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def upload_model(self, model_path: str, metadata: Dict = None) -> str:
        """Copy model to local storage (already there, just return path)"""
        return str(Path(model_path).resolve())
    
    def download_model(self, model_name: str) -> str:
        """Get local model path"""
        model_path = self.base_dir / model_name
        if model_path.exists():
            return str(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

class S3Storage:
    """AWS S3 storage backend"""
    
    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or os.getenv('ARTIFY_S3_BUCKET', 'artify-models')
        
        # Check if AWS CLI is available
        try:
            import boto3
            self.s3_client = boto3.client('s3')
        except ImportError:
            logger.warning("boto3 not installed - S3 storage disabled")
            self.s3_client = None
    
    def upload_model(self, model_path: str, metadata: Dict = None) -> str:
        """Upload model to S3"""
        
        if not self.s3_client:
            raise RuntimeError("S3 client not available")
        
        model_path = Path(model_path)
        s3_key = f"models/{model_path.name}"
        
        # Upload model file
        self.s3_client.upload_file(str(model_path), self.bucket_name, s3_key)
        
        # Upload metadata if provided
        if metadata:
            metadata_key = f"models/{model_path.stem}_metadata.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2)
            )
        
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def download_model(self, model_name: str) -> str:
        """Download model from S3 to local cache"""
        
        if not self.s3_client:
            raise RuntimeError("S3 client not available")
        
        # Local cache directory
        cache_dir = Path("models/cache")
        cache_dir.mkdir(exist_ok=True)
        
        local_path = cache_dir / model_name
        
        # Download if not cached
        if not local_path.exists():
            s3_key = f"models/{model_name}"
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Downloaded from S3: {model_name}")
        
        return str(local_path)

class HuggingFaceStorage:
    """HuggingFace Hub storage backend"""
    
    def __init__(self, repo_id: str = None):
        self.repo_id = repo_id or os.getenv('ARTIFY_HF_REPO', 'artify-community/style-models')
        self.token = os.getenv('HF_TOKEN')
        
        try:
            from huggingface_hub import upload_file, hf_hub_download
            self.upload_file = upload_file
            self.hf_hub_download = hf_hub_download
        except ImportError:
            logger.warning("huggingface_hub not installed - HF storage disabled")
            self.upload_file = None
            self.hf_hub_download = None
    
    def upload_model(self, model_path: str, metadata: Dict = None) -> str:
        """Upload model to HuggingFace Hub"""
        
        if not self.upload_file:
            raise RuntimeError("HuggingFace Hub not available")
        
        model_path = Path(model_path)
        
        # Upload model file
        url = self.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_path.name,
            repo_id=self.repo_id,
            token=self.token
        )
        
        # Upload metadata
        if metadata:
            metadata_file = model_path.parent / f"{model_path.stem}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.upload_file(
                path_or_fileobj=str(metadata_file),
                path_in_repo=metadata_file.name,
                repo_id=self.repo_id,
                token=self.token
            )
            
            # Cleanup temp metadata file
            metadata_file.unlink()
        
        return f"https://huggingface.co/{self.repo_id}/blob/main/{model_path.name}"
    
    def download_model(self, model_name: str) -> str:
        """Download model from HuggingFace Hub"""
        
        if not self.hf_hub_download:
            raise RuntimeError("HuggingFace Hub not available")
        
        local_path = self.hf_hub_download(
            repo_id=self.repo_id,
            filename=model_name,
            token=self.token,
            cache_dir="models/hf_cache"
        )
        
        return local_path

class ArtifyTrainer:
    """Complete training system for Artify models"""
    
    def __init__(self, 
                 training_environment: str = "local",
                 storage_backends: List[str] = ["local"],
                 model_config: Dict = None):
        
        self.training_environment = training_environment
        self.storage_backends = storage_backends
        self.storage_manager = ModelStorageManager()
        
        # Default model configuration for small models
        self.model_config = model_config or {
            "architecture": "FastStyleNetwork",
            "target_size_mb": 10,  # Target model size
            "optimization": {
                "quantization": True,
                "pruning": False,  # Can enable for even smaller models
                "distillation": False
            },
            "training": {
                "epochs": 50,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "content_weight": 1.0,
                "style_weight": 1e6,
                "tv_weight": 1e-6
            }
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training environment: {training_environment}")
        logger.info(f"Storage backends: {storage_backends}")
        logger.info(f"Device: {self.device}")
    
    def prepare_training_data(self, 
                            content_dir: str = None,
                            style_dir: str = "images/style", 
                            download_coco: bool = True,
                            coco_size: int = 5000):
        """Prepare training data"""
        
        # Download COCO if needed
        if download_coco and not content_dir:
            logger.info("Downloading COCO dataset...")
            downloader = COCODownloader(output_dir='data/coco')
            content_dir = downloader.download_subset(coco_size)
        
        if not content_dir:
            content_dir = "data/coco/subset"
        
        self.content_dir = content_dir
        self.style_dir = style_dir
        
        # Verify directories exist
        if not Path(content_dir).exists():
            raise FileNotFoundError(f"Content directory not found: {content_dir}")
        if not Path(style_dir).exists():
            raise FileNotFoundError(f"Style directory not found: {style_dir}")
        
        logger.info(f"Training data prepared - Content: {content_dir}, Style: {style_dir}")
    
    def create_optimized_model(self) -> torch.nn.Module:
        """Create optimized small model"""
        
        model = FastStyleNetwork()
        
        # Apply optimizations for smaller models
        if self.model_config["optimization"]["quantization"]:
            # Prepare for quantization (will be applied after training)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model = torch.quantization.prepare(model)
        
        return model
    
    def train_style_category(self, style_category: str) -> Dict:
        """Train model for specific style category"""
        
        logger.info(f"Training model for style: {style_category}")
        
        # Create dataset
        dataset = StyleTransferDataset(
            content_dir=self.content_dir,
            style_dir=self.style_dir,
            style_category=style_category,
            content_transform=get_train_transforms(),
            style_transform=get_train_transforms()
        )
        
        if len(dataset) == 0:
            raise ValueError(f"No training data found for style: {style_category}")
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.model_config["training"]["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Dataset size: {len(dataset)} samples")
        
        # Create optimized model
        model = self.create_optimized_model()
        
        # Create trainer
        trainer = StyleTransferTrainer(
            model=model,
            device=self.device,
            learning_rate=self.model_config["training"]["learning_rate"],
            content_weight=self.model_config["training"]["content_weight"],
            style_weight=self.model_config["training"]["style_weight"],
            tv_weight=self.model_config["training"]["tv_weight"]
        )
        
        # Train
        history = trainer.train(
            train_dataloader=dataloader,
            epochs=self.model_config["training"]["epochs"],
            save_dir="models",
            save_name=f"{style_category}_fast_network"
        )
        
        # Get best model path
        best_model_path = f"models/{style_category}_fast_network_best.pth"
        
        # Apply post-training optimizations
        optimized_model_path = self.optimize_model(best_model_path, style_category)
        
        # Calculate model size
        model_size_mb = Path(optimized_model_path).stat().st_size / (1024 * 1024)
        
        # Create metadata
        metadata = {
            "style_category": style_category,
            "architecture": self.model_config["architecture"],
            "model_size_mb": round(model_size_mb, 2),
            "training_config": self.model_config["training"],
            "optimization_config": self.model_config["optimization"],
            "training_samples": len(dataset),
            "final_loss": float(trainer.best_loss),
            "training_time": sum(history["epoch_times"]) if "epoch_times" in history else 0,
            "device": str(self.device),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Upload to storage backends
        storage_results = self.storage_manager.save_model(
            optimized_model_path, 
            self.storage_backends, 
            metadata
        )
        
        return {
            "style_category": style_category,
            "model_path": optimized_model_path,
            "model_size_mb": model_size_mb,
            "metadata": metadata,
            "storage_results": storage_results,
            "training_history": history
        }
    
    def optimize_model(self, model_path: str, style_category: str) -> str:
        """Apply post-training optimizations to reduce model size"""
        
        logger.info(f"Optimizing model: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = FastStyleNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        optimizations_applied = []
        
        # Apply quantization if enabled
        if self.model_config["optimization"]["quantization"]:
            try:
                # Prepare model for quantization
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                model = torch.quantization.prepare(model)
                
                # Calibrate with dummy data
                dummy_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    model(dummy_input, dummy_input)
                
                # Convert to quantized model
                quantized_model = torch.quantization.convert(model)
                
                # Save quantized model
                optimized_path = model_path.replace('.pth', '_optimized.pth')
                torch.save({
                    'model_state_dict': quantized_model.state_dict(),
                    'optimizations': ['quantization'],
                    'original_path': model_path
                }, optimized_path)
                
                optimizations_applied.append('quantization')
                
                # Check size reduction
                original_size = Path(model_path).stat().st_size / (1024 * 1024)
                optimized_size = Path(optimized_path).stat().st_size / (1024 * 1024)
                reduction = (original_size - optimized_size) / original_size * 100
                
                logger.info(f"Quantization applied: {original_size:.2f}MB → {optimized_size:.2f}MB ({reduction:.1f}% reduction)")
                
                return optimized_path
                
            except Exception as e:
                logger.warning(f"Quantization failed: {e}, using original model")
        
        return model_path
    
    def train_all_styles(self, style_categories: List[str] = None) -> Dict:
        """Train models for all style categories"""
        
        if style_categories is None:
            # Auto-detect style categories
            style_dir = Path(self.style_dir)
            style_categories = [d.name for d in style_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Training models for styles: {style_categories}")
        
        results = {}
        total_start_time = time.time()
        
        for style_category in style_categories:
            try:
                result = self.train_style_category(style_category)
                results[style_category] = result
                
                logger.info(f"✓ Completed {style_category}: {result['model_size_mb']:.2f}MB")
                
            except Exception as e:
                logger.error(f"✗ Failed {style_category}: {e}")
                results[style_category] = {"status": "failed", "error": str(e)}
        
        total_time = time.time() - total_start_time
        
        # Summary
        successful = [k for k, v in results.items() if "model_size_mb" in v]
        total_size = sum(v.get("model_size_mb", 0) for v in results.values())
        
        summary = {
            "total_styles": len(style_categories),
            "successful": len(successful),
            "failed": len(style_categories) - len(successful),
            "total_size_mb": round(total_size, 2),
            "average_size_mb": round(total_size / len(successful), 2) if successful else 0,
            "total_training_time": round(total_time / 60, 2),  # minutes
            "storage_backends": self.storage_backends,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Training complete! {summary['successful']}/{summary['total_styles']} models, {summary['total_size_mb']:.2f}MB total")
        
        return {
            "summary": summary,
            "results": results
        }

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train Artify style transfer models")
    
    # Training environment
    parser.add_argument("--environment", choices=["local", "aws"], default="local",
                       help="Training environment")
    
    # Storage backends
    parser.add_argument("--storage", nargs="+", 
                       choices=["local", "s3", "huggingface"],
                       default=["local"],
                       help="Storage backends to use")
    
    # Data arguments
    parser.add_argument("--content-dir", help="Content images directory")
    parser.add_argument("--style-dir", default="images/style", help="Style images directory")
    parser.add_argument("--download-coco", action="store_true", help="Download COCO dataset")
    parser.add_argument("--coco-size", type=int, default=5000, help="COCO subset size")
    
    # Training arguments
    parser.add_argument("--styles", nargs="+", help="Specific styles to train")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    
    # Model optimization
    parser.add_argument("--target-size", type=int, default=10, help="Target model size in MB")
    parser.add_argument("--quantization", action="store_true", help="Enable quantization")
    
    args = parser.parse_args()
    
    # Create model config
    model_config = {
        "architecture": "FastStyleNetwork",
        "target_size_mb": args.target_size,
        "optimization": {
            "quantization": args.quantization,
            "pruning": False,
            "distillation": False
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            "content_weight": 1.0,
            "style_weight": 1e6,
            "tv_weight": 1e-6
        }
    }
    
    # Initialize trainer
    trainer = ArtifyTrainer(
        training_environment=args.environment,
        storage_backends=args.storage,
        model_config=model_config
    )
    
    # Prepare data
    trainer.prepare_training_data(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        download_coco=args.download_coco,
        coco_size=args.coco_size
    )
    
    # Train models
    results = trainer.train_all_styles(args.styles)
    
    # Save results
    results_file = f"training_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 Training Complete!")
    print(f"📊 Results saved to: {results_file}")
    print(f"📦 {results['summary']['successful']} models trained")
    print(f"💾 Total size: {results['summary']['total_size_mb']:.2f}MB")
    print(f"📈 Average size: {results['summary']['average_size_mb']:.2f}MB per model")

if __name__ == "__main__":
    main()