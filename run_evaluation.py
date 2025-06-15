#!/usr/bin/env python3
"""
Quick evaluation runner for Artify
Usage: python run_evaluation.py [--quick] [--output-dir DIR]
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def setup_evaluation_environment():
    """Setup required directories and check dependencies"""
    
    # Create required directories
    directories = [
        "evaluation_results",
        "images/content", 
        "images/style",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check if we have any test images
    content_dir = Path("images/content")
    style_dir = Path("images/style")
    
    content_images = list(content_dir.glob("*.jpg")) + list(content_dir.glob("*.png"))
    style_images = list(style_dir.glob("*.jpg")) + list(style_dir.glob("*.png"))
    
    if not content_images:
        print("WARNING: No content images found in images/content/")
        print("Please add some .jpg or .png images to evaluate")
        return False
    
    if not style_images:
        print("WARNING: No style images found in images/style/")
        print("Please add some .jpg or .png images to evaluate")
        return False
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'PIL',
        'scikit-image',
        'scipy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True

def run_quick_evaluation():
    """Run a quick evaluation with minimal test cases"""
    
    print("Running quick evaluation (5 test pairs)...")
    
    try:
        from evaluation.complete_evaluator import ArtifyEvaluationSuite
        
        # Initialize with minimal test cases
        evaluator = ArtifyEvaluationSuite()
        
        # Create small test dataset
        test_pairs = evaluator.create_test_dataset(max_pairs=5)
        
        if not test_pairs:
            print("ERROR: Could not create test dataset")
            return False
        
        print(f"Created {len(test_pairs)} test pairs")
        
        # Load model and evaluate
        from core.StyleTransferModel import StyleTransferModel
        model = StyleTransferModel()
        
        results = evaluator.evaluate_single_method(model, test_pairs, "Artify-Quick")
        
        # Save results
        file_paths = evaluator.save_results(results, "artify_quick_eval")
        
        # Print summary
        print("\nQUICK EVALUATION COMPLETE!")
        print("=" * 40)
        
        validation = results['industry_validation']
        print(f"Industry Standards: {validation['overall']['standards_passed']}/4 passed")
        
        quality = results['quality']['aggregates']
        print(f"Overall Quality: {quality.get('overall_quality', {}).get('mean', 0):.3f}")
        
        performance = results['performance']
        print(f"Speed: {performance['speed'].get('fps', 0):.1f} FPS")
        print(f"Memory: {performance['memory'].get('peak_memory_mb', 0):.1f} MB")
        
        print(f"\nResults saved to: {file_paths['summary_path']}")
        
        return True
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_evaluation():
    """Run comprehensive evaluation"""
    
    print("Running comprehensive evaluation...")
    
    try:
        from evaluation.complete_evaluator import main as run_full_eval
        run_full_eval()
        return True
        
    except Exception as e:
        print(f"Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_images():
    """Create sample test images if none exist"""
    
    try:
        import torch
        from PIL import Image
        import numpy as np
        
        # Create sample content images
        content_dir = Path("images/content")
        if not list(content_dir.glob("*.jpg")):
            print("Creating sample content images...")
            
            # Generate simple test patterns
            for i in range(3):
                # Create a simple geometric pattern
                img_array = np.random.rand(256, 256, 3) * 255
                
                # Add some structure
                img_array[:, :128] = [100, 150, 200]  # Blue region
                img_array[:128, 128:] = [200, 100, 150]  # Pink region
                img_array[128:, :128] = [150, 200, 100]  # Green region
                
                img = Image.fromarray(img_array.astype(np.uint8))
                img.save(content_dir / f"sample_content_{i+1}.jpg")
        
        # Create sample style images
        style_dir = Path("images/style")
        if not list(style_dir.glob("*.jpg")):
            print("Creating sample style images...")
            
            for i in range(3):
                # Create artistic patterns
                img_array = np.random.rand(256, 256, 3) * 255
                
                # Add artistic structure
                if i == 0:  # Impressionist-style
                    img_array = np.random.normal(150, 50, (256, 256, 3))
                    img_array = np.clip(img_array, 0, 255)
                elif i == 1:  # Abstract style
                    for y in range(256):
                        for x in range(256):
                            img_array[y, x] = [
                                (x + y) % 255,
                                (x * 2) % 255,
                                (y * 2) % 255
                            ]
                else:  # Cubist style
                    img_array = np.random.rand(256, 256, 3) * 255
                    # Add geometric blocks
                    for block_y in range(0, 256, 64):
                        for block_x in range(0, 256, 64):
                            color = np.random.rand(3) * 255
                            img_array[block_y:block_y+64, block_x:block_x+64] = color
                
                img = Image.fromarray(img_array.astype(np.uint8))
                img.save(style_dir / f"sample_style_{i+1}.jpg")
        
        return True
        
    except Exception as e:
        print(f"Failed to create sample images: {e}")
        return False

def main():
    """Main evaluation runner"""
    
    parser = argparse.ArgumentParser(description="Run Artify evaluation")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick evaluation with minimal test cases")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create sample test images if none exist")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check setup, don't run evaluation")
    
    args = parser.parse_args()
    
    print("Artify Evaluation Runner")
    print("=" * 30)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("Please install missing dependencies first")
        return 1
    
    # Setup environment
    print("Setting up evaluation environment...")
    if not setup_evaluation_environment():
        if args.create_samples:
            print("Creating sample test images...")
            if create_sample_images():
                print("Sample images created successfully")
            else:
                print("Failed to create sample images")
                return 1
        else:
            print("Use --create-samples to generate test images")
            return 1
    
    if args.check_only:
        print("Setup check complete - ready for evaluation")
        return 0
    
    # Change output directory if specified
    if args.output_dir != "evaluation_results":
        os.environ["ARTIFY_EVAL_OUTPUT"] = args.output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    if args.quick:
        success = run_quick_evaluation()
    else:
        success = run_full_evaluation()
    
    if success:
        print("\nEvaluation completed successfully!")
        return 0
    else:
        print("\nEvaluation failed!")
        return 1

if __name__ == "__main__":
    exit(main())