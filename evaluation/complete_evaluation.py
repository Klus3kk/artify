import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.StyleTransferModel import StyleTransferModel, FastStyleNetwork
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from utilities.Logger import Logger
import logging

# Import evaluation modules
from notebooks.industry_evaluation_metrics import (
    PerceptualMetrics, TraditionalMetrics, IndustryBenchmark,
    PerformanceAnalysis, ComprehensiveEvaluator
)

logger = Logger.setup_logger(log_file="evaluation.log", log_level=logging.INFO)

class ArtifyEvaluator:
    """Complete evaluation system for Artify models"""
    
    def __init__(self, models_dir="models", results_dir="evaluation_results"):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.processor = ImageProcessor()
        self.registry = StyleRegistry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize evaluators
        self.benchmark = IndustryBenchmark()
        self.comprehensive = ComprehensiveEvaluator()
        
        logger.info(f"Evaluator initialized on {self.device}")
    
    def create_test_dataset(self, content_dir="images/content", style_dir="images/style", max_pairs=50):
        """Create test dataset from available images"""
        
        content_dir = Path(content_dir)
        style_dir = Path(style_dir)
        
        test_pairs = []
        
        # Get content images
        content_images = list(content_dir.glob("*.jpg")) + list(content_dir.glob("*.png"))
        
        # Get style images from registry
        for style_category in self.registry.styles.keys():
            style_images = self.registry.get_style_images(style_category)[:5]  # Max 5 per category
            
            for content_path in content_images[:10]:  # Max 10 content images
                for style_path in style_images:
                    if len(test_pairs) >= max_pairs:
                        break
                    
                    try:
                        # Load and preprocess
                        content = self.processor.preprocess_image(str(content_path))
                        style = self.processor.preprocess_image(str(style_path))
                        
                        # Convert to tensors
                        content_tensor = self._pil_to_tensor(content)
                        style_tensor = self._pil_to_tensor(style)
                        
                        test_pairs.append((content_tensor, style_tensor))
                        
                    except Exception as e:
                        logger.warning(f"Failed to load test pair {content_path}, {style_path}: {e}")
                        continue
                
                if len(test_pairs) >= max_pairs:
                    break
            
            if len(test_pairs) >= max_pairs:
                break
        
        logger.info(f"Created test dataset with {len(test_pairs)} pairs")
        return test_pairs
    
    def evaluate_artify_models(self, test_pairs=None, max_images=20):
        """Evaluate all Artify model variants"""
        
        if test_pairs is None:
            test_pairs = self.create_test_dataset(max_pairs=max_images)
        
        # Initialize models
        models_dict = {
            'Artify_Fast': FastStyleNetwork(),
            'Artify_Quality': StyleTransferModel(),
        }
        
        # Move models to device
        for name, model in models_dict.items():
            if hasattr(model, 'to'):
                models_dict[name] = model.to(self.device)
        
        # Run comprehensive evaluation
        results, comparison = self.comprehensive.compare_methods(models_dict, test_pairs)
        
        return results, comparison
    
    def benchmark_against_industry(self, test_pairs=None):
        """Benchmark against industry standards"""
        
        if test_pairs is None:
            test_pairs = self.create_test_dataset(max_pairs=30)
        
        # Get Artify results
        results, comparison = self.evaluate_artify_models(test_pairs)
        
        # Validate against industry standards
        validation = self.validate_industry_standards(results)
        
        return results, comparison, validation
    
    def validate_industry_standards(self, results):
        """Validate against Adobe, Google, Apple, Meta standards"""
        
        standards = {
            'Adobe_Quality': {
                'overall_quality': 0.75,
                'content_preservation': 0.80,
                'style_quality': 0.70
            },
            'Google_Performance': {
                'fps': 20.0,
                'inference_time_ms': 50.0
            },
            'Apple_Mobile': {
                'model_size_mb': 50.0,
                'memory_mb': 200.0
            },
            'Meta_RealTime': {
                'fps': 30.0,
                'latency_ms': 33.0
            }
        }
        
        validation_results = {}
        
        for method_name, method_results in results.items():
            validation = {}
            
            # Adobe quality standards
            quality = method_results['quality']['aggregates']
            validation['adobe_quality'] = {
                'overall_quality': quality.get('overall_quality', {}).get('mean', 0) >= standards['Adobe_Quality']['overall_quality'],
                'content_preservation': quality.get('content_preservation', {}).get('mean', 0) >= standards['Adobe_Quality']['content_preservation'],
                'style_quality': quality.get('style_quality', {}).get('mean', 0) >= standards['Adobe_Quality']['style_quality']
            }
            validation['adobe_quality']['passes'] = all(validation['adobe_quality'].values())
            
            # Google performance standards
            speed = method_results['speed']
            validation['google_performance'] = {
                'fps': speed.get('fps', 0) >= standards['Google_Performance']['fps'],
                'inference_time': speed.get('mean_time', 1.0) * 1000 <= standards['Google_Performance']['inference_time_ms']
            }
            validation['google_performance']['passes'] = all(validation['google_performance'].values())
            
            # Apple mobile standards
            memory = method_results['memory']
            complexity = method_results['complexity']
            validation['apple_mobile'] = {
                'model_size': complexity.get('model_size_mb', float('inf')) <= standards['Apple_Mobile']['model_size_mb'],
                'memory_usage': memory.get('peak_memory_mb', float('inf')) <= standards['Apple_Mobile']['memory_mb']
            }
            validation['apple_mobile']['passes'] = all(validation['apple_mobile'].values())
            
            # Meta real-time standards
            validation['meta_realtime'] = {
                'fps': speed.get('fps', 0) >= standards['Meta_RealTime']['fps'],
                'latency': speed.get('mean_time', 1.0) * 1000 <= standards['Meta_RealTime']['latency_ms']
            }
            validation['meta_realtime']['passes'] = all(validation['meta_realtime'].values())
            
            validation_results[method_name] = validation
        
        return validation_results
    
    def generate_evaluation_report(self, results, comparison, validation):
        """Generate comprehensive evaluation report"""
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_methods': len(results),
                'test_images': len(results[list(results.keys())[0]]['quality']['individual_results']),
                'industry_compliance': {}
            },
            'detailed_results': results,
            'comparison': comparison,
            'industry_validation': validation,
            'recommendations': self.generate_recommendations(results, validation)
        }
        
        # Calculate industry compliance summary
        for method in results.keys():
            if method in validation:
                compliance = {
                    'adobe': validation[method]['adobe_quality']['passes'],
                    'google': validation[method]['google_performance']['passes'],
                    'apple': validation[method]['apple_mobile']['passes'],
                    'meta': validation[method]['meta_realtime']['passes']
                }
                total_passed = sum(compliance.values())
                report['summary']['industry_compliance'][method] = {
                    'standards_passed': total_passed,
                    'standards_total': 4,
                    'compliance_rate': total_passed / 4,
                    'details': compliance
                }
        
        return report
    
    def generate_recommendations(self, results, validation):
        """Generate actionable recommendations"""
        
        recommendations = []
        
        for method, result in results.items():
            method_recs = []
            
            # Quality recommendations
            quality = result['quality']['aggregates']
            if quality.get('overall_quality', {}).get('mean', 0) < 0.75:
                method_recs.append("Improve overall quality - consider better loss functions or training data")
            
            if quality.get('content_preservation', {}).get('mean', 0) < 0.80:
                method_recs.append("Enhance content preservation - increase content loss weight")
            
            # Performance recommendations
            speed = result['speed']
            if speed.get('fps', 0) < 20:
                method_recs.append("Optimize inference speed - consider model pruning or quantization")
            
            # Memory recommendations
            memory = result['memory']
            if memory.get('peak_memory_mb', 0) > 200:
                method_recs.append("Reduce memory usage - optimize batch processing or model architecture")
            
            recommendations.append({
                'method': method,
                'recommendations': method_recs
            })
        
        return recommendations
    
    def save_results(self, report, filename="evaluation_report.json"):
        """Save evaluation results"""
        
        output_path = self.results_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Also save summary
        summary_path = self.results_dir / "evaluation_summary.txt"
        self.save_summary(report, summary_path)
        
        return output_path
    
    def save_summary(self, report, summary_path):
        """Save human-readable summary"""
        
        with open(summary_path, 'w') as f:
            f.write("ARTIFY EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Evaluation Date: {report['timestamp']}\n")
            f.write(f"Methods Tested: {report['summary']['total_methods']}\n")
            f.write(f"Test Images: {report['summary']['test_images']}\n\n")
            
            f.write("INDUSTRY COMPLIANCE\n")
            f.write("-" * 30 + "\n")
            
            for method, compliance in report['summary']['industry_compliance'].items():
                f.write(f"\n{method}:\n")
                f.write(f"  Standards Passed: {compliance['standards_passed']}/4\n")
                f.write(f"  Compliance Rate: {compliance['compliance_rate']:.1%}\n")
                
                details = compliance['details']
                f.write(f"  Adobe Quality: {'✓' if details['adobe'] else '✗'}\n")
                f.write(f"  Google Performance: {'✓' if details['google'] else '✗'}\n")
                f.write(f"  Apple Mobile: {'✓' if details['apple'] else '✗'}\n")
                f.write(f"  Meta Real-time: {'✓' if details['meta'] else '✗'}\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            for rec in report['recommendations']:
                f.write(f"\n{rec['method']}:\n")
                for recommendation in rec['recommendations']:
                    f.write(f"  • {recommendation}\n")
        
        logger.info(f"Summary saved to {summary_path}")
    
    def _pil_to_tensor(self, image):
        """Convert PIL image to tensor"""
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        return transform(image)

def main():
    """Run complete evaluation pipeline"""
    
    evaluator = ArtifyEvaluator()
    
    logger.info("Starting comprehensive evaluation...")
    
    # Run evaluation
    results, comparison, validation = evaluator.benchmark_against_industry()
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, comparison, validation)
    
    # Save results
    output_path = evaluator.save_results(report)
    
    logger.info("Evaluation complete!")
    
    for method, compliance in report['summary']['industry_compliance'].items():
        print(f"{method}: {compliance['standards_passed']}/4 standards passed ({compliance['compliance_rate']:.1%})")
    
    print(f"\nFull report saved to: {output_path}")

if __name__ == "__main__":
    main()