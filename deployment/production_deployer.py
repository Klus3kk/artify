#!/usr/bin/env python3
"""
Production Deployment Orchestrator for Artify
Handles complete deployment to AWS with monitoring
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utilities.Logger import Logger
import logging

logger = Logger.setup_logger(log_file="deployment.log", log_level=logging.INFO)

class ProductionDeployer:
    """Complete production deployment system"""
    
    def __init__(self, environment="production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.deployment_config = self.load_deployment_config()
        
        # AWS configuration
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.aws_profile = os.getenv("AWS_PROFILE", "default")
        
        logger.info(f"Deployer initialized for {environment} environment")
    
    def load_deployment_config(self):
        """Load deployment configuration"""
        
        config_path = self.project_root / "deployment" / f"{self.environment}.json"
        
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        
        # Default configuration
        return {
            "instance_type": "g4dn.xlarge",
            "min_instances": 2,
            "max_instances": 10,
            "target_utilization": 70,
            "health_check_path": "/",
            "ssl_enabled": True,
            "monitoring_enabled": True,
            "backup_enabled": True
        }
    
    def validate_prerequisites(self):
        """Validate all prerequisites are met"""
        
        logger.info("Validating deployment prerequisites...")
        
        checks = []
        
        # Check AWS CLI
        try:
            result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("AWS CLI", True, result.stdout.strip()))
            else:
                checks.append(("AWS CLI", False, "Not installed"))
        except FileNotFoundError:
            checks.append(("AWS CLI", False, "Not found"))
        
        # Check Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("Docker", True, result.stdout.strip()))
            else:
                checks.append(("Docker", False, "Not running"))
        except FileNotFoundError:
            checks.append(("Docker", False, "Not installed"))
        
        # Check Kubernetes
        try:
            result = subprocess.run(["kubectl", "version", "--client"], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("Kubernetes", True, "Available"))
            else:
                checks.append(("Kubernetes", False, "Not configured"))
        except FileNotFoundError:
            checks.append(("Kubernetes", False, "Not installed"))
        
        # Check AWS credentials
        try:
            result = subprocess.run(["aws", "sts", "get-caller-identity"], capture_output=True, text=True)
            if result.returncode == 0:
                identity = json.loads(result.stdout)
                checks.append(("AWS Credentials", True, f"User: {identity.get('Arn', 'Unknown')}"))
            else:
                checks.append(("AWS Credentials", False, "Invalid credentials"))
        except:
            checks.append(("AWS Credentials", False, "Not configured"))
        
        # Check required files
        required_files = [
            "Dockerfile",
            "requirements.txt",
            "api/FastAPIHandler.py",
            "core/StyleTransferModel.py"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                checks.append((f"File: {file_path}", True, "Found"))
            else:
                checks.append((f"File: {file_path}", False, "Missing"))
        
        # Print results
        print("\nPREREQUITE VALIDATION")
        print("=" * 50)
        
        all_passed = True
        for name, passed, details in checks:
            status = "✓" if passed else "✗"
            print(f"{status} {name}: {details}")
            if not passed:
                all_passed = False
        
        if not all_passed:
            logger.error("Prerequisites validation failed")
            return False
        
        logger.info("All prerequisites validated successfully")
        return True
    
    def build_docker_image(self, tag=None):
        """Build Docker image for deployment"""
        
        if tag is None:
            tag = f"artify:{self.environment}-{int(time.time())}"
        
        logger.info(f"Building Docker image: {tag}")
        
        # Build command
        cmd = [
            "docker", "build",
            "-t", tag,
            "-f", "Dockerfile",
            "."
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Docker image built successfully: {tag}")
                return tag
            else:
                logger.error(f"Docker build failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Docker build exception: {e}")
            return None
    
    def push_to_ecr(self, image_tag):
        """Push Docker image to ECR"""
        
        ecr_repo = f"artify-{self.environment}"
        account_id = self.get_aws_account_id()
        
        if not account_id:
            logger.error("Could not get AWS account ID")
            return None
        
        ecr_uri = f"{account_id}.dkr.ecr.{self.aws_region}.amazonaws.com/{ecr_repo}"
        
        logger.info(f"Pushing to ECR: {ecr_uri}")
        
        # Create ECR repository if it doesn't exist
        self.create_ecr_repository(ecr_repo)
        
        # Get ECR login token
        login_cmd = [
            "aws", "ecr", "get-login-password",
            "--region", self.aws_region
        ]
        
        try:
            login_result = subprocess.run(login_cmd, capture_output=True, text=True)
            if login_result.returncode != 0:
                logger.error("ECR login failed")
                return None
            
            # Docker login to ECR
            docker_login_cmd = [
                "docker", "login", "--username", "AWS",
                "--password-stdin", ecr_uri
            ]
            
            subprocess.run(docker_login_cmd, input=login_result.stdout, text=True)
            
            # Tag image for ECR
            full_ecr_tag = f"{ecr_uri}:latest"
            tag_cmd = ["docker", "tag", image_tag, full_ecr_tag]
            subprocess.run(tag_cmd)
            
            # Push to ECR
            push_cmd = ["docker", "push", full_ecr_tag]
            result = subprocess.run(push_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully pushed to ECR: {full_ecr_tag}")
                return full_ecr_tag
            else:
                logger.error(f"ECR push failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"ECR push exception: {e}")
            return None
    
    def create_ecr_repository(self, repo_name):
        """Create ECR repository if it doesn't exist"""
        
        cmd = [
            "aws", "ecr", "create-repository",
            "--repository-name", repo_name,
            "--region", self.aws_region
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"ECR repository created: {repo_name}")
        except:
            # Repository might already exist
            pass
    
    def get_aws_account_id(self):
        """Get AWS account ID"""
        
        try:
            cmd = ["aws", "sts", "get-caller-identity"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                identity = json.loads(result.stdout)
                return identity.get("Account")
        except:
            pass
        
        return None
    
    def deploy_to_kubernetes(self, image_uri):
        """Deploy to Kubernetes cluster"""
        
        logger.info("Deploying to Kubernetes...")
        
        # Update deployment manifests
        self.update_k8s_manifests(image_uri)
        
        # Apply manifests
        k8s_dir = self.project_root / "k8s"
        
        if not k8s_dir.exists():
            logger.error("Kubernetes manifests directory not found")
            return False
        
        # Apply each manifest
        for manifest_file in k8s_dir.glob("*.yaml"):
            cmd = ["kubectl", "apply", "-f", str(manifest_file)]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Applied manifest: {manifest_file.name}")
                else:
                    logger.error(f"Failed to apply {manifest_file.name}: {result.stderr}")
                    return False
            except Exception as e:
                logger.error(f"Kubernetes deployment error: {e}")
                return False
        
        logger.info("Kubernetes deployment completed")
        return True
    
    def update_k8s_manifests(self, image_uri):
        """Update Kubernetes manifests with new image"""
        
        k8s_dir = self.project_root / "k8s"
        
        for manifest_file in k8s_dir.glob("*.yaml"):
            with open(manifest_file, 'r') as f:
                content = f.read()
            
            # Replace image placeholder
            content = content.replace("{{IMAGE_URI}}", image_uri)
            content = content.replace("{{ENVIRONMENT}}", self.environment)
            
            with open(manifest_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Updated manifest: {manifest_file.name}")
    
    def setup_monitoring(self):
        """Setup monitoring and logging"""
        
        logger.info("Setting up monitoring...")
        
        # Deploy Prometheus and Grafana
        monitoring_manifests = [
            "prometheus-deployment.yaml",
            "grafana-deployment.yaml",
            "monitoring-configmap.yaml"
        ]
        
        for manifest in monitoring_manifests:
            manifest_path = self.project_root / "k8s" / "monitoring" / manifest
            
            if manifest_path.exists():
                cmd = ["kubectl", "apply", "-f", str(manifest_path)]
                subprocess.run(cmd)
                logger.info(f"Applied monitoring manifest: {manifest}")
        
        # Setup CloudWatch
        self.setup_cloudwatch_logging()
    
    def setup_cloudwatch_logging(self):
        """Setup CloudWatch logging"""
        
        logger.info("Setting up CloudWatch logging...")
        
        # Create log group
        log_group = f"/aws/artify/{self.environment}"
        
        cmd = [
            "aws", "logs", "create-log-group",
            "--log-group-name", log_group,
            "--region", self.aws_region
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"CloudWatch log group created: {log_group}")
        except:
            # Log group might already exist
            pass
    
    def run_health_checks(self):
        """Run post-deployment health checks"""
        
        logger.info("Running health checks...")
        
        # Get service endpoint
        endpoint = self.get_service_endpoint()
        
        if not endpoint:
            logger.error("Could not get service endpoint")
            return False
        
        # Test health endpoint
        import requests
        
        try:
            response = requests.get(f"{endpoint}/", timeout=30)
            
            if response.status_code == 200:
                logger.info("Health check passed")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Health check exception: {e}")
            return False
    
    def get_service_endpoint(self):
        """Get deployed service endpoint"""
        
        try:
            cmd = [
                "kubectl", "get", "service", "artify-api-service",
                "-o", "jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                hostname = result.stdout.strip("'")
                return f"http://{hostname}"
        except:
            pass
        
        return None
    
    def deploy_full_stack(self):
        """Deploy complete production stack"""
        
        logger.info("Starting full stack deployment...")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            return False
        
        # Build Docker image
        image_tag = self.build_docker_image()
        if not image_tag:
            return False
        
        # Push to ECR
        image_uri = self.push_to_ecr(image_tag)
        if not image_uri:
            return False
        
        # Deploy to Kubernetes
        if not self.deploy_to_kubernetes(image_uri):
            return False
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Wait for deployment to be ready
        logger.info("Waiting for deployment to be ready...")
        time.sleep(60)
        
        # Run health checks
        if not self.run_health_checks():
            logger.warning("Health checks failed - deployment may need time to initialize")
        
        logger.info("Full stack deployment completed!")
        
        # Print deployment summary
        self.print_deployment_summary()
        
        return True
    
    def print_deployment_summary(self):
        """Print deployment summary"""
        
        print(f"Environment: {self.environment}")
        print(f"AWS Region: {self.aws_region}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        endpoint = self.get_service_endpoint()
        if endpoint:
            print(f"API Endpoint: {endpoint}")
            print(f"Health Check: {endpoint}/")
            print(f"API Documentation: {endpoint}/docs")
        
        print("\nMonitoring:")
        print("- CloudWatch Logs: AWS Console")
        print("- Prometheus: kubectl port-forward svc/prometheus 9090:9090")
        print("- Grafana: kubectl port-forward svc/grafana 3000:3000")
        
        print("\nUseful Commands:")
        print("- kubectl get pods")
        print("- kubectl logs -l app=artify-api")
        print("- kubectl describe service artify-api-service")

def main():
    """Main deployment function"""
    
    parser = argparse.ArgumentParser(description="Deploy Artify to production")
    parser.add_argument("--environment", default="production", help="Deployment environment")
    parser.add_argument("--validate-only", action="store_true", help="Only validate prerequisites")
    parser.add_argument("--build-only", action="store_true", help="Only build Docker image")
    
    args = parser.parse_args()
    
    deployer = ProductionDeployer(args.environment)
    
    if args.validate_only:
        deployer.validate_prerequisites()
        return
    
    if args.build_only:
        image_tag = deployer.build_docker_image()
        if image_tag:
            print(f"Docker image built: {image_tag}")
        return
    
    # Full deployment
    success = deployer.deploy_full_stack()
    
    if success:
        print("\nDeployment successful!")
        sys.exit(0)
    else:
        print("\nDeployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()