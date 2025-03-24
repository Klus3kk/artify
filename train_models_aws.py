import os
import argparse
import boto3
import logging
import torch
import torchvision.models as models
from PIL import Image
import sys
from pathlib import Path
from huggingface_hub import login, HfApi, create_repo
import time

# Add project root to path to import from Artify modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from utilities.Logger import Logger

# Set up logger
logger = Logger.setup_logger(log_file="aws_training.log", log_level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Train style transfer models on AWS and upload to Hugging Face")
    parser.add_argument("--hf-token", required=True, help="Hugging Face API token")
    parser.add_argument("--aws-profile", default="default", help="AWS profile to use")
    parser.add_argument("--instance-type", default="g4dn.xlarge", help="EC2 instance type for training")
    parser.add_argument("--repo-id", default="YOUR_USERNAME/artify-models", help="Hugging Face repository ID")
    parser.add_argument("--style-categories", nargs="+", default=[], help="Style categories to train (default: all available)")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations (higher = better quality but slower)")
    parser.add_argument("--content-image", default="images/content/sample_content.jpg", help="Content image for training")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region to use")
    parser.add_argument("--local", action="store_true", help="Train locally instead of on AWS")
    parser.add_argument("--s3-bucket", default="", help="S3 bucket for training data")
    parser.add_argument("--ami-id", default="", help="AMI ID for EC2 instance")
    parser.add_argument("--key-name", default="artify-key", help="EC2 key pair name")
    parser.add_argument("--security-group-id", default="", help="Security group ID for EC2 instance")
    return parser.parse_args()

def setup_aws_session(profile_name, region_name):
    """Set up an AWS session with the given profile and region"""
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session

def create_ec2_instance(ec2_client, instance_type, ami_id, key_name, security_group_id, script_content):
    """Create and configure an EC2 instance for training"""
    logger.info(f"Creating EC2 instance of type {instance_type}")
    
    # Base64 encode the user data script
    import base64
    user_data = base64.b64encode(script_content.encode('utf-8')).decode('utf-8')
    
    # Create instance
    response = ec2_client.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        KeyName=key_name,
        SecurityGroupIds=[security_group_id],
        UserData=user_data,
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 100,  # GB
                    'VolumeType': 'gp2',
                    'DeleteOnTermination': True
                }
            }
        ]
    )
    
    instance_id = response['Instances'][0]['InstanceId']
    logger.info(f"Created instance {instance_id}")
    
    # Wait for instance to be running
    logger.info("Waiting for instance to start...")
    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    
    # Get the instance details
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    logger.info(f"Instance is running at {public_ip}")
    
    return instance_id, public_ip

def generate_training_script(style_categories, repo_id, hf_token, iterations, content_image, s3_bucket):
    """Generate the user data script that will run on the EC2 instance"""
    script = f"""#!/bin/bash
set -e

# Update and install dependencies
apt-get update
apt-get install -y python3-pip git
pip3 install torch torchvision pillow huggingface_hub boto3

# Clone the repository
git clone https://github.com/YOUR_USERNAME/artify.git
cd artify

# Create directories
mkdir -p images/content
mkdir -p models

# Download the content image from S3
aws s3 cp s3://{s3_bucket}/{content_image} images/content/

# Set environment variables
export HF_TOKEN="{hf_token}"

# Train models for each style category
"""
    
    for category in style_categories:
        script += f"""
# Train {category} model
python train_models.py --style-category {category} --iterations {iterations} --content-image images/content/{Path(content_image).name}
"""
    
    script += f"""
# Upload models to Hugging Face
huggingface-cli login --token {hf_token}
python -c "
from huggingface_hub import login, create_repo, upload_file
import os
import glob

login(token='{hf_token}')

try:
    create_repo('{repo_id}', private=False)
except:
    print('Repository already exists')

for model_path in glob.glob('models/*_model.pth'):
    print(f'Uploading {{model_path}}')
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id='{repo_id}'
    )
"

# Shutdown the instance when done
shutdown -h now
"""
    return script

def train_locally(args, style_categories):
    """Train models locally"""
    logger.info(f"Starting local training for {len(style_categories)} style categories")
    
    # Set up environment
    os.environ["HF_TOKEN"] = args.hf_token
    
    # Initialize components
    processor = ImageProcessor()
    content_image = processor.preprocess_image(args.content_image)
    registry = StyleRegistry()
    
    for category in style_categories:
        try:
            logger.info(f"Training model for style category: {category}")
            
            # Initialize model with HF token
            model = StyleTransferModel(args.hf_token)
            
            # Get a random style image from the category
            style_image_path = registry.get_random_style_image(category)
            style_image = processor.preprocess_image(style_image_path)
            
            # Define output path
            output_path = os.path.join("models", f"{category}_model.pth")
            
            # Train the model
            model.train_model(
                content_image=content_image,
                style_image=style_image,
                output_path=output_path,
                iterations=args.iterations
            )
            
            logger.info(f"Successfully trained model for {category} at {output_path}")
            
            # Upload the model to Hugging Face
            upload_to_huggingface(args.hf_token, args.repo_id, output_path)
            
        except Exception as e:
            logger.error(f"Error training model for {category}: {e}")

def upload_to_huggingface(token, repo_id, model_path):
    """Upload a model to Hugging Face Hub"""
    try:
        logger.info(f"Uploading {model_path} to Hugging Face repo {repo_id}")
        
        # Login to Hugging Face
        login(token=token)
        
        # Create API object
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id, exist_ok=True)
        except Exception as e:
            logger.warning(f"Repository creation issue (may already exist): {e}")
        
        # Upload file
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id
        )
        
        logger.info(f"Successfully uploaded {model_path} to {repo_id}")
    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        raise

def main():
    args = parse_args()
    
    # Get style categories
    registry = StyleRegistry()
    available_styles = list(registry.styles.keys())
    
    # Use provided style categories or all available styles
    style_categories = args.style_categories if args.style_categories else available_styles
    
    logger.info(f"Will train models for styles: {', '.join(style_categories)}")
    
    if args.local:
        # Train locally
        train_locally(args, style_categories)
    else:
        # Train on AWS
        try:
            session = setup_aws_session(args.aws_profile, args.aws_region)
            ec2_client = session.client('ec2')
            
            # Check required parameters
            if not args.ami_id:
                logger.error("AMI ID is required for AWS training. Use --ami-id parameter.")
                return
                
            if not args.security_group_id:
                logger.error("Security group ID is required for AWS training. Use --security-group-id parameter.")
                return
                
            if not args.s3_bucket:
                logger.error("S3 bucket name is required for AWS training. Use --s3-bucket parameter.")
                return
            
            # Generate the training script
            training_script = generate_training_script(
                style_categories, 
                args.repo_id, 
                args.hf_token, 
                args.iterations, 
                args.content_image,
                args.s3_bucket
            )
            
            # Create EC2 instance and start training
            instance_id, public_ip = create_ec2_instance(
                ec2_client, 
                args.instance_type, 
                args.ami_id, 
                args.key_name, 
                args.security_group_id, 
                training_script
            )
            
            logger.info(f"Training started on instance {instance_id} ({public_ip})")
            logger.info(f"The instance will automatically terminate when training is complete")
            logger.info(f"You can check training progress by SSH into the instance: ssh -i {args.key_name}.pem ubuntu@{public_ip}")
            
        except Exception as e:
            logger.error(f"Error during AWS training setup: {e}")
            raise

if __name__ == "__main__":
    main()