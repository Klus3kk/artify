"""
AWS EC2 launcher for training
Proper, clean implementation
"""

import boto3
import argparse
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSLauncher:
    """Simple AWS EC2 launcher for training"""
    
    def __init__(self, aws_profile='default', region='us-east-1'):
        self.session = boto3.Session(profile_name=aws_profile, region_name=region)
        self.ec2 = self.session.client('ec2')
        self.s3 = self.session.client('s3')
        self.region = region
        
        # Configuration
        self.instance_type = 'g4dn.xlarge'
        self.ami_id = 'ami-0c02fb55956c7d316'  # Amazon Linux 2 with NVIDIA
        self.key_name = 'artify-training-key'
        self.security_group_name = 'artify-training-sg'
        
        # Generate unique bucket name
        self.s3_bucket = f'artify-training-{int(time.time())}'
    
    def setup_infrastructure(self):
        """Setup AWS infrastructure"""
        logger.info("Setting up AWS infrastructure...")
        
        # Create S3 bucket
        try:
            if self.region == 'us-east-1':
                self.s3.create_bucket(Bucket=self.s3_bucket)
            else:
                self.s3.create_bucket(
                    Bucket=self.s3_bucket,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"Created S3 bucket: {self.s3_bucket}")
        except Exception as e:
            if 'BucketAlreadyExists' in str(e):
                logger.info(f"S3 bucket {self.s3_bucket} already exists")
            else:
                logger.error(f"Failed to create S3 bucket: {e}")
        
        # Create security group if it doesn't exist
        try:
            sg_response = self.ec2.create_security_group(
                GroupName=self.security_group_name,
                Description='Security group for Artify training'
            )
            sg_id = sg_response['GroupId']
            
            # Add SSH rule
            self.ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
            logger.info(f"Created security group: {sg_id}")
        except Exception as e:
            if 'InvalidGroup.Duplicate' in str(e):
                logger.info(f"Security group {self.security_group_name} already exists")
            else:
                logger.warning(f"Security group setup issue: {e}")
    
    def upload_style_images(self, style_dir='images/style'):
        """Upload style images to S3"""
        logger.info("Uploading style images to S3...")
        
        style_path = Path(style_dir)
        if not style_path.exists():
            raise FileNotFoundError(f"Style directory not found: {style_dir}")
        
        uploaded_count = 0
        for category_dir in style_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                for image_file in category_dir.glob("*"):
                    if image_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        s3_key = f"style_images/{category_name}/{image_file.name}"
                        self.s3.upload_file(str(image_file), self.s3_bucket, s3_key)
                        uploaded_count += 1
        
        logger.info(f"Uploaded {uploaded_count} style images to S3")
        return uploaded_count
    
    def generate_user_data_script(self, styles, epochs=50, coco_size=5000):
        """Generate SIMPLE user data script that downloads and runs our bash script"""
        styles_str = ' '.join(styles)
        
        return f"""#!/bin/bash
set -e

# Simple approach - download our setup script and run it
cd /home/ec2-user

# Download the setup script
curl -o setup_aws.sh https://raw.githubusercontent.com/YOUR_USERNAME/artify/main/training/setup_aws.sh
chmod +x setup_aws.sh

# Set environment variables
export STYLES="{styles_str}"
export EPOCHS={epochs}
export COCO_SIZE={coco_size}
export S3_BUCKET={self.s3_bucket}

# Run the setup script
./setup_aws.sh
"""
    
    def launch_training(self, styles, epochs=50, coco_size=5000):
        """Launch EC2 training instance"""
        logger.info(f"Launching training instance for styles: {styles}")
        
        # Generate user data
        user_data = self.generate_user_data_script(styles, epochs, coco_size)
        
        # Launch instance
        try:
            response = self.ec2.run_instances(
                ImageId=self.ami_id,
                InstanceType=self.instance_type,
                MinCount=1,
                MaxCount=1,
                KeyName=self.key_name,
                SecurityGroups=[self.security_group_name],
                UserData=user_data,
                BlockDeviceMappings=[{
                    'DeviceName': '/dev/xvda',
                    'Ebs': {
                        'VolumeSize': 100,
                        'VolumeType': 'gp3',
                        'DeleteOnTermination': True
                    }
                }],
                IamInstanceProfile={
                    'Name': 'EC2-S3-Access'  # You need to create this IAM role
                },
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'Artify-Training'},
                        {'Key': 'Project', 'Value': 'Artify'},
                        {'Key': 'AutoTerminate', 'Value': 'true'}
                    ]
                }]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"Instance launched: {instance_id}")
            
            # Wait for instance to be running
            logger.info("Waiting for instance to start...")
            waiter = self.ec2.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Get public IP
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
            
            return instance_id, public_ip
            
        except Exception as e:
            logger.error(f"Failed to launch instance: {e}")
            raise
    
    def download_results(self, local_dir='models'):
        """Download trained models from S3"""
        logger.info("Downloading trained models...")
        
        Path(local_dir).mkdir(exist_ok=True)
        
        try:
            # List and download all model files
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='trained_models/'
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    filename = Path(key).name
                    if filename:  # Skip directory entries
                        local_path = Path(local_dir) / filename
                        self.s3.download_file(self.s3_bucket, key, str(local_path))
                        logger.info(f"Downloaded: {local_path}")
                        
                logger.info(f"All models downloaded to {local_dir}/")
            else:
                logger.warning("No trained models found in S3")
                
        except Exception as e:
            logger.error(f"Failed to download models: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='AWS launcher for Artify training')
    parser.add_argument('--aws-profile', default='default', help='AWS profile')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--styles', nargs='+', 
                       default=['impressionism', 'cubism', 'abstract'],
                       help='Styles to train')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--coco-size', type=int, default=5000, help='COCO subset size')
    parser.add_argument('--download-only', action='store_true', 
                       help='Only download results')
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = AWSLauncher(args.aws_profile, args.region)
    
    if args.download_only:
        # Just download results
        launcher.download_results()
    else:
        # Full training pipeline
        logger.info("Starting AWS training pipeline...")
        
        # Setup infrastructure
        launcher.setup_infrastructure()
        
        # Upload style images
        launcher.upload_style_images()
        
        # Launch training
        instance_id, public_ip = launcher.launch_training(
            styles=args.styles,
            epochs=args.epochs,
            coco_size=args.coco_size
        )
        
        # Estimate completion time
        estimated_minutes = args.epochs * len(args.styles) * 2  # 2 minutes per epoch per style
        
        print(f"""
Training started on AWS!

Instance ID: {instance_id}
Public IP: {public_ip}
S3 Bucket: {launcher.s3_bucket}

Estimated completion: {estimated_minutes} minutes

Monitor progress:
ssh -i {launcher.key_name}.pem ec2-user@{public_ip}
tail -f /var/log/user-data.log

Download results when complete:
python training/aws_launcher.py --download-only --aws-profile {args.aws_profile}

Estimated cost: ${estimated_minutes * 0.526 / 60:.2f}
""")

if __name__ == "__main__":
    main()