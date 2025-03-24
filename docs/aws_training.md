# AWS Training for Artify

This document explains how to set up AWS for training Artify models and deploying them to Hugging Face.

## Prerequisites

1. AWS account with billing set up
2. AWS CLI installed and configured
3. Hugging Face account and API token
4. Python 3.8+ with required dependencies

## AWS Setup

### 1. Configure AWS CLI

```bash
# Install AWS CLI if needed
pip install awscli

# Configure your credentials
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region (e.g., eu-central-1), and output format (json).

### 2. Set up Security Group

```bash
# Create a security group for training instances
aws ec2 create-security-group --group-name artify-sg --description "Security group for Artify training"

# Add SSH access to the security group
aws ec2 authorize-security-group-ingress --group-name artify-sg --protocol tcp --port 22 --cidr 0.0.0.0/0

# Get the security group ID
aws ec2 describe-security-groups --group-names artify-sg
```

Note the `GroupId` from the output - you'll need it for the training script.

### 3. Create Key Pair

```bash
# Create a key pair for SSH access
aws ec2 create-key-pair --key-name artify-key --query 'KeyMaterial' --output text > artify-key.pem

# Set correct permissions
chmod 400 artify-key.pem
```

### 4. Create S3 Bucket for Training Data

```bash
# Create a unique bucket (bucket names are globally unique)
aws s3 mb s3://artify-training-$(date +%Y%m%d)

# Upload your images to S3
aws s3 cp images/content/sample_content.jpg s3://YOUR_BUCKET_NAME/
```

Replace `YOUR_BUCKET_NAME` with the bucket name you created.

### 5. Find a suitable AMI

```bash
# Find Deep Learning AMIs with PyTorch in your region
aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*" --query 'Images[*].[ImageId,Name]' --output text --region YOUR_REGION
```

Replace `YOUR_REGION` with your AWS region (e.g., eu-central-1).

## Update Training Script

Edit `train_models_aws.py` and update these values:

1. Default AWS region:
```python
parser.add_argument("--aws-region", default="YOUR_REGION", help="AWS region to use")
```

2. S3 bucket name in `generate_training_script`:
```python
# Download the content image from S3
aws s3 cp s3://YOUR_BUCKET_NAME/{content_image} images/content/
```

3. AMI ID, security group, and key pair in the `main` function:
```python
# Deep Learning AMI for your region
ami_id = "YOUR_AMI_ID"  # Replace with AMI ID from step 5

# Security parameters
key_name = "artify-key"
security_group_id = "YOUR_SECURITY_GROUP_ID"  # Replace with security group ID from step 2
```

## Training Models

### Local Training

```bash
python train_models_aws.py --hf-token YOUR_HF_TOKEN --style-categories impressionism cubism --local
```

### AWS Training

```bash
python train_models_aws.py --hf-token YOUR_HF_TOKEN --style-categories impressionism cubism --instance-type g4dn.xlarge
```

Additional options:
- `--instance-type`: GPU instance type (default: g4dn.xlarge)
- `--iterations`: Number of training iterations (default: 1000)
- `--repo-id`: Hugging Face repository ID (default: ClueSec/artify-models)

## Account Verification

When launching GPU instances for the first time in a region, AWS may require account verification. If you see a message like:

```
Your request for accessing resources in this region is being validated, and you will not be able to launch additional resources in this region until the validation is complete.
```

This is normal and typically completes in a few minutes to a few hours. You can:
1. Use local training in the meantime
2. Try a different region
3. Contact AWS support if the verification doesn't complete after 4 hours

## Monitoring Training

Once your instance is running, you can SSH into it to monitor progress:

```bash
ssh -i artify-key.pem ubuntu@YOUR_INSTANCE_IP
cd artify
tail -f aws_training.log
```

Replace `YOUR_INSTANCE_IP` with the public IP address shown in the script output.

## Cost Management

AWS GPU instances can be expensive. To manage costs:
- Use `g4dn.xlarge` instead of more expensive options like `p3.2xlarge`
- Limit the training iterations with `--iterations`
- Train only the styles you need with `--style-categories`
- The instance will automatically shut down when training is complete

## Troubleshooting

1. **"The requested configuration is currently not supported"**:
   - Try a different instance type or region
   
2. **"The specified bucket does not exist"**:
   - Double-check your S3 bucket name in the script
   
3. **"The image id does not exist"**:
   - Make sure you're using an AMI ID that exists in your region
   
4. **"Cannot access local variable where it is not associated with a value"**:
   - This is a bug in the style transfer code. Update the StyleTransferModel.py file to fix this issue.