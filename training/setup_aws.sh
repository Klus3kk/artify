#!/bin/bash
set -e

# AWS EC2 setup script for Artify training
# Proper bash script, no Python bullshit

echo "Starting Artify training setup on AWS..."

# Get parameters from environment or defaults
STYLES="${STYLES:-impressionism cubism abstract}"
EPOCHS="${EPOCHS:-50}"
COCO_SIZE="${COCO_SIZE:-5000}"
S3_BUCKET="${S3_BUCKET:-artify-training}"

echo "Training config:"
echo "  Styles: $STYLES"
echo "  Epochs: $EPOCHS"
echo "  COCO size: $COCO_SIZE"
echo "  S3 bucket: $S3_BUCKET"

# Update system
echo "Updating system..."
yum update -y
yum install -y git wget

# Install Python packages
echo "Installing Python packages..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install boto3 pillow tqdm requests

# Check GPU
echo "Checking GPU..."
nvidia-smi || echo "No GPU found"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create project directory
echo "Setting up project..."
cd /home/ec2-user
git clone https://github.com/YOUR_USERNAME/artify.git || {
    echo "Git clone failed. You need to:"
    echo "1. Create a GitHub repo with your Artify code"
    echo "2. Update the git clone URL above"
    exit 1
}

cd artify

# Download style images from S3
echo "Downloading style images from S3..."
aws s3 sync s3://$S3_BUCKET/style_images/ images/style/ --quiet

# Verify style images
echo "Verifying style images..."
for style in $STYLES; do
    count=$(ls images/style/$style/*.jpg images/style/$style/*.png 2>/dev/null | wc -l)
    echo "  $style: $count images"
    if [ $count -eq 0 ]; then
        echo "No images found for style: $style"
        exit 1
    fi
done

# Run training
echo "Starting training..."
python3 training/train_fast_networks.py \
    --download-coco \
    --coco-size $COCO_SIZE \
    --styles $STYLES \
    --epochs $EPOCHS \
    --batch-size 8 \
    --device cuda

# Upload results to S3
echo "Uploading results to S3..."
aws s3 sync models/ s3://$S3_BUCKET/trained_models/ --quiet

echo "Training complete!"
echo "Results uploaded to s3://$S3_BUCKET/trained_models/"

# Show final status
echo "Training summary:"
for style in $STYLES; do
    if [ -f "models/${style}_fast_network_best.pth" ]; then
        size=$(du -h "models/${style}_fast_network_best.pth" | cut -f1)
        echo "Great!$style: $size"
    else
        echo ";C$style: FAILED"
    fi
done

echo "All done! Instance will shutdown in 60 seconds..."
sleep 60
shutdown -h now