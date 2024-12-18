from deployment.CloudUploader import CloudUploader
from utilities.StyleRegistry import StyleRegistry
import os

# Initialize StyleRegistry
registry = StyleRegistry()

# Google Cloud Storage bucket name
bucket_name = "your-bucket-name"

# Upload each model to the cloud
for category in registry.styles.keys():
    model_path = f"models/{category}_model.pth"
    remote_path = f"models/{os.path.basename(model_path)}"
    print(f"Uploading {model_path} to {bucket_name}/{remote_path}...")
    CloudUploader.upload_to_cloud(bucket_name, model_path, remote_path)

print("All models uploaded successfully!")
