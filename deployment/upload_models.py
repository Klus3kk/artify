import os
from deployment.CloudUploader import CloudUploader
from utilities.StyleRegistry import StyleRegistry
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize StyleRegistry
registry = StyleRegistry()

# Google Cloud Storage bucket name
bucket_name = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")

# Upload each model to the cloud
for category in registry.styles.keys():
    model_path = f"models/{category}_model.pth"
    remote_path = f"models/{os.path.basename(model_path)}"

    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} does not exist. Skipping...")
        continue

    logging.info(f"Uploading {model_path} to gs://{bucket_name}/{remote_path}...")
    try:
        CloudUploader.upload_to_cloud(bucket_name, model_path, remote_path)
        logging.info(f"Successfully uploaded {model_path} to gs://{bucket_name}/{remote_path}.")
    except Exception as e:
        logging.error(f"Failed to upload {model_path}: {e}")

logging.info("All models uploaded successfully!")
