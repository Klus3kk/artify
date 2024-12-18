import os
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import logging


class CloudUploader:
    @staticmethod
    def upload_to_cloud(bucket_name, file_path, destination_blob_name):
        """
        Uploads a file to Google Cloud Storage.
        :param bucket_name: Name of the GCS bucket.
        :param file_path: Path to the local file.
        :param destination_blob_name: Name of the blob in GCS.
        """
        if not os.path.exists(file_path):
            logging.error(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")

        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path)
            logging.info(f"Successfully uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}.")
        except DefaultCredentialsError:
            logging.error("Google Cloud credentials not found. Please authenticate.")
            raise
        except Exception as e:
            logging.error(f"Failed to upload {file_path} to gs://{bucket_name}/{destination_blob_name}: {e}")
            raise

    @staticmethod
    def download_from_cloud(bucket_name, source_blob_name, destination_file_name):
        """
        Downloads a blob from Google Cloud Storage to a local file.
        :param bucket_name: Name of the GCS bucket.
        :param source_blob_name: Name of the blob in GCS.
        :param destination_file_name: Path to save the downloaded file.
        """
        try:
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            logging.info(f"Downloaded gs://{bucket_name}/{source_blob_name} to {destination_file_name}.")
        except DefaultCredentialsError:
            logging.error("Google Cloud credentials not found. Please authenticate.")
            raise
        except Exception as e:
            logging.error(f"Failed to download {source_blob_name} from gs://{bucket_name}: {e}")
            raise
