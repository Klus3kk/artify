from google.cloud import storage 

class CloudUploader:
    @staticmethod
    def upload_to_cloud(bucket_name, file_path, destination_blob_name):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        print(f"File {file_path} uploaded to {destination_blob_name}.")

    @staticmethod
    def download_from_cloud(bucket_name, source_blob_name, destination_file_name):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
