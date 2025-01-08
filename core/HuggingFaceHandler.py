from huggingface_hub import snapshot_download
import logging
from utilities.Logger import Logger

# Set up logger
logger = Logger.setup_logger(log_file="huggingface.log", log_level=logging.INFO)


class HuggingFaceHandler:
    def __init__(self, token: str):
        """
        Initialize Hugging Face Handler for managing models.

        :param token: Hugging Face API token.
        """
        self.token = token

    def download_model(self, model_name: str, cache_dir="models"):
        """
        Download a model from Hugging Face Hub if not already available.

        :param model_name: Name of the model to download (e.g., 'username/repo_name').
        :param cache_dir: Directory to save the downloaded model.
        :return: Path to the downloaded model.
        """
        try:
            logger.info(f"Checking for model '{model_name}' in cache or downloading it from Hugging Face Hub...")
            model_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir, token=self.token)
            logger.info(f"Model downloaded to {model_path}.")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model '{model_name}': {e}")
            raise
