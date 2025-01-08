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

    def download_model(self, repo_id: str, cache_dir="models"):
        """
        Download a Hugging Face repository.

        :param repo_id: Name of the Hugging Face repository (e.g., 'ClueSec/artify-models').
        :param cache_dir: Directory to cache downloaded models.
        :return: Path to the downloaded repository.
        """
        try:
            logger.info(f"Downloading repository '{repo_id}' from Hugging Face Hub...")
            repo_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, token=self.token)
            logger.info(f"Repository downloaded to {repo_path}.")
            return repo_path
        except Exception as e:
            logger.error(f"Failed to download repository '{repo_id}': {e}")
            raise
