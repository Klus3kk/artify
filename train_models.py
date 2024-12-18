from PIL import Image
from core.StyleTransferModel import StyleTransferModel
from utilities.StyleRegistry import StyleRegistry
from utilities.Logger import Logger
import os
import logging

# Setup logger
logger = Logger.setup_logger(log_file="training.log", log_level=logging.INFO)

# Initialize the style transfer model and registry
model = StyleTransferModel()
registry = StyleRegistry()

# Define the content image for training
content_image_path = "images/content/sample_content.jpg"
if not os.path.exists(content_image_path):
    logger.error(f"Content image not found at {content_image_path}. Exiting.")
    raise FileNotFoundError(f"Content image not found at {content_image_path}.")
content_image = Image.open(content_image_path)

# Ensure models directory exists
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Train a model for each style category
for category in registry.styles.keys():
    try:
        # Get a random style image from the category
        style_image_path = registry.get_random_style_image(category)
        if not os.path.exists(style_image_path):
            logger.warning(f"Style image not found: {style_image_path}. Skipping category {category}.")
            continue
        
        style_image = Image.open(style_image_path)

        # Define the output path for the trained model
        output_path = os.path.join(output_dir, f"{category}_model.pth")

        logger.info(f"Training model for '{category}' using style: {style_image_path}...")
        model.train_model(content_image, style_image, output_path)
        logger.info(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to train model for category '{category}': {e}")
