from PIL import Image
from core.StyleTransferModel import StyleTransferModel
from utilities.StyleRegistry import StyleRegistry
import os

# Initialize the style transfer model and registry
model = StyleTransferModel()
registry = StyleRegistry()

# Define the content image for training
content_image = Image.open("images/content/sample_content.jpg")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Train a model for each style category
for category in registry.styles.keys():
    # Get a random style image from the category
    style_image_path = registry.get_random_style_image(category)
    style_image = Image.open(style_image_path)

    # Define the output path for the trained model
    output_path = f"models/{category}_model.pth"

    print(f"Training model for {category} using style: {style_image_path}...")
    model.train_model(content_image, style_image, output_path)
    print(f"Model saved to {output_path}")
