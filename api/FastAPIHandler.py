from fastapi import FastAPI, File, UploadFile
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from PIL import Image
from io import BytesIO
from pathlib import Path
from werkzeug.utils import secure_filename
from utilities.Logger import Logger
import logging
import os

# Set up logger
logger = Logger.setup_logger(log_file="fastapi.log", log_level=logging.INFO)

# Initialize FastAPI and shared components
app = FastAPI()

# Retrieve Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN", "mock_token")  # Default to "mock_token" for testing purposes

# Initialize shared components
processor = ImageProcessor()
model = StyleTransferModel(hf_token)
registry = StyleRegistry()

# Output directory
OUTPUT_DIR = Path("images/output")
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create output directory: {e}")
    raise


@app.post("/apply_style/")
async def apply_style(content: UploadFile = File(...), style_category: str = "impressionism"):
    logger.info(f"Received style application request with category '{style_category}'")

    # Validate the style category
    if style_category not in registry.styles:
        logger.error(f"Invalid style category: '{style_category}'")
        return {"error": f"Invalid style category: '{style_category}'"}, 400

    model_path = f"models/{style_category}_model.pth"
    logger.info(f"Looking for model at: {model_path}")
    try:
        model.load_model(model_path)
    except FileNotFoundError:
        logger.error(f"Model for category '{style_category}' not found.")
        return {"error": f"Model for category '{style_category}' not found."}, 400
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {"error": "Failed to load the model. Please try again later."}, 500

    try:
        content_image = Image.open(BytesIO(await content.read())).convert("RGB")
    except Exception as e:
        logger.error(f"Error processing content image: {e}")
        return {"error": "Invalid content image. Please upload a valid image file."}, 400

    styled_image = model.apply_style(content_image, None)
    filename = secure_filename(content.filename)
    output_path = OUTPUT_DIR / f"styled_{filename}"
    processor.save_image(styled_image, output_path)
    logger.info(f"Styled image saved to {output_path}")

    return {"message": "Style applied successfully!", "output_path": str(output_path)}


@app.get("/")
def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to Artify! Use /apply_style to stylize your images."}
