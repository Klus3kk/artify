from fastapi import FastAPI, File, UploadFile
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from PIL import Image
from io import BytesIO
from pathlib import Path
import logging
from werkzeug.utils import secure_filename

# Initialize FastAPI and shared components
app = FastAPI()
processor = ImageProcessor()
model = StyleTransferModel()
registry = StyleRegistry()

# Output directory
OUTPUT_DIR = Path("images/output")
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logging.error(f"Failed to create output directory: {e}")
    raise

@app.post("/apply_style/")
async def apply_style(content: UploadFile = File(...), style_category: str = "impressionism"):
    # Dynamically load the model for the selected style
    model_path = f"models/{style_category}_model.pth"
    try:
        model.load_model_from_gcloud("your-bucket-name", model_path)
    except FileNotFoundError:
        return {"error": f"Model for category '{style_category}' not found."}
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return {"error": "Failed to load the model. Please try again later."}

    # Read content image
    try:
        content_image = Image.open(BytesIO(await content.read())).convert("RGB")
    except Exception as e:
        logging.error(f"Error processing content image: {e}")
        return {"error": "Invalid content image. Please upload a valid image file."}

    # Apply the style
    styled_image = model.apply_style(content_image, None)

    # Save and return the styled image
    filename = secure_filename(content.filename)
    output_path = OUTPUT_DIR / f"styled_{filename}"
    processor.save_image(styled_image, output_path)

    return {"message": "Style applied successfully!", "output_path": str(output_path)}

@app.get("/")
def root():
    return {"message": "Welcome to Artify! Use /apply_style to stylize your images."}
