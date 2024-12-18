from fastapi import FastAPI, File, UploadFile
from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from PIL import Image
from io import BytesIO
import os

app = FastAPI()
processor = ImageProcessor()
model = StyleTransferModel()
registry = StyleRegistry()

OUTPUT_DIR = "images/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/apply_style/")
async def apply_style(content: UploadFile = File(...), style_category: str = "impressionism"):
    # Dynamically load the model for the selected style
    model_path = f"models/{style_category}_model.pth"
    model.load_model_from_gcloud("your-bucket-name", model_path)

    # Read content image
    content_image = Image.open(BytesIO(await content.read())).convert("RGB")

    # Apply the style
    styled_image = model.apply_style(content_image, None)

    # Save and return the styled image
    output_path = os.path.join(OUTPUT_DIR, f"styled_{content.filename}")
    processor.save_image(styled_image, output_path)

    return {"message": "Style applied successfully!", "output_path": output_path}

@app.get("/")
def root():
    return {"message": "Welcome to Artify! Use /apply_style to stylize your images."}
