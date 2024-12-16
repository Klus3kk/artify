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
    content_image = Image.open(BytesIO(await content.read())).convert("RGB")
    style_image_path = registry.get_random_style_image(style_category)
    style_image = Image.open(style_image_path).convert("RGB")

    styled_image = model.apply_style(content_image, style_image)

    output_path = os.path.join(OUTPUT_DIR, f"styled_{content.filename}")
    processor.save_image(styled_image, output_path)

    return {"message": "Style applied successfully!", "output_path": output_path}

@app.get("/")
def root():
    return {"message": "Welcome to Artify! Use /apply_style to stylize your images."}
