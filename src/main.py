import sys
sys.path.append("..")  

import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from src.model import load_model, apply_style  

app = FastAPI()
style_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global style_model
    style_model = load_model('models/test.pth')
    yield  # Application runs here
    # Optionally, cleanup any resources after app shutdown

app = FastAPI(lifespan=lifespan)

@app.post("/style_transfer/")
async def style_transfer(image: UploadFile = File(...)):
    # Save uploaded image to a temporary file
    image_bytes = await image.read()
    image_path = "temp_uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Apply style transfer
    output_image = apply_style(style_model, image_path)

    # Convert output to bytes for HTTP response
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")
