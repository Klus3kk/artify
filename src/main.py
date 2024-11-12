import sys
sys.path.append("..")  

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from src.model import load_model, apply_style  

app = FastAPI()
style_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global style_model
    # Initialize model at startup
    style_model = load_model('models/style.pth')
    yield  # Application runs here
    # Optionally, cleanup any resources after app shutdown

app = FastAPI(lifespan=lifespan)

@app.post("/style_transfer/")
async def style_transfer(image: UploadFile = File(...)):
    # Here, we can access `style_model` as it was initialized in the lifespan
    pass  
