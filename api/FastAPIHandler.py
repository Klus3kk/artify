"""
Updated FastAPI Backend for Artify
Industry-grade style transfer API with proper error handling
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import sys
import time
import asyncio
from pathlib import Path
from PIL import Image
from io import BytesIO
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from utilities.Logger import Logger

# Setup logging
logger = Logger.setup_logger(log_file="api.log", log_level=logging.INFO)

# Initialize FastAPI
app = FastAPI(
    title="Artify API",
    description="Industry-grade neural style transfer API",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
processor = ImageProcessor()
registry = StyleRegistry()
model = None  # Will be initialized on first request

# Directories
OUTPUT_DIR = Path("images/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Response models
class StyleTransferResponse(BaseModel):
    message: str
    output_path: str
    processing_time: float
    model_mode: str
    style_category: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    available_styles: List[str]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str

# Initialize model lazily
async def get_model():
    """Initialize model on first request"""
    global model
    if model is None:
        try:
            logger.info("Initializing StyleTransferModel...")
            model = StyleTransferModel()
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise HTTPException(status_code=500, detail="Model initialization failed")
    return model

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        version="2.0.0",
        available_styles=list(registry.styles.keys())
    )

@app.get("/styles")
async def list_styles():
    """List available style categories"""
    return {
        "styles": registry.styles,
        "total_categories": len(registry.styles),
        "message": "Available style categories"
    }

@app.post("/apply_style/", response_model=StyleTransferResponse)
async def apply_style(
    content: UploadFile = File(..., description="Content image file"),
    style_category: str = Form("impressionism", description="Style category to apply"),
    mode: str = Form("fast", description="Processing mode: 'fast' or 'quality'"),
    output_format: str = Form("jpg", description="Output format: 'jpg' or 'png'")
):
    """
    Apply style transfer to uploaded image
    
    - **content**: Upload your content image (JPG, PNG)
    - **style_category**: Choose from available style categories
    - **mode**: 'fast' for speed (0.05s) or 'quality' for best results (0.3s)
    - **output_format**: Output file format
    """
    
    start_time = time.time()
    
    try:
        # Validate inputs
        if style_category not in registry.styles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style category. Available: {list(registry.styles.keys())}"
            )
        
        if mode not in ["fast", "quality"]:
            raise HTTPException(
                status_code=400,
                detail="Mode must be 'fast' or 'quality'"
            )
        
        if output_format not in ["jpg", "png"]:
            raise HTTPException(
                status_code=400,
                detail="Output format must be 'jpg' or 'png'"
            )
        
        # Validate file type
        if not content.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Get model
        style_model = await get_model()
        
        # Set model mode
        if mode == "fast":
            style_model.set_fast_mode()
        else:
            style_model.set_quality_mode()
        
        logger.info(f"Processing request: style={style_category}, mode={mode}")
        
        # Process content image
        try:
            content_data = await content.read()
            content_image = Image.open(BytesIO(content_data)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Get style image
        try:
            style_image_path = registry.get_random_style_image(style_category)
            style_image = processor.preprocess_image(style_image_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load style image: {str(e)}"
            )
        
        # Apply style transfer
        try:
            styled_image = style_model.apply_style(content_image, style_image)
        except Exception as e:
            logger.error(f"Style transfer failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Style transfer failed: {str(e)}"
            )
        
        # Save result
        try:
            filename = f"styled_{int(time.time())}_{content.filename}"
            if output_format == "png":
                filename = filename.replace(".jpg", ".png").replace(".jpeg", ".png")
            else:
                filename = filename.replace(".png", ".jpg")
            
            output_path = OUTPUT_DIR / filename
            processor.save_image(styled_image, output_path)
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save result: {str(e)}"
            )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Style transfer completed in {processing_time:.3f}s")
        
        return StyleTransferResponse(
            message="Style applied successfully!",
            output_path=str(output_path),
            processing_time=processing_time,
            model_mode=mode,
            style_category=style_category
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/download/{filename}")
async def download_result(filename: str):
    """Download processed image"""
    
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.post("/apply_style_batch/")
async def apply_style_batch(
    content_images: List[UploadFile] = File(..., description="Multiple content images"),
    style_category: str = Form("impressionism", description="Style category to apply"),
    mode: str = Form("fast", description="Processing mode"),
):
    """
    Apply style transfer to multiple images
    Maximum 10 images per batch
    """
    
    if len(content_images) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )
    
    # Validate style category
    if style_category not in registry.styles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style category. Available: {list(registry.styles.keys())}"
        )
    
    # Get model
    style_model = await get_model()
    
    # Set mode
    if mode == "fast":
        style_model.set_fast_mode()
    else:
        style_model.set_quality_mode()
    
    results = []
    
    for i, content in enumerate(content_images):
        try:
            # Process each image
            content_data = await content.read()
            content_image = Image.open(BytesIO(content_data)).convert("RGB")
            
            # Get style image
            style_image_path = registry.get_random_style_image(style_category)
            style_image = processor.preprocess_image(style_image_path)
            
            # Apply style
            styled_image = style_model.apply_style(content_image, style_image)
            
            # Save
            filename = f"batch_{int(time.time())}_{i}_{content.filename}"
            output_path = OUTPUT_DIR / filename
            processor.save_image(styled_image, output_path)
            
            results.append({
                "original_filename": content.filename,
                "output_path": str(output_path),
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Failed to process {content.filename}: {e}")
            results.append({
                "original_filename": content.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "message": f"Processed {len(content_images)} images",
        "results": results,
        "style_category": style_category,
        "mode": mode
    }

@app.get("/performance_test/")
async def performance_test():
    """Test API performance with dummy data"""
    
    try:
        # Get model
        style_model = await get_model()
        
        # Create test image
        test_image = Image.new("RGB", (512, 512), color="red")
        
        # Test both modes
        results = {}
        
        for mode in ["fast", "quality"]:
            if mode == "fast":
                style_model.set_fast_mode()
            else:
                style_model.set_quality_mode()
            
            start_time = time.time()
            styled_image = style_model.apply_style(test_image, test_image)
            processing_time = time.time() - start_time
            
            results[mode] = {
                "processing_time": processing_time,
                "fps": 1.0 / processing_time if processing_time > 0 else 0
            }
        
        return {
            "message": "Performance test completed",
            "results": results,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "api.FastAPIHandler:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )