"""
Enhanced FastAPI Backend for Artify
Production-ready API with video processing, batch operations, and comprehensive error handling
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import os
import sys
import time
import asyncio
import aiofiles
import cv2
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from io import BytesIO
import logging
import json
import uuid
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.StyleTransferModel import StyleTransferModel
from core.ImageProcessor import ImageProcessor
from utilities.StyleRegistry import StyleRegistry
from utilities.Logger import Logger
from evaluation.evaluation_metrics import ComprehensiveMetrics, PerformanceBenchmark

# Setup logging
logger = Logger.setup_logger(log_file="api.log", log_level=logging.INFO)

# Initialize FastAPI with comprehensive metadata
app = FastAPI(
    title="Artify API",
    description="Industry-grade neural style transfer API with real-time processing capabilities",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global components
processor = ImageProcessor()
registry = StyleRegistry()
model = None
evaluation_metrics = None
performance_benchmark = None

# Directories
OUTPUT_DIR = Path("api_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# In-memory task tracking
active_tasks: Dict[str, Dict] = {}

# Enhanced response models
class StyleTransferResponse(BaseModel):
    task_id: str
    message: str
    output_path: Optional[str] = None
    processing_time: Optional[float] = None
    model_mode: str
    style_category: Optional[str] = None
    quality_metrics: Optional[Dict] = None

class BatchStyleTransferResponse(BaseModel):
    batch_id: str
    message: str
    total_images: int
    processed_images: int
    failed_images: int
    results: List[StyleTransferResponse]
    total_processing_time: float

class VideoStyleTransferResponse(BaseModel):
    task_id: str
    message: str
    output_path: Optional[str] = None
    processing_time: Optional[float] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    available_styles: List[str]
    model_loaded: bool
    gpu_available: bool
    memory_usage: Optional[Dict] = None

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str
    request_id: Optional[str] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict] = None
    created_at: str
    updated_at: str

class ModelPerformanceResponse(BaseModel):
    fps: float
    memory_usage_mb: float
    model_size_mb: float
    inference_time_ms: float
    benchmark_date: str

# Enhanced initialization
async def initialize_model():
    """Initialize model and evaluation components"""
    global model, evaluation_metrics, performance_benchmark
    
    if model is None:
        try:
            logger.info("Initializing StyleTransferModel...")
            model = StyleTransferModel()
            
            # Initialize evaluation components
            evaluation_metrics = ComprehensiveMetrics()
            performance_benchmark = PerformanceBenchmark()
            
            logger.info("Model and evaluation components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise HTTPException(status_code=500, detail="Model initialization failed")
    
    return model

# Utility functions
def generate_task_id() -> str:
    """Generate unique task ID"""
    return str(uuid.uuid4())

def get_memory_usage() -> Dict:
    """Get current memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**2),
                "gpu_memory_reserved": torch.cuda.memory_reserved() / (1024**2),
                "gpu_max_memory": torch.cuda.max_memory_allocated() / (1024**2)
            }
    except:
        pass
    return {"gpu_available": False}

async def save_uploaded_file(upload_file: UploadFile, filename: str) -> Path:
    """Save uploaded file to temporary directory"""
    file_path = TEMP_DIR / filename
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()
        await f.write(content)
    
    return file_path

def cleanup_temp_files(*file_paths: Path):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")

# Main endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        version="3.0.0",
        available_styles=list(registry.styles.keys()),
        model_loaded=model is not None,
        gpu_available=gpu_available,
        memory_usage=get_memory_usage()
    )

@app.post("/style-transfer/", response_model=StyleTransferResponse)
async def apply_style_transfer(
    background_tasks: BackgroundTasks,
    content: UploadFile = File(..., description="Content image file"),
    style: UploadFile = File(None, description="Style image file (optional if using style_category)"),
    style_category: Optional[str] = Form(None, description="Predefined style category"),
    quality_mode: str = Form("balanced", description="Quality mode: fast, balanced, high"),
    output_size: Optional[int] = Form(512, description="Output image size"),
    evaluate_quality: bool = Form(False, description="Include quality evaluation metrics"),
    async_processing: bool = Form(False, description="Process asynchronously")
):
    """Enhanced style transfer with quality evaluation and async processing"""
    
    task_id = generate_task_id()
    
    try:
        # Initialize model if needed
        await initialize_model()
        
        # Validate inputs
        if not style and not style_category:
            raise HTTPException(status_code=400, detail="Either style image or style_category must be provided")
        
        # Save uploaded files
        content_path = await save_uploaded_file(content, f"{task_id}_content.jpg")
        
        style_path = None
        if style:
            style_path = await save_uploaded_file(style, f"{task_id}_style.jpg")
        elif style_category:
            # Get style from registry
            style_images = registry.get_style_images(style_category)
            if not style_images:
                raise HTTPException(status_code=400, detail=f"Style category '{style_category}' not found")
            style_path = Path(style_images[0])
        
        if async_processing:
            # Start background task
            active_tasks[task_id] = {
                "status": "pending",
                "progress": 0.0,
                "message": "Task queued for processing",
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            background_tasks.add_task(
                process_style_transfer_async,
                task_id, content_path, style_path, style_category,
                quality_mode, output_size, evaluate_quality
            )
            
            return StyleTransferResponse(
                task_id=task_id,
                message="Style transfer started asynchronously",
                model_mode="async",
                style_category=style_category
            )
        else:
            # Process synchronously
            result = await process_style_transfer_sync(
                task_id, content_path, style_path, style_category,
                quality_mode, output_size, evaluate_quality
            )
            
            # Cleanup
            background_tasks.add_task(cleanup_temp_files, content_path)
            if style_path and style_path.parent == TEMP_DIR:
                background_tasks.add_task(cleanup_temp_files, style_path)
            
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Style transfer failed: {e}")
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")

async def process_style_transfer_sync(
    task_id: str, content_path: Path, style_path: Path, style_category: Optional[str],
    quality_mode: str, output_size: int, evaluate_quality: bool
) -> StyleTransferResponse:
    """Process style transfer synchronously"""
    
    start_time = time.time()
    
    try:
        # Load and preprocess images
        content_image = processor.load_image(str(content_path), target_size=(output_size, output_size))
        style_image = processor.load_image(str(style_path), target_size=(output_size, output_size))
        
        # Apply style transfer
        stylized_image = model.stylize(content_image, style_image)
        
        # Save result
        output_filename = f"{task_id}_stylized.jpg"
        output_path = OUTPUT_DIR / output_filename
        processor.save_image(stylized_image, str(output_path))
        
        processing_time = time.time() - start_time
        
        # Quality evaluation if requested
        quality_metrics = None
        if evaluate_quality and evaluation_metrics:
            try:
                quality_metrics = evaluation_metrics.evaluate_style_transfer(
                    stylized_image, content_image, style_image
                )
            except Exception as e:
                logger.warning(f"Quality evaluation failed: {e}")
        
        return StyleTransferResponse(
            task_id=task_id,
            message="Style transfer completed successfully",
            output_path=str(output_path),
            processing_time=processing_time,
            model_mode=quality_mode,
            style_category=style_category,
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Sync processing failed: {e}")
        raise

async def process_style_transfer_async(
    task_id: str, content_path: Path, style_path: Path, style_category: Optional[str],
    quality_mode: str, output_size: int, evaluate_quality: bool
):
    """Process style transfer asynchronously"""
    
    try:
        # Update task status
        active_tasks[task_id].update({
            "status": "processing",
            "progress": 0.1,
            "message": "Loading images...",
            "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Process (similar to sync version but with progress updates)
        result = await process_style_transfer_sync(
            task_id, content_path, style_path, style_category,
            quality_mode, output_size, evaluate_quality
        )
        
        # Update final status
        active_tasks[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "Style transfer completed",
            "result": result.dict(),
            "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Cleanup
        cleanup_temp_files(content_path)
        if style_path.parent == TEMP_DIR:
            cleanup_temp_files(style_path)
            
    except Exception as e:
        logger.error(f"Async processing failed: {e}")
        active_tasks[task_id].update({
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
            "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        })

@app.post("/batch-style-transfer/", response_model=BatchStyleTransferResponse)
async def batch_style_transfer(
    background_tasks: BackgroundTasks,
    content_images: List[UploadFile] = File(..., description="Multiple content images"),
    style: UploadFile = File(None, description="Style image for all content images"),
    style_category: Optional[str] = Form(None, description="Style category for all images"),
    quality_mode: str = Form("balanced", description="Quality mode for all images"),
    output_size: int = Form(512, description="Output size for all images")
):
    """Batch process multiple images with the same style"""
    
    batch_id = generate_task_id()
    start_time = time.time()
    
    try:
        await initialize_model()
        
        if not style and not style_category:
            raise HTTPException(status_code=400, detail="Either style image or style_category must be provided")
        
        # Process style
        style_path = None
        if style:
            style_path = await save_uploaded_file(style, f"{batch_id}_style.jpg")
        elif style_category:
            style_images = registry.get_style_images(style_category)
            if not style_images:
                raise HTTPException(status_code=400, detail=f"Style category '{style_category}' not found")
            style_path = Path(style_images[0])
        
        results = []
        processed_count = 0
        failed_count = 0
        
        # Process each content image
        for i, content_image in enumerate(content_images):
            try:
                task_id = f"{batch_id}_{i}"
                content_path = await save_uploaded_file(content_image, f"{task_id}_content.jpg")
                
                result = await process_style_transfer_sync(
                    task_id, content_path, style_path, style_category,
                    quality_mode, output_size, False  # No quality eval for batch
                )
                
                results.append(result)
                processed_count += 1
                
                # Cleanup content file
                background_tasks.add_task(cleanup_temp_files, content_path)
                
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                failed_count += 1
                
                # Add failed result
                results.append(StyleTransferResponse(
                    task_id=f"{batch_id}_{i}",
                    message=f"Failed: {str(e)}",
                    model_mode=quality_mode,
                    style_category=style_category
                ))
        
        total_time = time.time() - start_time
        
        # Cleanup style file if uploaded
        if style and style_path.parent == TEMP_DIR:
            background_tasks.add_task(cleanup_temp_files, style_path)
        
        return BatchStyleTransferResponse(
            batch_id=batch_id,
            message=f"Batch processing completed: {processed_count} successful, {failed_count} failed",
            total_images=len(content_images),
            processed_images=processed_count,
            failed_images=failed_count,
            results=results,
            total_processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/video-style-transfer/", response_model=VideoStyleTransferResponse)
async def video_style_transfer(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Input video file"),
    style: UploadFile = File(None, description="Style image"),
    style_category: Optional[str] = Form(None, description="Style category"),
    output_fps: Optional[int] = Form(None, description="Output video FPS (default: input FPS)"),
    max_resolution: int = Form(512, description="Maximum resolution for processing"),
    async_processing: bool = Form(True, description="Process video asynchronously")
):
    """Video style transfer with frame-by-frame processing"""
    
    task_id = generate_task_id()
    
    try:
        await initialize_model()
        
        if not style and not style_category:
            raise HTTPException(status_code=400, detail="Either style image or style_category must be provided")
        
        # Save uploaded video
        video_path = await save_uploaded_file(video, f"{task_id}_input.mp4")
        
        # Process style
        style_path = None
        if style:
            style_path = await save_uploaded_file(style, f"{task_id}_style.jpg")
        elif style_category:
            style_images = registry.get_style_images(style_category)
            if not style_images:
                raise HTTPException(status_code=400, detail=f"Style category '{style_category}' not found")
            style_path = Path(style_images[0])
        
        if async_processing:
            # Start background processing
            active_tasks[task_id] = {
                "status": "pending",
                "progress": 0.0,
                "message": "Video queued for processing",
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            background_tasks.add_task(
                process_video_async,
                task_id, video_path, style_path, output_fps, max_resolution
            )
            
            return VideoStyleTransferResponse(
                task_id=task_id,
                message="Video processing started asynchronously"
            )
        else:
            # Process synchronously (not recommended for large videos)
            result = await process_video_sync(
                task_id, video_path, style_path, output_fps, max_resolution
            )
            
            # Cleanup
            background_tasks.add_task(cleanup_temp_files, video_path)
            if style and style_path.parent == TEMP_DIR:
                background_tasks.add_task(cleanup_temp_files, style_path)
            
            return result
            
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

async def process_video_sync(
    task_id: str, video_path: Path, style_path: Path, 
    output_fps: Optional[int], max_resolution: int
) -> VideoStyleTransferResponse:
    """Process video synchronously"""
    
    start_time = time.time()
    
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_fps is None:
            output_fps = fps
        
        # Calculate output size maintaining aspect ratio
        if width > height:
            out_width = min(max_resolution, width)
            out_height = int(height * out_width / width)
        else:
            out_height = min(max_resolution, height)
            out_width = int(width * out_height / height)
        
        # Setup output video
        output_path = OUTPUT_DIR / f"{task_id}_stylized.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (out_width, out_height))
        
        # Load style image
        style_image = processor.load_image(str(style_path), target_size=(out_width, out_height))
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL/tensor format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((out_width, out_height))
            frame_tensor = processor.pil_to_tensor(frame_pil)
            
            # Apply style transfer
            stylized_tensor = model.stylize(frame_tensor, style_image)
            stylized_pil = processor.tensor_to_pil(stylized_tensor)
            stylized_frame = cv2.cvtColor(np.array(stylized_pil), cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(stylized_frame)
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        
        return VideoStyleTransferResponse(
            task_id=task_id,
            message="Video processing completed successfully",
            output_path=str(output_path),
            processing_time=processing_time,
            total_frames=frame_count,
            fps=output_fps
        )
        
    except Exception as e:
        logger.error(f"Video sync processing failed: {e}")
        raise

async def process_video_async(
    task_id: str, video_path: Path, style_path: Path,
    output_fps: Optional[int], max_resolution: int
):
    """Process video asynchronously with progress updates"""
    
    try:
        # Similar to sync version but with progress updates
        result = await process_video_sync(task_id, video_path, style_path, output_fps, max_resolution)
        
        active_tasks[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "message": "Video processing completed",
            "result": result.dict(),
            "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Cleanup
        cleanup_temp_files(video_path)
        if style_path.parent == TEMP_DIR:
            cleanup_temp_files(style_path)
            
    except Exception as e:
        logger.error(f"Video async processing failed: {e}")
        active_tasks[task_id].update({
            "status": "failed",
            "message": f"Video processing failed: {str(e)}",
            "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        })

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of async task"""
    
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        message=task_info["message"],
        result=task_info.get("result"),
        created_at=task_info["created_at"],
        updated_at=task_info["updated_at"]
    )

@app.get("/performance", response_model=ModelPerformanceResponse)
async def get_model_performance():
    """Get current model performance metrics"""
    
    await initialize_model()
    
    if performance_benchmark is None:
        raise HTTPException(status_code=503, detail="Performance benchmark not available")
    
    try:
        # Run performance benchmark
        speed_results = performance_benchmark.benchmark_speed(model, iterations=10)
        memory_results = performance_benchmark.benchmark_memory(model)
        complexity_results = performance_benchmark.model_complexity(model)
        
        return ModelPerformanceResponse(
            fps=speed_results["fps"],
            memory_usage_mb=memory_results["peak_memory_mb"],
            model_size_mb=complexity_results["model_size_mb"],
            inference_time_ms=speed_results["mean_time"] * 1000,
            benchmark_date=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        raise HTTPException(status_code=500, detail="Performance benchmark failed")

@app.get("/styles", response_model=Dict[str, List[str]])
async def list_available_styles():
    """List all available style categories and their images"""
    
    styles_info = {}
    
    for category, info in registry.styles.items():
        style_images = registry.get_style_images(category)
        styles_info[category] = {
            "description": info.get("description", ""),
            "image_count": len(style_images),
            "sample_images": style_images[:3]  # First 3 as samples
        }
    
    return styles_info

@app.get("/download/{filename}")
async def download_result(filename: str):
    """Download processed result file"""
    
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

@app.delete("/cleanup/{task_id}")
async def cleanup_task_files(task_id: str):
    """Clean up files associated with a task"""
    
    try:
        # Remove task from active tasks
        if task_id in active_tasks:
            del active_tasks[task_id]
        
        # Clean up output files
        for file_path in OUTPUT_DIR.glob(f"{task_id}*"):
            file_path.unlink()
        
        # Clean up temp files  
        for file_path in TEMP_DIR.glob(f"{task_id}*"):
            file_path.unlink()
        
        return {"message": f"Cleaned up files for task {task_id}"}
        
    except Exception as e:
        logger.error(f"Cleanup failed for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

@app.post("/admin/cleanup-all")
async def cleanup_all_files(
    credentials: HTTPAuthorizationCredentials = security
):
    """Admin endpoint to clean up all temporary and output files"""
    
    # Basic auth check (implement proper auth for production)
    if not credentials or credentials.credentials != "admin-secret":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Clear active tasks
        active_tasks.clear()
        
        # Clean up all files
        for file_path in OUTPUT_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        
        return {"message": "All files cleaned up successfully"}
        
    except Exception as e:
        logger.error(f"Global cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

@app.get("/metrics/usage")
async def get_usage_metrics():
    """Get API usage metrics"""
    
    return {
        "active_tasks": len(active_tasks),
        "disk_usage": {
            "output_files": len(list(OUTPUT_DIR.glob("*"))),
            "temp_files": len(list(TEMP_DIR.glob("*")))
        },
        "memory_usage": get_memory_usage(),
        "uptime": time.time() - start_time if 'start_time' in globals() else 0
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc) if app.debug else None,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

# Middleware for request tracking
@app.middleware("http")
async def add_request_id(request, call_next):
    """Add request ID for tracking"""
    
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    
    global start_time
    start_time = time.time()
    
    logger.info("Artify API starting up...")
    
    # Pre-load model for faster first request
    try:
        await initialize_model()
        logger.info("Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-load model: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    
    logger.info("Artify API shutting down...")
    
    # Cancel any active background tasks
    active_tasks.clear()
    
    # Optional: Clean up temp files
    try:
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
    except Exception as e:
        logger.warning(f"Cleanup on shutdown failed: {e}")

def create_app():
    """Factory function to create app instance"""
    return app

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api.FastAPIHandler:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )