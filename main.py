from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import asyncio
from huggingface_hub import hf_hub_download

# Import configuration and services
from config.settings import settings
from utils.logger import logger

# Import routers
from api.endpoints import health, prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Plant Disease Detection API...")
    
    try:
        # Get cache directory from environment variable or use a default
        cache_dir = os.environ.get("HF_HOME", "/tmp/huggingface")
        
        logger.info(f"Downloading model from Hugging Face Hub using cache_dir: {cache_dir}...")
        
        # Download model from Hugging Face
        # Replace with your actual Hugging Face model repo and filename
        model_path = hf_hub_download(
            repo_id=settings.hf_model_repo,  # e.g., "your-username/plant-disease-model"
            filename=settings.hf_model_filename,  # e.g., "model.keras" or "pytorch_model.bin"
            cache_dir=cache_dir
        )
        
        logger.info(f"Model downloaded to: {model_path}")
        
        # Load the model using your existing model loader
        from models.ml_model import model_instance
        success = await asyncio.to_thread(model_instance.load_model_from_path, model_path)
        
        if success:
            logger.info("Model loaded successfully from Hugging Face during startup")
            
            # Optional: Warm-up the model with a dummy prediction
            try:
                await asyncio.to_thread(model_instance.warm_up)
                logger.info("Model warm-up completed successfully")
            except Exception as e:
                logger.warning(f"Model warm-up failed, but continuing: {e}")
                # Don't fail startup if warm-up fails - the model is still loaded
                
            # Store model in app state for access in endpoints
            app.state.model = model_instance
        else:
            logger.error("Failed to load model from Hugging Face")
            raise RuntimeError("Failed to load plant disease model")
    
    except Exception as e:
        logger.exception("Failed during startup:")
        raise RuntimeError("Failed to load plant disease model from Hugging Face") from e
    
    yield
    
    # Shutdown
    logger.info("Shutting down Plant Disease Detection API...")
    if hasattr(app.state, "model"):
        logger.info("Releasing model resources")
        del app.state.model

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)

# Include routers
app.include_router(health.router)
app.include_router(prediction.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": settings.app_name,
        "version": settings.version,
        "model_source": "Hugging Face Hub",
        "endpoints": {
            "/predict": "POST - Upload image for disease prediction",
            "/predict/batch": "POST - Upload multiple images for batch prediction",
            "/health": "GET - Health check",
            "/health/model": "GET - Model information",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Run the application using uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level
    )