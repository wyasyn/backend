from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import asyncio
import signal
import sys
import atexit
from typing import Optional
from huggingface_hub import hf_hub_download

# Import configuration and services
from config.settings import settings
from utils.logger import logger

# Import routers
from api.endpoints import health, prediction

# Global variables for cleanup
model_instance: Optional[object] = None
app_instance: Optional[FastAPI] = None

def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully"""
    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM"
    }
    signal_name = signal_names.get(signum, f"Signal {signum}")
    
    logger.info(f"Received {signal_name}. Initiating graceful shutdown...")
    
    # Perform immediate cleanup
    cleanup_resources()
    
    # Exit gracefully
    logger.info("Shutdown complete. Goodbye!")
    sys.exit(0)

def cleanup_resources() -> None:
    """Cleanup function for resources"""
    global model_instance
    
    try:
        logger.info("Cleaning up resources...")
        
        # Cleanup model resources (with safety check)
        if 'model_instance' in globals() and model_instance and hasattr(model_instance, 'cleanup'):
            logger.info("Cleaning up model resources")
            model_instance.cleanup()
        
        # Clear any global references
        model_instance = None
        
        logger.info("Resource cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with startup and shutdown logic"""
    global model_instance
    
    # Startup
    logger.info("🚀 Starting Plant Disease Detection API...")
    
    try:
        # Get cache directory from environment variable or use a default
        cache_dir = os.environ.get("HF_HOME", "/tmp/huggingface")
        
        logger.info(f"📥 Downloading model from Hugging Face Hub using cache_dir: {cache_dir}...")
        
        # Download model from Hugging Face with error handling
        try:
            model_path = hf_hub_download(
                repo_id=settings.hf_model_repo,
                filename=settings.hf_model_filename,
                cache_dir=cache_dir
            )
            logger.info(f"✅ Model downloaded to: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download model from Hugging Face: {e}")
            raise RuntimeError(f"Failed to download model: {e}") from e
        
        # Load the model using your existing model loader
        try:
            from models.ml_model import model_instance as ml_model
            model_instance = ml_model
            
            success = await asyncio.to_thread(model_instance.load_model_from_path, model_path)
            
            if success:
                logger.info("✅ Model loaded successfully from Hugging Face during startup")
                
                # Optional: Warm-up the model with a dummy prediction
                try:
                    await asyncio.to_thread(model_instance.warm_up)
                    logger.info("🔥 Model warm-up completed successfully")
                except Exception as e:
                    logger.warning(f"⚠️  Model warm-up failed, but continuing: {e}")
                    # Don't fail startup if warm-up fails - the model is still loaded
                
                # Store model in app state for access in endpoints
                app.state.model = model_instance
                app.state.model_loaded = True
                
            else:
                logger.error("❌ Failed to load model from Hugging Face")
                raise RuntimeError("Failed to load plant disease model")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e
        
        logger.info("🎉 API startup completed successfully!")
        
    except Exception as e:
        logger.exception("💥 Failed during startup:")
        # Cleanup any partial resources
        cleanup_resources()
        raise RuntimeError("Failed to start Plant Disease Detection API") from e
    
    # Yield control to the application
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Plant Disease Detection API...")
    
    try:
        # Mark model as unloaded
        if hasattr(app.state, "model_loaded"):
            app.state.model_loaded = False
        
        # Clean up model resources
        if hasattr(app.state, "model") and app.state.model:
            logger.info("🧹 Releasing model resources")
            
            # If your model has a cleanup method, call it
            if hasattr(app.state.model, 'cleanup'):
                app.state.model.cleanup()
            
            del app.state.model
        
        # Final cleanup
        cleanup_resources()
        
        logger.info("✅ Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan
)

# Store app instance globally for signal handlers
app_instance = app

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
    """Root endpoint with API information"""
    model_status = "loaded" if hasattr(app.state, "model_loaded") and app.state.model_loaded else "not loaded"
    
    return {
        "message": f"🌱 {settings.app_name}",
        "version": settings.version,
        "model_source": "Hugging Face Hub",
        "model_status": model_status,
        "endpoints": {
            "/predict": "POST - Upload image for disease prediction",
            "/predict/batch": "POST - Upload multiple images for batch prediction",
            "/health": "GET - Health check",
            "/health/model": "GET - Model information",
            "/docs": "GET - API documentation",
            "/redoc": "GET - Alternative API documentation"
        },
        "status": "healthy" if model_status == "loaded" else "starting"
    }

# Health check endpoint for the root level
@app.get("/ping")
async def ping():
    """Simple ping endpoint for load balancers"""
    return {"status": "alive", "message": "pong"}

# Setup signal handlers and cleanup
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    # Handle Ctrl+C (SIGINT) and termination (SIGTERM)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function to run at exit
    atexit.register(cleanup_resources)
    
    logger.info("Signal handlers registered for graceful shutdown")

if __name__ == "__main__":
    import uvicorn
    
    # Setup signal handlers
    setup_signal_handlers()
    
    try:
        logger.info(f"🚀 Starting server on {settings.host}:{settings.port}")
        
        # Run the application using uvicorn with better error handling
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level=settings.log_level,
            access_log=True,
            use_colors=True,
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user (Ctrl+C)")
        cleanup_resources()
        
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        cleanup_resources()
        sys.exit(1)
        
    finally:
        logger.info("👋 Server shutdown complete")