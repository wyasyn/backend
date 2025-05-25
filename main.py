from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


# Import configuration and services
from config.settings import settings
from models.ml_model import model_instance
from utils.logger import logger

# Import routers
from api.endpoints import health, prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # Startup
    logger.info("Starting Plant Disease Detection API...")
    
    # Load model
    success = model_instance.load_model()
    if success:
        logger.info("Model loaded successfully during startup")
    else:
        logger.warning("Model not loaded during startup. Manual loading may be required.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Plant Disease Detection API...")

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