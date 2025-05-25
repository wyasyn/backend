from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Configuration
    app_name: str = "Plant Disease Detection API"
    version: str = "1.0.0"
    description: str = "AI-powered plant disease detection using EfficientNetB3"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    
    # Model Configuration
    model_path: str = "best_model_32epochs.keras"
    backup_model_path: str = "efficientnet_32epochs_20250525_075651.keras"
    image_size: tuple = (300, 300)
    
    # API Limits
    max_batch_size: int = 10
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # CORS Configuration
    allow_origins: List[str] = ["*"]
    allow_credentials: bool = True
    allow_methods: List[str] = ["*"]
    allow_headers: List[str] = ["*"]
    
    # Logging
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()