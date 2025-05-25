import os
from typing import List

class Settings:
    # App configuration
    app_name: str = "Plant Disease Detection API"
    description: str = "AI-powered plant disease detection using computer vision"
    version: str = "1.0.0"
    
    # Server configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    reload: bool = os.getenv("RELOAD", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    # CORS configuration
    allow_origins: List[str] = os.getenv("ALLOW_ORIGINS", "*").split(",")
    allow_credentials: bool = os.getenv("ALLOW_CREDENTIALS", "true").lower() == "true"
    allow_methods: List[str] = os.getenv("ALLOW_METHODS", "*").split(",")
    allow_headers: List[str] = os.getenv("ALLOW_HEADERS", "*").split(",")
    
    # Hugging Face model configuration
    hf_model_repo: str = os.getenv("HF_MODEL_REPO", "yasyn14/smart-leaf-model")
    hf_model_filename: str = os.getenv("HF_MODEL_FILENAME", "best_model_32epochs.keras")
    hf_cache_dir: str = os.getenv("HF_HOME", "/tmp/huggingface")
    
    # Logging configuration
    log_format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")
    log_max_bytes: int = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 10MB
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", 5))
    
    # Model configuration
    model_input_size: tuple = (300, 300)  # EfficientNetB3 input size
    image_size: tuple = (300, 300)
    model_batch_size: int = int(os.getenv("MODEL_BATCH_SIZE", 32))
    
    # File upload configuration
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

# Create settings instance
settings = Settings()