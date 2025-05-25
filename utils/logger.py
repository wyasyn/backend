import logging
import os
from logging.handlers import RotatingFileHandler
from config.settings import settings

def setup_logger(name: str) -> logging.Logger:
    """
    Set up logger with file and console handlers
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(settings.log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional - only if log directory exists or can be created)
    try:
        log_dir = os.path.dirname(settings.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, continue with console logging only
        logger.warning(f"Could not set up file logging: {e}")
    
    return logger

# Create the main logger instance
logger = setup_logger("plant_disease_api")