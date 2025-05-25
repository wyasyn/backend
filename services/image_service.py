import numpy as np
from PIL import Image
import io
from typing import Union
from utils.logger import logger
from config.settings import settings

class ImageService:
   
    
    @staticmethod
    def validate_image_file(content_type: str, file_size: int) -> bool:
       
        # Check content type
        if not content_type.startswith('image/'):
            return False
        
        # Check file size
        if file_size > settings.max_file_size:
            return False
        
        return True
    
    @staticmethod
    def preprocess_image(image: Union[Image.Image, bytes]) -> np.ndarray:
        
        try:
            # Convert bytes to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize(settings.image_size)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Normalize pixel values to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            logger.debug(f"Image preprocessed to shape: {img_array.shape}")
            return img_array
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
       
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "has_transparency": image.mode in ('RGBA', 'LA', 'P')
        }

# Global image service instance
image_service = ImageService()