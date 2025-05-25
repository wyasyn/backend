import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from typing import Optional
from utils.logger import logger
from config.settings import settings
from utils.constants import CLASS_NAMES, CLEAN_CLASS_NAMES

class PlantDiseaseModel:
    
    
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.is_loaded: bool = False
        self.model_path: Optional[str] = None
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        
        try:
            # Use provided path or default from settings
            path = model_path or settings.model_path
            
            # Try backup path if main path fails
            if not os.path.exists(path):
                logger.warning(f"Model not found at {path}, trying backup path")
                path = settings.backup_model_path
                
            if not os.path.exists(path):
                logger.error(f"Model not found at {path}")
                return False
            
            # Load the model
            self.model = load_model(path)
            self.model_path = path
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    def predict(self, image_array: np.ndarray) -> np.ndarray:
       
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            predictions = self.model.predict(image_array, verbose=0)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> dict:
       
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        return {
            "model_type": "EfficientNetB3",
            "input_shape": list(settings.image_size) + [3],
            "num_classes": len(CLASS_NAMES),
            "classes": CLEAN_CLASS_NAMES,
            "model_path": self.model_path
        }
    
    def format_predictions(self, predictions: np.ndarray, top_k: int = 5) -> dict:
       
        try:
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = CLEAN_CLASS_NAMES[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])
            
            # Get top k predictions
            top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
            all_predictions = {}
            
            for idx in top_k_indices:
                class_name = CLEAN_CLASS_NAMES[idx]
                prob = float(predictions[0][idx])
                all_predictions[class_name] = round(prob, 4)
            
            return {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "all_predictions": all_predictions
            }
        
        except Exception as e:
            logger.error(f"Error formatting predictions: {str(e)}")
            raise RuntimeError(f"Error formatting predictions: {str(e)}")

# Global model instance
model_instance = PlantDiseaseModel()