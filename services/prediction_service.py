import numpy as np
from typing import Dict, List
from models.ml_model import model_instance
from services.image_service import image_service
from utils.logger import logger
from utils.constants import CLEAN_CLASS_NAMES

class PredictionService:
  
    
    def __init__(self):
        self.model = model_instance
    
    def predict_single_image(self, image_data: bytes) -> Dict:
        
        try:
            # Preprocess image
            processed_image = image_service.preprocess_image(image_data)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Format results
            results = self.model.format_predictions(predictions)
            
            # Generate message
            message = self._generate_message(results["predicted_class"], results["confidence"])
            
            return {
                "predicted_class": results["predicted_class"],
                "confidence": results["confidence"],
                "all_predictions": results["all_predictions"],
                "message": message
            }
        
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch_images(self, images_data: List[bytes]) -> List[Dict]:
       
        results = []
        
        for i, image_data in enumerate(images_data):
            try:
                result = self.predict_single_image(image_data)
                results.append({
                    "success": True,
                    "prediction": result["predicted_class"],
                    "confidence": result["confidence"],
                    "top_predictions": result["all_predictions"]
                })
                
            except Exception as e:
                logger.error(f"Batch prediction failed for image {i}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def _generate_message(self, predicted_class: str, confidence: float) -> str:
        
        if "healthy" in predicted_class.lower():
            plant = predicted_class.split(' - ')[0] if ' - ' in predicted_class else "plant"
            return f"Good news! Your {plant} appears healthy (confidence: {confidence:.1%})."
        else:
            parts = predicted_class.split(' - ')
            if len(parts) >= 2:
                plant = parts[0]
                disease = parts[1]
                return f"Disease detected: {disease} in {plant} (confidence: {confidence:.1%}). Consider appropriate treatment."
            else:
                return f"Disease detected: {predicted_class} (confidence: {confidence:.1%}). Consider appropriate treatment."

# Global prediction service instance
prediction_service = PredictionService()