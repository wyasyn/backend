import os
import numpy as np
from typing import Optional, Dict, Any
import tensorflow as tf
from PIL import Image
from utils.logger import logger

class MLModel:
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.is_loaded: bool = False
        self.model_path: Optional[str] = None
        
        # Disable OneDNN optimizations if needed
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load model from local path (legacy method for backward compatibility)
        """
        try:
            if model_path is None:
                # Try to load from default location if no path provided
                model_path = os.getenv("MODEL_PATH", "models/best_model_32epochs.keras")
            
            return self.load_model_from_path(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_model_from_path(self, model_path: str) -> bool:
        """
        Load model from a specific file path (used for Hugging Face models)
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load the model based on file extension
            if model_path.endswith('.keras'):
                self.model = tf.keras.models.load_model(model_path)
            elif model_path.endswith(('.h5', '.hdf5')):
                self.model = tf.keras.models.load_model(model_path)
            elif model_path.endswith('.pb'):
                # For SavedModel format
                self.model = tf.keras.models.load_model(model_path)
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return False
            
            self.model_path = model_path
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            self.model = None
            self.is_loaded = False
            return False
    
    def warm_up(self) -> None:
        """
        Warm up the model with a dummy prediction
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded")
        
        try:
            # Get the actual input shape from the model
            input_shape = self.model.input_shape
            height, width = input_shape[1], input_shape[2]
            
            # Create dummy input matching your model's expected input shape
            dummy_input = np.zeros((1, height, width, 3), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            logger.info("Model warm-up completed successfully")
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image, target_size: tuple = None) -> np.ndarray:
        """
        Preprocess image for model input
        """
        try:
            # Get target size from model if not provided
            if target_size is None and self.model is not None:
                input_shape = self.model.input_shape
                target_size = (input_shape[1], input_shape[2])  # (height, width)
            elif target_size is None:
                target_size = (300, 300)  # Default for EfficientNetB3
            
            # Resize image
            image = image.resize(target_size)
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array with proper ownership
            img_array = np.array(image)
            
            # Force complete independence from original data
            img_array = np.copy(img_array)
            
            # Ensure proper data type
            img_array = img_array.astype(np.float32)
            
            # Normalize pixel values (adjust based on your model's training)
            img_array = np.divide(img_array, 255.0, dtype=np.float32)
            
            # Add batch dimension - create new array
            img_array = np.expand_dims(img_array, axis=0)
            
            # Final safety check
            if img_array.base is not None:
                logger.warning("Array still has references in MLModel, forcing final copy")
                img_array = np.array(img_array, copy=True)
            
            logger.debug(f"MLModel preprocessing - Array shape: {img_array.shape}, owns data: {img_array.base is None}")
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Make prediction on a single image
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded")
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Process predictions (adjust based on your model's output)
            if len(predictions.shape) == 2 and predictions.shape[1] > 1:
                # Multi-class classification
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                
                # You should replace this with your actual class names
                class_names = self.get_class_names()
                predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Class_{predicted_class_idx}"
            else:
                # Binary classification
                confidence = float(predictions[0][0])
                predicted_class = "Diseased" if confidence > 0.5 else "Healthy"
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "raw_predictions": predictions.tolist()
            }
        
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    def predict_from_array(self, processed_image: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on a preprocessed image array (skip preprocessing)
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded")
        
        try:
            # Skip preprocessing, use the array directly
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Process predictions (same logic as predict method)
            if len(predictions.shape) == 2 and predictions.shape[1] > 1:
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                
                class_names = self.get_class_names()
                predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Class_{predicted_class_idx}"
            else:
                confidence = float(predictions[0][0])
                predicted_class = "Diseased" if confidence > 0.5 else "Healthy"
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "raw_predictions": predictions.tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, images: list) -> list:
        """
        Make predictions on multiple images
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded")
        
        results = []
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for image: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def get_class_names(self) -> list:
        """
        Return class names for your model
        Replace this with your actual class names
        """
        # Example class names for plant diseases
        return [
            "Healthy",
            "Bacterial_spot",
            "Early_blight",
            "Late_blight",
            "Leaf_mold",
            "Septoria_leaf_spot",
            "Spider_mites",
            "Target_spot",
            "Yellow_leaf_curl_virus",
            "Mosaic_virus"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        """
        if not self.is_loaded or self.model is None:
            return {"status": "Model not loaded"}
        
        try:
            return {
                "status": "Model loaded",
                "model_path": self.model_path,
                "input_shape": str(self.model.input_shape) if hasattr(self.model, 'input_shape') else "Unknown",
                "output_shape": str(self.model.output_shape) if hasattr(self.model, 'output_shape') else "Unknown",
                "total_params": self.model.count_params() if hasattr(self.model, 'count_params') else "Unknown",
                "class_names": self.get_class_names()
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"status": "Error getting model info", "error": str(e)}

# Create global model instance
model_instance = MLModel()