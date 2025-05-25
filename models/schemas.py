from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    all_predictions: Dict[str, float] = Field(..., description="Top predictions with confidence scores")
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": "Tomato - Early blight",
                "confidence": 0.9234,
                "all_predictions": {
                    "Tomato - Early blight": 0.9234,
                    "Tomato - Late blight": 0.0456,
                    "Tomato - Leaf Mold": 0.0234,
                    "Tomato - healthy": 0.0056,
                    "Tomato - Bacterial spot": 0.0020
                },
                "message": "Disease detected: Early blight in Tomato. Consider appropriate treatment."
            }
        }

class BatchPredictionItem(BaseModel):
    filename: str
    success: bool
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    top_predictions: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    results: List[BatchPredictionItem]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: Optional[str] = None
    
class ModelInfoResponse(BaseModel):
    model_type: str
    input_shape: List[int]
    num_classes: int
    classes: List[str]
    model_path: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None