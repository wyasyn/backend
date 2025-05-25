from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from typing import List
from models.schemas import PredictionResponse, BatchPredictionResponse, BatchPredictionItem
from services.prediction_service import prediction_service
from api.dependencies import get_model, validate_single_file, validate_batch_files
from utils.logger import logger
from utils.constants import HTTP_MESSAGES

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_model=PredictionResponse)
async def predict_plant_disease(
    file: UploadFile = Depends(validate_single_file),
    model=Depends(get_model)
):
   
    try:
        # Read image data
        image_data = await file.read()
        
        # Make prediction
        result = prediction_service.predict_single_image(image_data)
        
        return PredictionResponse(
            success=True,
            prediction=result["predicted_class"],
            confidence=result["confidence"],
            all_predictions=result["all_predictions"],
            message=result["message"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=HTTP_MESSAGES["PREDICTION_FAILED"].format(error=str(e))
        )

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = Depends(validate_batch_files),
    model=Depends(get_model)
):
   
    try:
        # Read all image data
        images_data = []
        filenames = []
        
        for file in files:
            image_data = await file.read()
            images_data.append(image_data)
            filenames.append(file.filename)
        
        # Make batch predictions
        results = prediction_service.predict_batch_images(images_data)
        
        # Format response
        batch_results = []
        for filename, result in zip(filenames, results):
            batch_item = BatchPredictionItem(filename=filename, **result)
            batch_results.append(batch_item)
        
        return BatchPredictionResponse(results=batch_results)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=HTTP_MESSAGES["PREDICTION_FAILED"].format(error=str(e))
        )