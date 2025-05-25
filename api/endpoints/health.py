from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
from models.schemas import HealthResponse, ModelInfoResponse
from models.ml_model import model_instance
from api.dependencies import get_model
from utils.constants import HTTP_MESSAGES
from utils.logger import logger

router = APIRouter(prefix="/health", tags=["Health"])

# Store startup time
startup_time = datetime.now()

@router.get("/", response_model=HealthResponse)
async def health_check():
    
    uptime = str(datetime.now() - startup_time).split('.')[0]  # Remove microseconds
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_instance.is_loaded,
        uptime=uptime
    )

@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info(model=Depends(get_model)):
    
    try:
        info = model.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.post("/load-model")
async def load_model_manually():
    
    try:
        success = model_instance.load_model()
        if success:
            return {"message": HTTP_MESSAGES["MODEL_LOAD_SUCCESS"]}
        else:
            raise HTTPException(status_code=500, detail=HTTP_MESSAGES["MODEL_LOAD_FAILED"])
    except Exception as e:
        logger.error(f"Manual model loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")