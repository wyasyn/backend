from fastapi import HTTPException, UploadFile, Depends
from typing import List
from models.ml_model import model_instance
from services.image_service import image_service
from config.settings import settings
from utils.constants import HTTP_MESSAGES

def get_model():
   
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail=HTTP_MESSAGES["MODEL_NOT_LOADED"])
    return model_instance

def validate_single_file(file: UploadFile):
   
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=HTTP_MESSAGES["INVALID_FILE_TYPE"])
    
    return file

def validate_batch_files(files: List[UploadFile]):
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=HTTP_MESSAGES["BATCH_SIZE_EXCEEDED"].format(max_size=settings.max_batch_size)
        )
    
    for file in files:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=HTTP_MESSAGES["INVALID_FILE_TYPE"])
    
    return files