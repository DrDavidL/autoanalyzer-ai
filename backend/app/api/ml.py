"""
Machine Learning API endpoints
High priority component for async ML training and model management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any
import uuid
import asyncio
from datetime import datetime

from app.models.schemas import (
    MLTrainRequest, MLTrainResponse, MLPredictRequest, MLPredictResponse,
    MLExplainRequest, JobStatusResponse, ErrorResponse, ErrorCode
)
from app.services.ml_service import MLService
from app.services.session_service import SessionService
from app.core.logging import get_logger

logger = get_logger("api.ml")
router = APIRouter()

# Dependency injection
def get_ml_service() -> MLService:
    return MLService()

def get_session_service() -> SessionService:
    return SessionService()


@router.post("/train", response_model=MLTrainResponse)
async def train_model(
    request: MLTrainRequest,
    ml_service: MLService = Depends(get_ml_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Start asynchronous machine learning model training
    
    This endpoint queues a model training job and returns immediately with a job ID.
    The actual training happens in the background using Celery workers.
    """
    try:
        logger.info(f"Starting ML training for session {request.session_id}")
        
        # Validate session exists and has data
        session_data = await session_service.get_session_data(request.session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Session {request.session_id} not found"
                }
            )
        
        # Validate target and feature columns exist in data
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": "No data found in session"
                }
            )
        
        available_columns = list(df.columns)
        if request.target_column not in available_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": f"Target column '{request.target_column}' not found in data"
                }
            )
        
        missing_features = [col for col in request.feature_columns if col not in available_columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": f"Feature columns not found: {missing_features}"
                }
            )
        
        # Generate unique IDs
        job_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        # Queue the training job
        await ml_service.train_model_async(
            job_id=job_id,
            model_id=model_id,
            request=request
        )
        
        logger.info(f"ML training job {job_id} queued for model {model_id}")
        
        return MLTrainResponse(
            job_id=job_id,
            model_id=model_id,
            status="queued",
            estimated_duration=180,  # 3 minutes estimate
            message="Model training job queued successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting ML training: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to start model training"
            }
        )


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Get the status of a machine learning training job
    
    Returns current progress, status, and results if completed.
    """
    try:
        logger.debug(f"Getting status for ML job {job_id}")
        
        status = await ml_service.get_job_status(job_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Job {job_id} not found"
                }
            )
        
        return JobStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get job status"
            }
        )


@router.get("/jobs/{job_id}/results")
async def get_job_results(
    job_id: str,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Get the results of a completed machine learning training job
    
    Returns detailed training results, metrics, and model information.
    """
    try:
        logger.debug(f"Getting results for ML job {job_id}")
        
        results = await ml_service.get_job_results(job_id)
        if not results:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Results for job {job_id} not found"
                }
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get job results"
            }
        )


@router.post("/predict", response_model=MLPredictResponse)
async def predict(
    request: MLPredictRequest,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Make predictions using a trained model
    
    Returns prediction and confidence scores for the input data.
    """
    try:
        logger.debug(f"Making prediction with model {request.model_id}")
        
        result = await ml_service.predict(request.model_id, request.data)
        if not result:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Model {request.model_id} not found"
                }
            )
        
        return MLPredictResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to make prediction"
            }
        )


@router.post("/explain", response_model=JobStatusResponse)
async def explain_model(
    request: MLExplainRequest,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Generate SHAP explanations for a trained model
    
    This is a long-running operation that generates model interpretability
    analysis using SHAP values.
    """
    try:
        logger.info(f"Starting SHAP explanation for model {request.model_id}")
        
        job_id = str(uuid.uuid4())
        
        # Queue the explanation job
        await ml_service.explain_model_async(
            job_id=job_id,
            request=request
        )
        
        logger.info(f"SHAP explanation job {job_id} queued")
        
        return JobStatusResponse(
            job_id=job_id,
            status="queued",
            progress=0.0,
            message="SHAP explanation job queued successfully"
        )
        
    except Exception as e:
        logger.error(f"Error starting model explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to start model explanation"
            }
        )


@router.get("/models/{model_id}")
async def get_model_info(
    model_id: str,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Get information about a trained model
    
    Returns model metadata, performance metrics, and training details.
    """
    try:
        logger.debug(f"Getting info for model {model_id}")
        
        model_info = await ml_service.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Model {model_id} not found"
                }
            )
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get model information"
            }
        )


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Delete a trained model and its associated data
    
    This will remove the model from storage and cache.
    """
    try:
        logger.info(f"Deleting model {model_id}")
        
        success = await ml_service.delete_model(model_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Model {model_id} not found"
                }
            )
        
        return {"success": True, "message": f"Model {model_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to delete model"
            }
        )


@router.get("/models")
async def list_models(
    session_id: str = None,
    ml_service: MLService = Depends(get_ml_service)
):
    """
    List available trained models
    
    Optionally filter by session ID to get models for a specific session.
    """
    try:
        logger.debug(f"Listing models for session {session_id}")
        
        models = await ml_service.list_models(session_id)
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to list models"
            }
        )
