"""
Data Management API endpoints
Handles data upload, preprocessing, and session management
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Dict, Any, List
import uuid
import pandas as pd
import io
import base64
from datetime import datetime

from app.models.schemas import (
    DataUploadRequest, DataUploadResponse, DataPreprocessRequest,
    SessionCreateRequest, SessionCreateResponse, SessionInfo,
    ErrorCode, BaseResponse
)
from app.services.data_service import DataService
from app.services.session_service import SessionService
from app.core.logging import get_logger

logger = get_logger("api.data")
router = APIRouter()

# Dependency injection
def get_data_service() -> DataService:
    return DataService()

def get_session_service() -> SessionService:
    return SessionService()


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(
    request: SessionCreateRequest = None,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Create a new data analysis session
    
    Sessions isolate user data and provide a workspace for analysis.
    """
    try:
        if request is None:
            request = SessionCreateRequest()
            
        session_id = str(uuid.uuid4())
        
        session_info = await session_service.create_session(
            session_id=session_id,
            user_id=request.user_id,
            session_name=request.session_name
        )
        
        logger.info(f"Created new session {session_id}")
        
        return SessionCreateResponse(
            session_id=session_id,
            expires_at=session_info['expires_at'],
            message="Session created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to create session"
            }
        )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Get information about a session
    
    Returns session metadata and current status.
    """
    try:
        session_info = await session_service.get_session_info(session_id)
        if not session_info:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Session {session_id} not found"
                }
            )
        
        return SessionInfo(**session_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get session information"
            }
        )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Delete a session and all associated data
    
    This will remove all data, models, and results for the session.
    """
    try:
        success = await session_service.delete_session(session_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Session {session_id} not found"
                }
            )
        
        logger.info(f"Deleted session {session_id}")
        return {"success": True, "message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to delete session"
            }
        )


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    session_id: str = None,
    data_service: DataService = Depends(get_data_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Upload data file to a session
    
    Supports CSV, Excel, and JSON files. Automatically detects data types
    and provides summary information.
    """
    try:
        # Create session if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            await session_service.create_session(session_id)
        
        logger.info(f"Uploading file {file.filename} to session {session_id}")
        
        # Read file content
        content = await file.read()
        
        # Process the uploaded data
        result = await data_service.process_upload(
            session_id=session_id,
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )
        
        logger.info(f"Successfully uploaded {file.filename} with {result['rows']} rows")
        
        return DataUploadResponse(
            session_id=session_id,
            filename=file.filename,
            rows=result['rows'],
            columns=result['columns'],
            column_names=result['column_names'],
            data_types=result['data_types'],
            message="File uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to upload file: {str(e)}"
            }
        )


@router.post("/upload-json", response_model=DataUploadResponse)
async def upload_data_json(
    request: DataUploadRequest,
    data_service: DataService = Depends(get_data_service)
):
    """
    Upload data via JSON request
    
    Alternative to file upload for programmatic data submission.
    """
    try:
        logger.info(f"Uploading JSON data to session {request.session_id}")
        
        # Decode file content
        if request.file_type == "csv":
            content = base64.b64decode(request.file_content)
        else:
            content = request.file_content.encode('utf-8')
        
        # Process the uploaded data
        result = await data_service.process_upload(
            session_id=request.session_id,
            filename=request.filename,
            content=content,
            content_type=f"application/{request.file_type}"
        )
        
        logger.info(f"Successfully uploaded {request.filename} with {result['rows']} rows")
        
        return DataUploadResponse(
            session_id=request.session_id,
            filename=request.filename,
            rows=result['rows'],
            columns=result['columns'],
            column_names=result['column_names'],
            data_types=result['data_types'],
            message="Data uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error uploading JSON data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to upload data: {str(e)}"
            }
        )


@router.get("/sessions/{session_id}/data")
async def get_session_data(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    session_service: SessionService = Depends(get_session_service)
):
    """
    Get data from a session
    
    Returns paginated data with optional filtering and sorting.
    """
    try:
        data = await session_service.get_session_data(session_id)
        if not data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"No data found for session {session_id}"
                }
            )
        
        df = data.get('dataframe')
        if df is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": "No dataframe found in session"
                }
            )
        
        # Apply pagination
        total_rows = len(df)
        paginated_df = df.iloc[offset:offset + limit]
        
        return {
            "data": paginated_df.to_dict('records'),
            "total_rows": total_rows,
            "offset": offset,
            "limit": limit,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get session data"
            }
        )


@router.post("/sessions/{session_id}/preprocess")
async def preprocess_data(
    session_id: str,
    request: DataPreprocessRequest,
    data_service: DataService = Depends(get_data_service)
):
    """
    Apply preprocessing operations to session data
    
    Supports various data cleaning and transformation operations.
    """
    try:
        logger.info(f"Preprocessing data for session {session_id}")
        
        result = await data_service.preprocess_data(
            session_id=session_id,
            operations=request.operations
        )
        
        logger.info(f"Preprocessing completed for session {session_id}")
        
        return {
            "success": True,
            "message": "Data preprocessing completed",
            "operations_applied": len(request.operations),
            "rows_affected": result.get('rows_affected', 0),
            "summary": result.get('summary', {})
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to preprocess data: {str(e)}"
            }
        )


@router.get("/sessions/{session_id}/summary")
async def get_data_summary(
    session_id: str,
    data_service: DataService = Depends(get_data_service)
):
    """
    Get statistical summary of session data
    
    Returns descriptive statistics and data quality information.
    """
    try:
        summary = await data_service.get_data_summary(session_id)
        if not summary:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"No data found for session {session_id}"
                }
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get data summary"
            }
        )


@router.get("/demo-datasets")
async def list_demo_datasets():
    """
    List available demo datasets
    
    Returns information about built-in datasets for testing and learning.
    """
    try:
        datasets = [
            {
                "id": "breast_cancer",
                "name": "Breast Cancer Wisconsin",
                "description": "Breast cancer diagnostic data",
                "rows": 569,
                "columns": 32,
                "target": "diagnosis"
            },
            {
                "id": "diabetes",
                "name": "Diabetes Prediction",
                "description": "Diabetes prediction dataset",
                "rows": 768,
                "columns": 9,
                "target": "Outcome"
            },
            {
                "id": "stroke",
                "name": "Stroke Prediction",
                "description": "Healthcare stroke prediction data",
                "rows": 5110,
                "columns": 12,
                "target": "stroke"
            }
        ]
        
        return {"datasets": datasets}
        
    except Exception as e:
        logger.error(f"Error listing demo datasets: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to list demo datasets"
            }
        )


@router.post("/demo-datasets/{dataset_id}/load")
async def load_demo_dataset(
    dataset_id: str,
    session_id: str = None,
    data_service: DataService = Depends(get_data_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Load a demo dataset into a session
    
    Convenient way to get started with analysis using built-in datasets.
    """
    try:
        # Create session if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            await session_service.create_session(session_id)
        
        logger.info(f"Loading demo dataset {dataset_id} into session {session_id}")
        
        result = await data_service.load_demo_dataset(session_id, dataset_id)
        
        logger.info(f"Successfully loaded demo dataset {dataset_id}")
        
        return DataUploadResponse(
            session_id=session_id,
            filename=f"{dataset_id}.csv",
            rows=result['rows'],
            columns=result['columns'],
            column_names=result['column_names'],
            data_types=result['data_types'],
            message=f"Demo dataset {dataset_id} loaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error loading demo dataset: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to load demo dataset: {str(e)}"
            }
        )
