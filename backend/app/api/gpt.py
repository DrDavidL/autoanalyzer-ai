"""
GPT Analysis API endpoints
Handles AI-powered code generation and analysis
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import uuid
import asyncio

from app.models.schemas import (
    GPTAnalysisRequest, GPTAnalysisResponse, GPTAnalysisResult,
    JobStatusResponse, ErrorCode
)
from app.services.gpt_service import GPTService
from app.services.session_service import SessionService
from app.core.logging import get_logger

logger = get_logger("api.gpt")
router = APIRouter()

# Dependency injection
def get_gpt_service() -> GPTService:
    return GPTService()

def get_session_service() -> SessionService:
    return SessionService()


@router.post("/analyze", response_model=GPTAnalysisResponse)
async def start_gpt_analysis(
    request: GPTAnalysisRequest,
    gpt_service: GPTService = Depends(get_gpt_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Start GPT-powered data analysis
    
    This endpoint queues a GPT analysis job that generates and executes
    Python code based on natural language questions about the data.
    """
    try:
        logger.info(f"Starting GPT analysis for session {request.session_id}")
        
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
        
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": "No data found in session"
                }
            )
        
        # Generate unique IDs
        job_id = str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())
        
        # Queue the GPT analysis job
        await gpt_service.analyze_async(
            job_id=job_id,
            analysis_id=analysis_id,
            request=request
        )
        
        logger.info(f"GPT analysis job {job_id} queued for analysis {analysis_id}")
        
        return GPTAnalysisResponse(
            job_id=job_id,
            analysis_id=analysis_id,
            status="queued",
            estimated_duration=120,  # 2 minutes estimate
            message="GPT analysis job queued successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting GPT analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to start GPT analysis"
            }
        )


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_analysis_status(
    job_id: str,
    gpt_service: GPTService = Depends(get_gpt_service)
):
    """
    Get the status of a GPT analysis job
    
    Returns current progress, status, and intermediate results if available.
    """
    try:
        logger.debug(f"Getting status for GPT analysis job {job_id}")
        
        status = await gpt_service.get_job_status(job_id)
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
        logger.error(f"Error getting analysis status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get analysis status"
            }
        )


@router.get("/jobs/{job_id}/results", response_model=GPTAnalysisResult)
async def get_analysis_results(
    job_id: str,
    gpt_service: GPTService = Depends(get_gpt_service)
):
    """
    Get the results of a completed GPT analysis job
    
    Returns the complete analysis including generated code, outputs,
    visualizations, and summary.
    """
    try:
        logger.debug(f"Getting results for GPT analysis job {job_id}")
        
        results = await gpt_service.get_job_results(job_id)
        if not results:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Results for job {job_id} not found"
                }
            )
        
        return GPTAnalysisResult(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to get analysis results"
            }
        )


@router.post("/refine")
async def refine_analysis(
    job_id: str,
    refinement_request: str,
    gpt_service: GPTService = Depends(get_gpt_service)
):
    """
    Refine an existing GPT analysis
    
    Allows iterative improvement of the analysis based on user feedback.
    """
    try:
        logger.info(f"Refining GPT analysis job {job_id}")
        
        # Create new job for refinement
        new_job_id = str(uuid.uuid4())
        
        # Queue the refinement job
        await gpt_service.refine_analysis_async(
            original_job_id=job_id,
            new_job_id=new_job_id,
            refinement_request=refinement_request
        )
        
        logger.info(f"GPT refinement job {new_job_id} queued")
        
        return {
            "job_id": new_job_id,
            "status": "queued",
            "message": "Analysis refinement job queued successfully"
        }
        
    except Exception as e:
        logger.error(f"Error refining analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to refine analysis"
            }
        )


@router.post("/execute-code")
async def execute_code(
    session_id: str,
    code: str,
    gpt_service: GPTService = Depends(get_gpt_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Execute Python code in a sandboxed environment
    
    Allows direct code execution with access to session data.
    """
    try:
        logger.info(f"Executing code for session {session_id}")
        
        # Validate session exists and has data
        session_data = await session_service.get_session_data(session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Session {session_id} not found"
                }
            )
        
        # Execute the code
        result = await gpt_service.execute_code(
            session_id=session_id,
            code=code
        )
        
        logger.info(f"Code execution completed for session {session_id}")
        
        return {
            "success": True,
            "output": result.get('output', ''),
            "error": result.get('error'),
            "plots": result.get('plots', []),
            "execution_time": result.get('execution_time', 0),
            "message": "Code executed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing code: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to execute code: {str(e)}"
            }
        )


@router.post("/generate-code")
async def generate_code(
    session_id: str,
    question: str,
    context: str = None,
    gpt_service: GPTService = Depends(get_gpt_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Generate Python code based on natural language question
    
    Returns code without executing it, allowing review before execution.
    """
    try:
        logger.info(f"Generating code for session {session_id}")
        
        # Validate session exists and has data
        session_data = await session_service.get_session_data(session_id)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Session {session_id} not found"
                }
            )
        
        # Generate code
        result = await gpt_service.generate_code(
            session_id=session_id,
            question=question,
            context=context
        )
        
        logger.info(f"Code generation completed for session {session_id}")
        
        return {
            "success": True,
            "code": result.get('code', ''),
            "explanation": result.get('explanation', ''),
            "confidence": result.get('confidence', 0.0),
            "message": "Code generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating code: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to generate code: {str(e)}"
            }
        )


@router.post("/export/{job_id}")
async def export_analysis(
    job_id: str,
    format: str = "docx",
    gpt_service: GPTService = Depends(get_gpt_service)
):
    """
    Export GPT analysis results to document format
    
    Supports Word (DOCX) and PDF export formats.
    """
    try:
        logger.info(f"Exporting analysis {job_id} to {format}")
        
        if format not in ["docx", "pdf"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR,
                    "message": "Invalid export format. Must be 'docx' or 'pdf'"
                }
            )
        
        # Export the analysis
        result = await gpt_service.export_analysis(job_id, format)
        if not result:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Analysis {job_id} not found"
                }
            )
        
        logger.info(f"Analysis {job_id} exported to {format}")
        
        return {
            "success": True,
            "export_url": result.get('export_url'),
            "filename": result.get('filename'),
            "format": format,
            "message": f"Analysis exported to {format} successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to export analysis: {str(e)}"
            }
        )


@router.get("/templates")
async def list_analysis_templates():
    """
    List available analysis templates
    
    Returns pre-defined analysis templates for common use cases.
    """
    try:
        templates = [
            {
                "id": "exploratory_data_analysis",
                "name": "Exploratory Data Analysis",
                "description": "Comprehensive EDA with summary statistics and visualizations",
                "questions": [
                    "What are the basic statistics of this dataset?",
                    "Are there any missing values or outliers?",
                    "What are the distributions of key variables?",
                    "Are there any interesting correlations?"
                ]
            },
            {
                "id": "predictive_modeling",
                "name": "Predictive Modeling",
                "description": "Build and evaluate machine learning models",
                "questions": [
                    "What is the best model for predicting [target variable]?",
                    "Which features are most important for prediction?",
                    "How well does the model perform on test data?",
                    "What are the model's limitations?"
                ]
            },
            {
                "id": "statistical_testing",
                "name": "Statistical Testing",
                "description": "Hypothesis testing and statistical analysis",
                "questions": [
                    "Is there a significant difference between groups?",
                    "Are these variables correlated?",
                    "What factors influence the outcome?",
                    "Are the assumptions for this test met?"
                ]
            },
            {
                "id": "time_series_analysis",
                "name": "Time Series Analysis",
                "description": "Analyze temporal patterns and trends",
                "questions": [
                    "What are the trends and seasonal patterns?",
                    "Can we forecast future values?",
                    "Are there any anomalies or outliers?",
                    "What drives the changes over time?"
                ]
            },
            {
                "id": "clinical_research",
                "name": "Clinical Research Analysis",
                "description": "Medical and healthcare data analysis",
                "questions": [
                    "What are the patient demographics?",
                    "Are there differences in outcomes between treatments?",
                    "What are the risk factors for this condition?",
                    "How effective is this intervention?"
                ]
            }
        ]
        
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Error listing templates: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to list analysis templates"
            }
        )


@router.post("/templates/{template_id}/apply")
async def apply_template(
    template_id: str,
    session_id: str,
    customizations: Dict[str, Any] = None,
    gpt_service: GPTService = Depends(get_gpt_service)
):
    """
    Apply an analysis template to session data
    
    Automatically generates analysis based on template questions and data.
    """
    try:
        logger.info(f"Applying template {template_id} to session {session_id}")
        
        # Generate unique IDs
        job_id = str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())
        
        # Queue the template analysis job
        await gpt_service.apply_template_async(
            job_id=job_id,
            analysis_id=analysis_id,
            template_id=template_id,
            session_id=session_id,
            customizations=customizations or {}
        )
        
        logger.info(f"Template analysis job {job_id} queued")
        
        return {
            "job_id": job_id,
            "analysis_id": analysis_id,
            "status": "queued",
            "template_id": template_id,
            "estimated_duration": 300,  # 5 minutes for template analysis
            "message": f"Template {template_id} analysis queued successfully"
        }
        
    except Exception as e:
        logger.error(f"Error applying template: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to apply template: {str(e)}"
            }
        )


@router.get("/analyses")
async def list_analyses(
    session_id: str = None,
    gpt_service: GPTService = Depends(get_gpt_service)
):
    """
    List GPT analyses
    
    Optionally filter by session ID to get analyses for a specific session.
    """
    try:
        logger.debug(f"Listing analyses for session {session_id}")
        
        analyses = await gpt_service.list_analyses(session_id)
        return {"analyses": analyses}
        
    except Exception as e:
        logger.error(f"Error listing analyses: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to list analyses"
            }
        )


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    gpt_service: GPTService = Depends(get_gpt_service)
):
    """
    Delete a GPT analysis and its associated data
    
    This will remove the analysis results and any exported documents.
    """
    try:
        logger.info(f"Deleting analysis {analysis_id}")
        
        success = await gpt_service.delete_analysis(analysis_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": ErrorCode.SESSION_NOT_FOUND,
                    "message": f"Analysis {analysis_id} not found"
                }
            )
        
        return {"success": True, "message": f"Analysis {analysis_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to delete analysis"
            }
        )
