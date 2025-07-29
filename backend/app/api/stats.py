"""
Statistical Analysis API endpoints
Handles statistical tests and analysis operations
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import uuid
import asyncio

from app.models.schemas import (
    StatTestRequest, StatTestResponse, StatTestType,
    JobStatusResponse, ErrorCode
)
from app.services.stats_service import StatsService
from app.services.session_service import SessionService
from app.core.logging import get_logger

logger = get_logger("api.stats")
router = APIRouter()

# Dependency injection
def get_stats_service() -> StatsService:
    return StatsService()

def get_session_service() -> SessionService:
    return SessionService()


@router.post("/test", response_model=StatTestResponse)
async def run_statistical_test(
    request: StatTestRequest,
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Run a statistical test on session data
    
    Supports various statistical tests including t-tests, ANOVA, chi-square,
    correlation analysis, and regression.
    """
    try:
        logger.info(f"Running {request.test_type} test for session {request.session_id}")
        
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
        
        # Validate columns exist
        available_columns = list(df.columns)
        missing_columns = [col for col in request.columns if col not in available_columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": f"Columns not found: {missing_columns}"
                }
            )
        
        # Run the statistical test
        result = await stats_service.run_test(
            test_type=request.test_type,
            data=df,
            columns=request.columns,
            config=request.config
        )
        
        logger.info(f"Completed {request.test_type} test with p-value {result.get('p_value')}")
        
        return StatTestResponse(
            test_type=request.test_type,
            statistic=result['statistic'],
            p_value=result['p_value'],
            confidence_interval=result.get('confidence_interval'),
            effect_size=result.get('effect_size'),
            interpretation=result['interpretation'],
            plot_url=result.get('plot_url'),
            message="Statistical test completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running statistical test: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to run statistical test: {str(e)}"
            }
        )


@router.post("/ttest")
async def run_ttest(
    session_id: str,
    column1: str,
    column2: str = None,
    test_type: str = "two_sample",
    alpha: float = 0.05,
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Run t-test analysis
    
    Supports one-sample, two-sample, and paired t-tests.
    """
    try:
        # Map test type to enum
        test_type_map = {
            "one_sample": StatTestType.TTEST_ONE_SAMPLE,
            "two_sample": StatTestType.TTEST_TWO_SAMPLE,
            "paired": StatTestType.TTEST_PAIRED
        }
        
        if test_type not in test_type_map:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR,
                    "message": f"Invalid test type. Must be one of: {list(test_type_map.keys())}"
                }
            )
        
        columns = [column1]
        if column2:
            columns.append(column2)
        
        request = StatTestRequest(
            session_id=session_id,
            test_type=test_type_map[test_type],
            columns=columns,
            config={"alpha": alpha}
        )
        
        return await run_statistical_test(request, stats_service, session_service)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running t-test: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to run t-test: {str(e)}"
            }
        )


@router.post("/anova")
async def run_anova(
    session_id: str,
    dependent_var: str,
    independent_vars: List[str],
    anova_type: str = "one_way",
    alpha: float = 0.05,
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Run ANOVA analysis
    
    Supports one-way and two-way ANOVA with post-hoc tests.
    """
    try:
        # Map ANOVA type to enum
        anova_type_map = {
            "one_way": StatTestType.ANOVA_ONE_WAY,
            "two_way": StatTestType.ANOVA_TWO_WAY
        }
        
        if anova_type not in anova_type_map:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR,
                    "message": f"Invalid ANOVA type. Must be one of: {list(anova_type_map.keys())}"
                }
            )
        
        columns = [dependent_var] + independent_vars
        
        request = StatTestRequest(
            session_id=session_id,
            test_type=anova_type_map[anova_type],
            columns=columns,
            config={
                "dependent_var": dependent_var,
                "independent_vars": independent_vars,
                "alpha": alpha
            }
        )
        
        return await run_statistical_test(request, stats_service, session_service)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running ANOVA: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to run ANOVA: {str(e)}"
            }
        )


@router.post("/chi-square")
async def run_chi_square(
    session_id: str,
    column1: str,
    column2: str,
    alpha: float = 0.05,
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Run chi-square test of independence
    
    Tests association between two categorical variables.
    """
    try:
        request = StatTestRequest(
            session_id=session_id,
            test_type=StatTestType.CHI_SQUARE,
            columns=[column1, column2],
            config={"alpha": alpha}
        )
        
        return await run_statistical_test(request, stats_service, session_service)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running chi-square test: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to run chi-square test: {str(e)}"
            }
        )


@router.post("/correlation")
async def run_correlation(
    session_id: str,
    columns: List[str],
    method: str = "pearson",
    alpha: float = 0.05,
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Run correlation analysis
    
    Supports Pearson, Spearman, and Kendall correlation methods.
    """
    try:
        if method not in ["pearson", "spearman", "kendall"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR,
                    "message": "Invalid correlation method. Must be one of: pearson, spearman, kendall"
                }
            )
        
        request = StatTestRequest(
            session_id=session_id,
            test_type=StatTestType.CORRELATION,
            columns=columns,
            config={"method": method, "alpha": alpha}
        )
        
        return await run_statistical_test(request, stats_service, session_service)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running correlation analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to run correlation analysis: {str(e)}"
            }
        )


@router.post("/regression")
async def run_regression(
    session_id: str,
    dependent_var: str,
    independent_vars: List[str],
    regression_type: str = "linear",
    alpha: float = 0.05,
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Run regression analysis
    
    Supports linear and logistic regression with model diagnostics.
    """
    try:
        # Map regression type to enum
        regression_type_map = {
            "linear": StatTestType.REGRESSION_LINEAR,
            "logistic": StatTestType.REGRESSION_LOGISTIC
        }
        
        if regression_type not in regression_type_map:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR,
                    "message": f"Invalid regression type. Must be one of: {list(regression_type_map.keys())}"
                }
            )
        
        columns = [dependent_var] + independent_vars
        
        request = StatTestRequest(
            session_id=session_id,
            test_type=regression_type_map[regression_type],
            columns=columns,
            config={
                "dependent_var": dependent_var,
                "independent_vars": independent_vars,
                "alpha": alpha
            }
        )
        
        return await run_statistical_test(request, stats_service, session_service)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running regression: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to run regression: {str(e)}"
            }
        )


@router.get("/sessions/{session_id}/descriptive")
async def get_descriptive_stats(
    session_id: str,
    columns: List[str] = None,
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Get descriptive statistics for session data
    
    Returns mean, median, std, quartiles, and other summary statistics.
    """
    try:
        logger.info(f"Getting descriptive stats for session {session_id}")
        
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
        
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": "No data found in session"
                }
            )
        
        # Get descriptive statistics
        result = await stats_service.get_descriptive_stats(df, columns)
        
        logger.info(f"Generated descriptive stats for {len(result)} columns")
        
        return {
            "success": True,
            "statistics": result,
            "message": "Descriptive statistics generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting descriptive stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to get descriptive statistics: {str(e)}"
            }
        )


@router.post("/survival-analysis")
async def run_survival_analysis(
    session_id: str,
    duration_col: str,
    event_col: str,
    covariates: List[str] = None,
    analysis_type: str = "kaplan_meier",
    stats_service: StatsService = Depends(get_stats_service),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Run survival analysis
    
    Supports Kaplan-Meier estimation and Cox proportional hazards regression.
    """
    try:
        logger.info(f"Running {analysis_type} survival analysis for session {session_id}")
        
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
        
        df = session_data.get('dataframe')
        if df is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": "No data found in session"
                }
            )
        
        # Validate required columns
        required_cols = [duration_col, event_col]
        if covariates:
            required_cols.extend(covariates)
        
        available_columns = list(df.columns)
        missing_columns = [col for col in required_cols if col not in available_columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": ErrorCode.INVALID_DATA,
                    "message": f"Columns not found: {missing_columns}"
                }
            )
        
        # Run survival analysis
        result = await stats_service.run_survival_analysis(
            data=df,
            duration_col=duration_col,
            event_col=event_col,
            covariates=covariates,
            analysis_type=analysis_type
        )
        
        logger.info(f"Completed {analysis_type} survival analysis")
        
        return {
            "success": True,
            "analysis_type": analysis_type,
            "results": result,
            "message": f"{analysis_type} survival analysis completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running survival analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": f"Failed to run survival analysis: {str(e)}"
            }
        )


@router.get("/test-types")
async def list_test_types():
    """
    List available statistical test types
    
    Returns information about supported statistical tests and their parameters.
    """
    try:
        test_types = {
            "t_tests": {
                "one_sample": {
                    "description": "One-sample t-test",
                    "parameters": ["column", "mu", "alpha"],
                    "use_case": "Test if sample mean differs from population mean"
                },
                "two_sample": {
                    "description": "Two-sample t-test",
                    "parameters": ["column1", "column2", "alpha", "equal_var"],
                    "use_case": "Compare means of two independent groups"
                },
                "paired": {
                    "description": "Paired t-test",
                    "parameters": ["column1", "column2", "alpha"],
                    "use_case": "Compare means of paired observations"
                }
            },
            "anova": {
                "one_way": {
                    "description": "One-way ANOVA",
                    "parameters": ["dependent_var", "independent_var", "alpha"],
                    "use_case": "Compare means across multiple groups"
                },
                "two_way": {
                    "description": "Two-way ANOVA",
                    "parameters": ["dependent_var", "factor1", "factor2", "alpha"],
                    "use_case": "Analyze effects of two factors on dependent variable"
                }
            },
            "non_parametric": {
                "chi_square": {
                    "description": "Chi-square test of independence",
                    "parameters": ["column1", "column2", "alpha"],
                    "use_case": "Test association between categorical variables"
                }
            },
            "correlation": {
                "pearson": {
                    "description": "Pearson correlation",
                    "parameters": ["columns", "alpha"],
                    "use_case": "Linear relationship between continuous variables"
                },
                "spearman": {
                    "description": "Spearman correlation",
                    "parameters": ["columns", "alpha"],
                    "use_case": "Monotonic relationship between variables"
                }
            },
            "regression": {
                "linear": {
                    "description": "Linear regression",
                    "parameters": ["dependent_var", "independent_vars", "alpha"],
                    "use_case": "Model linear relationship between variables"
                },
                "logistic": {
                    "description": "Logistic regression",
                    "parameters": ["dependent_var", "independent_vars", "alpha"],
                    "use_case": "Model binary outcomes"
                }
            },
            "survival": {
                "kaplan_meier": {
                    "description": "Kaplan-Meier survival estimation",
                    "parameters": ["duration_col", "event_col", "groupby"],
                    "use_case": "Estimate survival probabilities over time"
                },
                "cox_regression": {
                    "description": "Cox proportional hazards regression",
                    "parameters": ["duration_col", "event_col", "covariates"],
                    "use_case": "Model hazard ratios with covariates"
                }
            }
        }
        
        return {"test_types": test_types}
        
    except Exception as e:
        logger.error(f"Error listing test types: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "Failed to list test types"
            }
        )
