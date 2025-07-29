"""
AutoAnalyzer AI - FastAPI Backend
Main application entry point for the API backend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api import ml, stats, gpt, data

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting AutoAnalyzer AI Backend")
    yield
    logger.info("Shutting down AutoAnalyzer AI Backend")


# Create FastAPI application
app = FastAPI(
    title="AutoAnalyzer AI Backend",
    description="High-performance backend for AutoAnalyzer AI data analysis platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ml.router, prefix="/api/ml", tags=["machine-learning"])
app.include_router(stats.router, prefix="/api/stats", tags=["statistics"])
app.include_router(gpt.router, prefix="/api/gpt", tags=["gpt-analysis"])
app.include_router(data.router, prefix="/api/data", tags=["data-management"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AutoAnalyzer AI Backend",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": "2025-01-24T15:26:00Z"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "internal_server_error",
            "message": "An internal server error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
