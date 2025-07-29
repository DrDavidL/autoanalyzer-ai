"""
Configuration settings for AutoAnalyzer AI Backend
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"],
        env="ALLOWED_ORIGINS"
    )
    
    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    
    # Celery configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Database configuration
    database_url: str = Field(default="sqlite:///./autoanalyzer.db", env="DATABASE_URL")
    
    # API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(default="", env="AZURE_OPENAI_API_KEY")
    
    # Task configuration
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")  # 5 minutes
    max_task_retries: int = Field(default=3, env="MAX_TASK_RETRIES")
    
    # File storage
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    results_dir: str = Field(default="./results", env="RESULTS_DIR")
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    
    # Session configuration
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")  # 1 hour
    max_sessions_per_user: int = Field(default=5, env="MAX_SESSIONS_PER_USER")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # ML Configuration
    ml_model_cache_size: int = Field(default=10, env="ML_MODEL_CACHE_SIZE")
    shap_sample_size: int = Field(default=100, env="SHAP_SAMPLE_SIZE")
    cv_folds: int = Field(default=5, env="CV_FOLDS")
    
    # GPT Configuration
    gpt_model: str = Field(default="gpt-4", env="GPT_MODEL")
    gpt_temperature: float = Field(default=0.1, env="GPT_TEMPERATURE")
    gpt_max_tokens: int = Field(default=2000, env="GPT_MAX_TOKENS")
    gpt_timeout: int = Field(default=60, env="GPT_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


# Ensure required directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.results_dir, exist_ok=True)
