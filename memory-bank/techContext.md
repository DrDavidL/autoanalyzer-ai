# AutoAnalyzer AI - Technical Context

## Current Technology Stack

### Core Framework
- **Streamlit 1.28+**: Web application framework
- **Python 3.9+**: Primary programming language
- **Docker**: Containerization for deployment

### Data Processing & Analytics
- **pandas 2.0+**: Data manipulation and analysis
- **numpy 1.24+**: Numerical computing
- **scipy 1.10+**: Scientific computing and statistics
- **scikit-learn 1.3+**: Machine learning algorithms
- **xgboost 1.7+**: Gradient boosting framework
- **lifelines 0.27+**: Survival analysis

### Visualization
- **matplotlib 3.7+**: Core plotting library
- **seaborn 0.12+**: Statistical data visualization
- **plotly 5.15+**: Interactive visualizations
- **sweetviz 2.1+**: Automated EDA reports
- **ydata-profiling 4.3+**: Comprehensive data profiling

### Machine Learning & AI
- **shap 0.42+**: Model interpretability
- **langchain 0.0.300+**: LLM integration framework
- **openai 0.28+**: GPT API integration
- **azure-openai**: Azure OpenAI services

### Statistical Analysis
- **statsmodels 0.14+**: Advanced statistical modeling
- **tableone 0.8+**: Clinical table generation
- **missingno 0.5+**: Missing data visualization

### Document Generation
- **python-docx 0.8+**: Word document creation
- **openpyxl 3.1+**: Excel file handling

## Development Environment

### Local Development
```bash
# Python environment
python 3.9+
pip install -r requirements.txt

# Environment variables
OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
PASSWORD=your_password
```

### Docker Configuration
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
```

### Dependencies Management
- **requirements.txt**: Production dependencies
- **Dockerfile**: Container configuration
- **No virtual environment**: Direct pip installation

## Current Technical Constraints

### Performance Limitations
- **Single Process**: Streamlit runs in single process
- **GIL Constraints**: Python Global Interpreter Lock limits concurrency
- **Memory Usage**: Each session loads full libraries (~500MB)
- **Blocking Operations**: ML training blocks entire application
- **No Caching**: Repeated computations not cached

### Scalability Issues
- **Session State**: In-memory only, lost on restart
- **File Storage**: Local filesystem only
- **No Load Balancing**: Single instance deployment
- **Resource Contention**: Shared CPU/memory across users

### Security Considerations
- **Code Execution**: GPT generates and executes arbitrary Python code
- **Input Validation**: Limited validation on user inputs
- **Authentication**: Simple password-based access
- **Data Isolation**: No user data separation

## Target Technology Stack

### Backend Framework
- **FastAPI 0.100+**: High-performance async web framework
- **uvicorn 0.23+**: ASGI server for FastAPI
- **pydantic 2.0+**: Data validation and serialization

### Task Queue & Caching
- **Celery 5.3+**: Distributed task queue
- **Redis 7.0+**: In-memory cache and message broker
- **SQLite/PostgreSQL**: Metadata and job status storage

### API & Communication
- **httpx 0.24+**: Async HTTP client for Streamlit→FastAPI
- **websockets 11.0+**: Real-time progress updates
- **Server-Sent Events**: Alternative to WebSockets

### Monitoring & Logging
- **structlog 23.1+**: Structured logging
- **prometheus-client 0.17+**: Metrics collection
- **health checks**: Built-in FastAPI health endpoints

## Migration Technology Decisions

### Why FastAPI?
1. **Performance**: 2-3x faster than Flask/Django
2. **Async Support**: Native async/await for non-blocking operations
3. **Type Safety**: Pydantic integration for request/response validation
4. **Documentation**: Auto-generated OpenAPI/Swagger docs
5. **Ecosystem**: Rich ecosystem for ML/data science

### Why Redis?
1. **Speed**: In-memory performance for caching
2. **Pub/Sub**: Message passing for real-time updates
3. **TTL Support**: Automatic cleanup of expired data
4. **Clustering**: Horizontal scaling capability
5. **Python Integration**: Excellent asyncio support

### Why Celery?
1. **Maturity**: Battle-tested distributed task queue
2. **Flexibility**: Multiple broker and result backend options
3. **Monitoring**: Built-in monitoring and management tools
4. **Scaling**: Easy horizontal worker scaling
5. **Error Handling**: Robust retry and error handling

## Development Setup Changes

### New Project Structure
```
autoanalyzer-ai/
├── frontend/                 # Streamlit application
│   ├── main.py
│   ├── ui.py
│   └── api_client.py        # New: API communication
├── backend/                 # New: FastAPI application
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── services/
│   │   ├── models/
│   │   └── workers/
│   └── requirements.txt
├── shared/                  # Shared utilities
│   ├── data_processing.py
│   ├── stats.py
│   ├── plotting.py
│   └── ml.py
├── docker-compose.yml       # Multi-service deployment
└── requirements.txt         # Frontend dependencies
```

### Docker Compose Configuration
```yaml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
      - redis
    
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - redis
    
  worker:
    build: ./backend
    command: celery -A app.workers worker --loglevel=info
    depends_on:
      - redis
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## API Design Standards

### Request/Response Models
```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class MLTrainRequest(BaseModel):
    session_id: str
    target_column: str
    feature_columns: List[str]
    model_type: str
    config: Dict[str, Any]
    
class MLTrainResponse(BaseModel):
    job_id: str
    status: str
    estimated_duration: Optional[int]

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: Optional[str]
    result: Optional[Dict[str, Any]]
```

### Error Handling Standards
```python
from fastapi import HTTPException
from enum import Enum

class ErrorCode(str, Enum):
    INVALID_DATA = "invalid_data"
    MODEL_TRAINING_FAILED = "model_training_failed"
    SESSION_NOT_FOUND = "session_not_found"
    RESOURCE_EXHAUSTED = "resource_exhausted"

class APIError(HTTPException):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail={
            "error_code": code,
            "message": message
        })
```

## Testing Strategy

### Unit Testing
- **pytest 7.4+**: Test framework
- **pytest-asyncio 0.21+**: Async test support
- **httpx**: API client testing
- **Coverage**: Code coverage reporting

### Integration Testing
- **testcontainers 3.7+**: Docker-based integration tests
- **Redis testing**: In-memory Redis for tests
- **Database fixtures**: Test data management

### Performance Testing
- **locust 2.16+**: Load testing framework
- **Memory profiling**: Track memory usage improvements
- **Response time monitoring**: API performance metrics

## Deployment Considerations

### Container Orchestration
- **Docker Compose**: Development and small deployments
- **Kubernetes**: Production scaling (future)
- **Health Checks**: Liveness and readiness probes

### Resource Requirements
```yaml
# Current (per user)
memory: 500MB
cpu: 0.5 cores

# Target (per user)
memory: 100MB
cpu: 0.1 cores

# Backend services
fastapi: 256MB, 0.25 cores
worker: 512MB, 0.5 cores
redis: 128MB, 0.1 cores
```

### Environment Configuration
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    database_url: str = "sqlite:///./app.db"
    openai_api_key: str
    azure_openai_endpoint: str
    max_workers: int = 4
    task_timeout: int = 300
    
    class Config:
        env_file = ".env"
```

## Security Enhancements

### Input Validation
- **Pydantic models**: Strict type validation
- **File upload limits**: Size and type restrictions
- **SQL injection prevention**: Parameterized queries
- **Code execution sandboxing**: Restricted Python execution

### Authentication & Authorization
- **JWT tokens**: Stateless authentication
- **Rate limiting**: Per-user request limits
- **CORS configuration**: Secure cross-origin requests
- **API key management**: Secure credential storage

### Data Protection
- **Session isolation**: User data separation
- **Temporary data cleanup**: Automatic data expiration
- **Audit logging**: Track user actions
- **Error sanitization**: No sensitive data in error messages

This technical foundation provides the infrastructure needed to migrate from a monolithic Streamlit application to a scalable, high-performance FastAPI backend while maintaining all existing functionality and significantly improving concurrent user support.
