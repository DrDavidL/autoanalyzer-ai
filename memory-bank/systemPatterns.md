# AutoAnalyzer AI - System Patterns

## Current Architecture

### Monolithic Streamlit Application
```
main.py (Entry Point)
├── UI Components (ui.py)
├── Data Processing (data_processing.py)
├── Machine Learning (ml.py)
├── Statistical Analysis (stats.py)
├── Plotting Functions (plotting.py)
├── LLM Integration (llm_integration.py)
├── Utilities (utils.py)
└── Explanations (explanations/)
```

### Key Technical Decisions

#### Session State Management
- **Pattern**: Streamlit session state for user data persistence
- **Implementation**: `st.session_state` dictionary for dataframes, models, results
- **Limitations**: Memory grows with concurrent users, no cross-session persistence

#### Data Flow Architecture
```
User Upload → Session State → Direct Function Calls → Results Display
```
- **Synchronous Processing**: All operations block the UI thread
- **Memory Intensive**: Each session loads full ML libraries and data
- **No Caching**: Repeated operations recalculate from scratch

#### Component Relationships

##### Data Processing Layer
- `data_processing.py`: Core data manipulation functions
- `stats.py`: Statistical test implementations
- `plotting.py`: Visualization generation
- **Pattern**: Pure functions with dataframe input/output
- **Dependencies**: pandas, numpy, scipy, matplotlib, seaborn

##### Machine Learning Layer
- `ml.py`: Model training and evaluation pipeline
- `explanations/`: SHAP analysis and model interpretation
- **Pattern**: Scikit-learn pipeline with custom preprocessing
- **Dependencies**: sklearn, xgboost, shap, lifelines

##### LLM Integration Layer
- `llm_integration.py`: GPT analysis and code generation
- **Pattern**: Langchain agents with pandas dataframe tools
- **Dependencies**: langchain, openai, azure-openai

## Target FastAPI Architecture

### Hybrid Architecture Design
```
Streamlit Frontend (UI Layer)
    ↓ HTTP/WebSocket
FastAPI Backend (API Layer)
    ↓ Task Queue
Worker Processes (Computation Layer)
    ↓ Shared Storage
Redis/Database (State Layer)
```

### API Design Patterns

#### RESTful Endpoints
```python
# Data Management
POST /api/data/upload
GET /api/data/{session_id}
POST /api/data/{session_id}/preprocess

# Statistical Analysis
POST /api/stats/ttest
POST /api/stats/anova
POST /api/stats/regression

# Machine Learning
POST /api/ml/train
GET /api/ml/job/{job_id}/status
GET /api/ml/job/{job_id}/results

# GPT Analysis
POST /api/gpt/analyze
GET /api/gpt/job/{job_id}/status
GET /api/gpt/job/{job_id}/results
```

#### Async Task Pattern
```python
@app.post("/api/ml/train")
async def train_model(request: MLTrainRequest):
    job_id = await queue.enqueue(
        train_model_task,
        data=request.data,
        config=request.config
    )
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/ml/job/{job_id}/status")
async def get_job_status(job_id: str):
    status = await queue.get_status(job_id)
    return {"job_id": job_id, "status": status}
```

### Critical Implementation Paths

#### 1. Machine Learning Pipeline Migration
**Current Flow:**
```python
# Synchronous, blocking
model = train_model(df, config)
results = evaluate_model(model, test_data)
shap_values = explain_model(model, data)
```

**Target Flow:**
```python
# Asynchronous, non-blocking
job_id = await ml_service.train_async(df, config)
# Frontend polls for completion
results = await ml_service.get_results(job_id)
```

#### 2. Statistical Analysis Migration
**Current Flow:**
```python
# Direct function calls
result = stats.run_ttest(df, col1, col2)
plot = plotting.create_boxplot(df, x, y)
```

**Target Flow:**
```python
# API calls with caching
result = await api_client.post("/stats/ttest", data)
plot_url = await api_client.get(f"/plots/{result.plot_id}")
```

#### 3. GPT Analysis Migration
**Current Flow:**
```python
# Synchronous code generation and execution
code = llm.generate_code(question, df)
result = exec(code)  # Blocks UI
```

**Target Flow:**
```python
# Async with progress tracking
job_id = await gpt_service.analyze_async(question, df)
# WebSocket for real-time updates
results = await gpt_service.stream_results(job_id)
```

## Design Patterns for Migration

### 1. Repository Pattern for Data Access
```python
class DataRepository:
    async def save_dataframe(self, session_id: str, df: pd.DataFrame)
    async def get_dataframe(self, session_id: str) -> pd.DataFrame
    async def cache_result(self, key: str, result: Any)
    async def get_cached_result(self, key: str) -> Any
```

### 2. Service Layer Pattern
```python
class MLService:
    def __init__(self, queue: TaskQueue, cache: Cache):
        self.queue = queue
        self.cache = cache
    
    async def train_model_async(self, data: dict) -> str:
        job_id = await self.queue.enqueue(train_model_task, data)
        return job_id
    
    async def get_results(self, job_id: str) -> dict:
        return await self.cache.get(f"ml_results:{job_id}")
```

### 3. Factory Pattern for Model Creation
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: dict):
        if model_type == "random_forest":
            return RandomForestClassifier(**config)
        elif model_type == "xgboost":
            return XGBClassifier(**config)
        # ... other models
```

### 4. Observer Pattern for Progress Tracking
```python
class ProgressTracker:
    def __init__(self):
        self.observers = []
    
    def attach(self, observer):
        self.observers.append(observer)
    
    def notify(self, progress: float, message: str):
        for observer in self.observers:
            observer.update(progress, message)
```

## State Management Strategy

### Session Isolation
- **Current**: Single process with shared session state
- **Target**: Per-user namespaced storage with Redis
- **Benefits**: Memory efficiency, crash isolation, horizontal scaling

### Caching Strategy
- **Level 1**: In-memory results cache (FastAPI process)
- **Level 2**: Redis cache for expensive computations
- **Level 3**: File system cache for large datasets and models

### Data Persistence
- **Temporary Data**: Redis with TTL for active sessions
- **Results**: File storage with metadata in database
- **Models**: Pickle serialization with versioning

## Error Handling Patterns

### Graceful Degradation
```python
async def robust_analysis(data, fallback_method=None):
    try:
        return await primary_analysis(data)
    except ResourceExhaustedException:
        if fallback_method:
            return await fallback_method(data)
        raise ServiceUnavailableError("Analysis temporarily unavailable")
```

### Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
```

## Performance Optimization Patterns

### Lazy Loading
- Load ML libraries only when needed
- Initialize models on first use
- Stream large datasets in chunks

### Connection Pooling
- Database connection pools
- HTTP client connection reuse
- Redis connection pooling

### Background Processing
- Pre-compute common statistical summaries
- Cache visualization templates
- Warm up model instances

## Security Patterns

### Input Validation
```python
from pydantic import BaseModel, validator

class MLTrainRequest(BaseModel):
    data: dict
    model_type: str
    config: dict
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_models = ['random_forest', 'xgboost', 'logistic_regression']
        if v not in allowed_models:
            raise ValueError(f'Model type must be one of {allowed_models}')
        return v
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/ml/train")
@limiter.limit("5/minute")
async def train_model(request: Request, data: MLTrainRequest):
    # Implementation
```

### Sandboxed Execution
```python
# For GPT-generated code execution
import subprocess
import tempfile

def execute_code_safely(code: str, timeout: int = 30):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
        f.write(code)
        f.flush()
        
        result = subprocess.run(
            ['python', f.name],
            capture_output=True,
            timeout=timeout,
            text=True
        )
        return result.stdout, result.stderr
```

This architecture provides the foundation for migrating from a monolithic Streamlit application to a scalable, concurrent FastAPI backend while maintaining all existing functionality and improving performance characteristics.
