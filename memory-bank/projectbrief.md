# AutoAnalyzer AI - Project Brief

## Project Overview
AutoAnalyzer AI is a comprehensive data analysis and machine learning platform built with Streamlit that provides interactive data exploration, automated machine learning, and AI-powered analysis capabilities for healthcare and research data.

## Core Requirements

### Current Architecture
- **Frontend**: Streamlit web application
- **Backend**: Monolithic Python application with direct function calls
- **Data Processing**: Synchronous operations blocking the UI
- **ML Pipeline**: Single-threaded model training
- **GPT Analysis**: Synchronous LLM calls with code execution
- **Deployment**: Docker container with single process

### Performance Challenges
- **Concurrency**: Limited to 2-3 simultaneous users
- **Memory Usage**: 500MB+ per user session
- **Blocking Operations**: ML training and GPT analysis block entire application
- **Resource Efficiency**: Poor utilization on low-powered hosts
- **Scalability**: Cannot handle multiple concurrent analyses

## Migration Goals

### Phase 1: Core API Backend (Weeks 1-2)
1. **Extract Heavy Computations to FastAPI**
   - Machine learning pipeline (`ml.py`)
   - Statistical functions (`stats.py`) 
   - Data processing operations (`data_processing.py`)
   - Plotting functions (`plotting.py`)

2. **Implement Job Queue System**
   - Async ML training with progress tracking
   - Background task processing
   - Result caching and retrieval

3. **Add Infrastructure**
   - Authentication and rate limiting
   - Request queuing and throttling
   - Health checks and monitoring

### Phase 2: Frontend Integration (Weeks 3-4)
1. **Modify Streamlit Frontend**
   - Replace direct function calls with API requests
   - Implement polling for long-running operations
   - Add progress tracking UI components

2. **State Management**
   - Session isolation between users
   - Result caching and persistence
   - Error handling and recovery

### Expected Performance Improvements
- **Concurrent Users**: 2-3 → 10-15 users
- **Memory Usage**: 500MB → 100MB per user
- **Resource Efficiency**: 30% → 85%
- **Response Time**: 2-10s → 0.5-2s
- **Operations**: Blocking → Non-blocking

## Technical Scope

### Components for FastAPI Migration
1. **Machine Learning Pipeline** (Priority: High)
   - Model training and evaluation
   - SHAP analysis and explanations
   - Feature preprocessing and selection

2. **Statistical Analysis** (Priority: Medium)
   - T-tests, ANOVA, Chi-square tests
   - Regression analysis
   - Survival analysis (Cox, Kaplan-Meier)

3. **Data Processing** (Priority: Medium)
   - Data cleaning and preprocessing
   - PCA and dimensionality reduction
   - Categorical encoding

4. **GPT Analysis Engine** (Priority: High)
   - Code generation and execution
   - Result interpretation
   - Iterative refinement

### Components Remaining in Streamlit
- User interface and forms
- File upload handling
- Result visualization and display
- Session management
- Authentication UI

## Success Criteria
1. **Performance**: Support 10+ concurrent users on low-powered Docker host
2. **Reliability**: Non-blocking operations with proper error handling
3. **Maintainability**: Clean separation between UI and business logic
4. **Compatibility**: Preserve all existing functionality
5. **Scalability**: Foundation for future horizontal scaling

## Constraints
- Must maintain existing user experience
- No breaking changes to current workflows
- Preserve all analysis capabilities
- Support existing data formats and demo datasets
- Maintain Docker deployment compatibility
