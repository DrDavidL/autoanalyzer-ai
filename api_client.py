"""
API Client for AutoAnalyzer AI Backend
Provides interface between Streamlit frontend and FastAPI backend
"""

import httpx
import asyncio
import streamlit as st
from typing import Dict, Any, Optional, List
import pandas as pd
import io
import time
import json

class AutoAnalyzerAPIClient:
    """Client for communicating with AutoAnalyzer AI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 30.0
        
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict = None, 
        files: Dict = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """Make HTTP request to backend"""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.timeout
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, params=data)
                elif method.upper() == "POST":
                    if files:
                        response = await client.post(url, data=data, files=files)
                    else:
                        response = await client.post(url, json=data)
                elif method.upper() == "DELETE":
                    response = await client.delete(url)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException:
            raise Exception(f"Request timeout after {timeout}s")
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json() if e.response.content else {"message": str(e)}
            raise Exception(f"API Error: {error_detail.get('detail', str(e))}")
        except Exception as e:
            raise Exception(f"Connection error: {str(e)}")
    
    # Session Management
    async def create_session(self, session_name: str = None) -> Dict[str, Any]:
        """Create a new session"""
        data = {"session_name": session_name} if session_name else {}
        return await self._make_request("POST", "/api/data/sessions", data)
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        return await self._make_request("GET", f"/api/data/sessions/{session_id}")
    
    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session"""
        return await self._make_request("DELETE", f"/api/data/sessions/{session_id}")
    
    # Data Management
    async def upload_file(self, session_id: str, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload a file to session"""
        files = {"file": (filename, io.BytesIO(file_content))}
        data = {"session_id": session_id} if session_id else {}
        return await self._make_request("POST", "/api/data/upload", data, files, timeout=60.0)
    
    async def load_demo_dataset(self, dataset_id: str, session_id: str = None) -> Dict[str, Any]:
        """Load a demo dataset"""
        data = {"session_id": session_id} if session_id else {}
        return await self._make_request("POST", f"/api/data/demo-datasets/{dataset_id}/load", data)
    
    async def get_session_data(self, session_id: str, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get data from session"""
        params = {"limit": limit, "offset": offset}
        return await self._make_request("GET", f"/api/data/sessions/{session_id}/data", params)
    
    async def get_data_summary(self, session_id: str) -> Dict[str, Any]:
        """Get data summary"""
        return await self._make_request("GET", f"/api/data/sessions/{session_id}/summary")
    
    # Statistical Analysis
    async def run_ttest(
        self, 
        session_id: str, 
        column1: str, 
        column2: str = None, 
        test_type: str = "two_sample"
    ) -> Dict[str, Any]:
        """Run t-test analysis"""
        data = {
            "session_id": session_id,
            "column1": column1,
            "column2": column2,
            "test_type": test_type
        }
        return await self._make_request("POST", "/api/stats/ttest", data, timeout=60.0)
    
    async def run_correlation(
        self, 
        session_id: str, 
        columns: List[str], 
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """Run correlation analysis"""
        data = {
            "session_id": session_id,
            "columns": columns,
            "method": method
        }
        return await self._make_request("POST", "/api/stats/correlation", data, timeout=60.0)
    
    async def get_descriptive_stats(self, session_id: str, columns: List[str] = None) -> Dict[str, Any]:
        """Get descriptive statistics"""
        params = {"columns": columns} if columns else {}
        return await self._make_request("GET", f"/api/stats/sessions/{session_id}/descriptive", params)
    
    # Machine Learning
    async def train_model(
        self, 
        session_id: str, 
        target_column: str, 
        feature_columns: List[str],
        model_type: str = "auto"
    ) -> Dict[str, Any]:
        """Start model training"""
        data = {
            "session_id": session_id,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "model_type": model_type
        }
        return await self._make_request("POST", "/api/ml/train", data, timeout=10.0)
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        return await self._make_request("GET", f"/api/ml/jobs/{job_id}/status")
    
    async def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get job results"""
        return await self._make_request("GET", f"/api/ml/jobs/{job_id}/results")
    
    # GPT Analysis
    async def start_gpt_analysis(
        self, 
        session_id: str, 
        question: str, 
        context: str = None
    ) -> Dict[str, Any]:
        """Start GPT analysis"""
        data = {
            "session_id": session_id,
            "question": question,
            "context": context
        }
        return await self._make_request("POST", "/api/gpt/analyze", data, timeout=10.0)
    
    async def generate_code(
        self, 
        session_id: str, 
        question: str, 
        context: str = None
    ) -> Dict[str, Any]:
        """Generate code without executing"""
        data = {
            "session_id": session_id,
            "question": question,
            "context": context
        }
        return await self._make_request("POST", "/api/gpt/generate-code", data, timeout=60.0)
    
    async def execute_code(self, session_id: str, code: str) -> Dict[str, Any]:
        """Execute Python code"""
        data = {
            "session_id": session_id,
            "code": code
        }
        return await self._make_request("POST", "/api/gpt/execute-code", data, timeout=120.0)
    
    # Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        return await self._make_request("GET", "/health")


# Streamlit Integration Helpers
class StreamlitAPIHelper:
    """Helper class for integrating API client with Streamlit"""
    
    def __init__(self, api_client: AutoAnalyzerAPIClient):
        self.client = api_client
    
    def run_async(self, coro):
        """Run async function in Streamlit"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    def ensure_session(self) -> str:
        """Ensure we have a valid session ID"""
        if "api_session_id" not in st.session_state:
            try:
                result = self.run_async(self.client.create_session())
                st.session_state.api_session_id = result["session_id"]
                st.success(f"Created new session: {result['session_id'][:8]}...")
            except Exception as e:
                st.error(f"Failed to create session: {e}")
                return None
        
        return st.session_state.api_session_id
    
    def upload_file_to_api(self, uploaded_file) -> bool:
        """Upload file using API"""
        session_id = self.ensure_session()
        if not session_id:
            return False
        
        try:
            with st.spinner("Uploading file to backend..."):
                file_content = uploaded_file.read()
                result = self.run_async(
                    self.client.upload_file(session_id, file_content, uploaded_file.name)
                )
                
                st.success(f"Uploaded {uploaded_file.name}: {result['rows']} rows, {result['columns']} columns")
                
                # Store result in session state for compatibility
                st.session_state.upload_result = result
                return True
                
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return False
    
    def load_demo_dataset_api(self, dataset_id: str) -> bool:
        """Load demo dataset using API"""
        session_id = self.ensure_session()
        if not session_id:
            return False
        
        try:
            with st.spinner(f"Loading {dataset_id} dataset..."):
                result = self.run_async(
                    self.client.load_demo_dataset(dataset_id, session_id)
                )
                
                st.success(f"Loaded {dataset_id}: {result['rows']} rows, {result['columns']} columns")
                
                # Store result in session state for compatibility
                st.session_state.upload_result = result
                return True
                
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return False
    
    def get_data_for_display(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get data from API for display"""
        session_id = st.session_state.get("api_session_id")
        if not session_id:
            return None
        
        try:
            result = self.run_async(
                self.client.get_session_data(session_id, limit=limit)
            )
            
            if result and "data" in result:
                return pd.DataFrame(result["data"])
            
        except Exception as e:
            st.error(f"Failed to get data: {e}")
        
        return None
    
    def run_statistical_test_api(self, test_type: str, **kwargs) -> Optional[Dict]:
        """Run statistical test using API"""
        session_id = st.session_state.get("api_session_id")
        if not session_id:
            st.error("No active session")
            return None
        
        try:
            with st.spinner(f"Running {test_type} analysis..."):
                if test_type == "ttest":
                    result = self.run_async(
                        self.client.run_ttest(session_id, **kwargs)
                    )
                elif test_type == "correlation":
                    result = self.run_async(
                        self.client.run_correlation(session_id, **kwargs)
                    )
                else:
                    st.error(f"Unsupported test type: {test_type}")
                    return None
                
                return result
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return None
    
    def poll_job_status(self, job_id: str, max_wait: int = 300) -> Optional[Dict]:
        """Poll job status until completion"""
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            while time.time() - start_time < max_wait:
                status = self.run_async(self.client.get_job_status(job_id))
                
                if status["status"] == "completed":
                    progress_bar.progress(1.0)
                    status_text.success("Analysis completed!")
                    
                    # Get results
                    results = self.run_async(self.client.get_job_results(job_id))
                    return results
                
                elif status["status"] == "failed":
                    progress_bar.empty()
                    status_text.error(f"Analysis failed: {status.get('message', 'Unknown error')}")
                    return None
                
                else:
                    # Update progress
                    progress = status.get("progress", 0.0)
                    progress_bar.progress(progress)
                    status_text.info(f"Status: {status['status']} ({progress:.1%})")
                    
                    time.sleep(2)  # Poll every 2 seconds
            
            # Timeout
            progress_bar.empty()
            status_text.error("Analysis timed out")
            return None
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"Error polling job status: {e}")
            return None


# Global API client instance
@st.cache_resource
def get_api_client() -> AutoAnalyzerAPIClient:
    """Get cached API client instance"""
    base_url = "http://localhost:8000"  # Default for development
    return AutoAnalyzerAPIClient(base_url)

@st.cache_resource
def get_api_helper() -> StreamlitAPIHelper:
    """Get cached API helper instance"""
    client = get_api_client()
    return StreamlitAPIHelper(client)
