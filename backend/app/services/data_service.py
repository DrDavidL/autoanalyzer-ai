"""
Data Service
Handles data upload, processing, and management
"""

import io
import pandas as pd
from typing import Dict, Any, List, Optional
from app.services.session_service import SessionService
from app.core.logging import get_logger

logger = get_logger("services.data")

class DataService:
    """Service for data processing and management"""
    
    def __init__(self):
        self.session_service = SessionService()
    
    async def process_upload(
        self,
        session_id: str,
        filename: str,
        content: bytes,
        content_type: str = None
    ) -> Dict[str, Any]:
        """Process uploaded file and store in session"""
        try:
            # Determine file type and read data
            if filename.endswith('.csv') or 'csv' in (content_type or ''):
                df = pd.read_csv(io.BytesIO(content))
            elif filename.endswith(('.xlsx', '.xls')) or 'excel' in (content_type or ''):
                df = pd.read_excel(io.BytesIO(content))
            elif filename.endswith('.json') or 'json' in (content_type or ''):
                df = pd.read_json(io.BytesIO(content))
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Basic data validation
            if df.empty:
                raise ValueError("Uploaded file is empty")
            
            # Store dataframe in session
            await self.session_service.store_session_data(session_id, "dataframe", df)
            await self.session_service.store_session_data(session_id, "filename", filename)
            
            # Generate summary information
            result = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            logger.info(f"Processed upload for session {session_id}: {filename} ({result['rows']} rows)")
            return result
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            raise
    
    async def get_data_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistical summary of session data"""
        try:
            df = await self.session_service.get_session_data(session_id, "dataframe")
            if df is None:
                return None
            
            # Basic info
            summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            # Missing values
            summary["missing_values"] = df.isnull().sum().to_dict()
            summary["missing_percentage"] = (df.isnull().sum() / len(df) * 100).to_dict()
            
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                summary["categorical_summary"] = {}
                for col in categorical_cols:
                    summary["categorical_summary"][col] = {
                        "unique_count": df[col].nunique(),
                        "top_values": df[col].value_counts().head(5).to_dict()
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return None
    
    async def preprocess_data(
        self,
        session_id: str,
        operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply preprocessing operations to session data"""
        try:
            df = await self.session_service.get_session_data(session_id, "dataframe")
            if df is None:
                raise ValueError("No data found in session")
            
            original_shape = df.shape
            rows_affected = 0
            
            for operation in operations:
                op_type = operation.get("type")
                
                if op_type == "drop_missing":
                    # Drop rows with missing values
                    before_rows = len(df)
                    df = df.dropna()
                    rows_affected += before_rows - len(df)
                    
                elif op_type == "fill_missing":
                    # Fill missing values
                    method = operation.get("method", "mean")
                    columns = operation.get("columns", [])
                    
                    for col in columns:
                        if col in df.columns:
                            if method == "mean" and df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].mean(), inplace=True)
                            elif method == "median" and df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].median(), inplace=True)
                            elif method == "mode":
                                df[col].fillna(df[col].mode()[0], inplace=True)
                            elif method == "forward_fill":
                                df[col].fillna(method='ffill', inplace=True)
                            elif method == "backward_fill":
                                df[col].fillna(method='bfill', inplace=True)
                
                elif op_type == "remove_duplicates":
                    # Remove duplicate rows
                    before_rows = len(df)
                    df = df.drop_duplicates()
                    rows_affected += before_rows - len(df)
                
                elif op_type == "convert_types":
                    # Convert column data types
                    conversions = operation.get("conversions", {})
                    for col, new_type in conversions.items():
                        if col in df.columns:
                            try:
                                if new_type == "numeric":
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                elif new_type == "datetime":
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                                elif new_type == "category":
                                    df[col] = df[col].astype('category')
                            except Exception as e:
                                logger.warning(f"Failed to convert {col} to {new_type}: {e}")
            
            # Store updated dataframe
            await self.session_service.store_session_data(session_id, "dataframe", df)
            
            result = {
                "original_shape": original_shape,
                "new_shape": df.shape,
                "rows_affected": rows_affected,
                "operations_applied": len(operations)
            }
            
            logger.info(f"Applied {len(operations)} preprocessing operations to session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    async def load_demo_dataset(self, session_id: str, dataset_id: str) -> Dict[str, Any]:
        """Load a demo dataset into session"""
        try:
            # Demo datasets mapping
            demo_datasets = {
                "breast_cancer": "data/breastcancernew.csv",
                "diabetes": "data/diabetes_prediction_dataset.csv",
                "stroke": "data/healthcare-dataset-stroke-data.csv"
            }
            
            if dataset_id not in demo_datasets:
                raise ValueError(f"Unknown dataset: {dataset_id}")
            
            # Load the dataset
            file_path = demo_datasets[dataset_id]
            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                # Fallback to generating sample data
                df = self._generate_sample_data(dataset_id)
            
            # Store in session
            await self.session_service.store_session_data(session_id, "dataframe", df)
            await self.session_service.store_session_data(session_id, "filename", f"{dataset_id}.csv")
            
            result = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict()
            }
            
            logger.info(f"Loaded demo dataset {dataset_id} for session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading demo dataset: {e}")
            raise
    
    def _generate_sample_data(self, dataset_id: str) -> pd.DataFrame:
        """Generate sample data if demo files not available"""
        import numpy as np
        
        if dataset_id == "breast_cancer":
            # Generate sample breast cancer data
            n_samples = 569
            np.random.seed(42)
            
            data = {
                'radius_mean': np.random.normal(14, 3, n_samples),
                'texture_mean': np.random.normal(19, 4, n_samples),
                'perimeter_mean': np.random.normal(92, 24, n_samples),
                'area_mean': np.random.normal(655, 352, n_samples),
                'smoothness_mean': np.random.normal(0.096, 0.014, n_samples),
                'diagnosis': np.random.choice(['M', 'B'], n_samples, p=[0.37, 0.63])
            }
            
        elif dataset_id == "diabetes":
            # Generate sample diabetes data
            n_samples = 768
            np.random.seed(42)
            
            data = {
                'Pregnancies': np.random.poisson(3, n_samples),
                'Glucose': np.random.normal(120, 30, n_samples),
                'BloodPressure': np.random.normal(70, 12, n_samples),
                'BMI': np.random.normal(32, 7, n_samples),
                'Age': np.random.randint(21, 81, n_samples),
                'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
            }
            
        elif dataset_id == "stroke":
            # Generate sample stroke data
            n_samples = 1000
            np.random.seed(42)
            
            data = {
                'age': np.random.randint(18, 90, n_samples),
                'hypertension': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'heart_disease': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
                'avg_glucose_level': np.random.normal(106, 45, n_samples),
                'bmi': np.random.normal(28, 7, n_samples),
                'stroke': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
            }
        
        else:
            raise ValueError(f"No sample data generator for {dataset_id}")
        
        return pd.DataFrame(data)
