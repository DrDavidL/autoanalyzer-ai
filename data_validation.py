import pandas as pd
import logging
from typing import Optional, Tuple
import monitoring

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_ROWS = 100000  # 100K rows
DEFAULT_MAX_COLUMNS = 1000  # 1000 columns
DEFAULT_MAX_FILE_SIZE_MB = 100  # 100 MB
DEFAULT_MAX_MEMORY_USAGE_MB = 500  # 500 MB

class DataValidator:
    """Validate data size and structure to prevent resource exhaustion"""
    
    def __init__(self, 
                 max_rows: int = DEFAULT_MAX_ROWS,
                 max_columns: int = DEFAULT_MAX_COLUMNS,
                 max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
                 max_memory_usage_mb: int = DEFAULT_MAX_MEMORY_USAGE_MB):
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_file_size_mb = max_file_size_mb
        self.max_memory_usage_mb = max_memory_usage_mb
    
    def validate_dataframe(self, df: pd.DataFrame, name: str = "dataframe") -> Tuple[bool, str]:
        """
        Validate a dataframe against size limits
        
        Args:
            df: DataFrame to validate
            name: Name of the dataframe for logging
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check dimensions
            rows, cols = df.shape
            if rows > self.max_rows:
                return False, f"DataFrame exceeds maximum rows limit ({rows} > {self.max_rows})"
            
            if cols > self.max_columns:
                return False, f"DataFrame exceeds maximum columns limit ({cols} > {self.max_columns})"
            
            # Check memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            memory_mb = memory_usage / (1024 * 1024)
            monitoring.log_dataframe_size(df, name)
            
            if memory_mb > self.max_memory_usage_mb:
                return False, f"DataFrame exceeds maximum memory usage ({memory_mb:.2f} MB > {self.max_memory_usage_mb} MB)"
            
            logger.debug(f"DataFrame '{name}' validation passed: {rows} rows, {cols} columns, {memory_mb:.2f} MB")
            return True, f"Valid: {rows} rows, {cols} columns, {memory_mb:.2f} MB"
            
        except Exception as e:
            logger.error(f"Error validating dataframe '{name}': {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_file_size(self, file_path: str, name: str = "file") -> Tuple[bool, str]:
        """
        Validate a file size against limits
        
        Args:
            file_path: Path to the file
            name: Name of the file for logging
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            import os
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > self.max_file_size_mb:
                return False, f"File exceeds maximum size limit ({file_size_mb:.2f} MB > {self.max_file_size_mb} MB)"
            
            logger.debug(f"File '{name}' validation passed: {file_size_mb:.2f} MB")
            return True, f"Valid: {file_size_mb:.2f} MB"
            
        except Exception as e:
            logger.error(f"Error validating file '{name}': {e}")
            return False, f"Validation error: {str(e)}"
    
    def validate_csv_file(self, file_path: str, name: str = "CSV file") -> Tuple[bool, str]:
        """
        Validate a CSV file by checking its size and sampling rows to estimate full size
        
        Args:
            file_path: Path to the CSV file
            name: Name of the file for logging
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # First check file size
            file_valid, file_msg = self.validate_file_size(file_path, name)
            if not file_valid:
                return False, file_msg
            
            # Sample first few rows to estimate full dataframe size
            sample_df = pd.read_csv(file_path, nrows=1000)
            sample_valid, sample_msg = self.validate_dataframe(sample_df, f"{name} sample")
            
            if not sample_valid:
                return False, f"Sample validation failed: {sample_msg}"
            
            # If sample is valid, assume full file is valid
            return True, f"CSV file appears valid based on sampling: {sample_msg}"
            
        except Exception as e:
            logger.error(f"Error validating CSV file '{name}': {e}")
            return False, f"Validation error: {str(e)}"

# Global validator instance with default limits
validator = DataValidator()

def validate_uploaded_data(df: pd.DataFrame, name: str = "uploaded_data") -> bool:
    """
    Validate uploaded data and show warnings in Streamlit if limits are exceeded
    
    Args:
        df: DataFrame to validate
        name: Name of the data for display
        
    Returns:
        True if valid, False if invalid
    """
    import streamlit as st
    
    is_valid, message = validator.validate_dataframe(df, name)
    
    if not is_valid:
        st.error(f"Data validation failed: {message}")
        st.warning(f"Please reduce your data size. Limits: {validator.max_rows} rows, {validator.max_columns} columns, {validator.max_memory_usage_mb} MB")
        return False
    else:
        st.success(f"Data validation passed: {message}")
        return True

def set_validation_limits(max_rows: Optional[int] = None,
                         max_columns: Optional[int] = None,
                         max_file_size_mb: Optional[int] = None,
                         max_memory_usage_mb: Optional[int] = None):
    """
    Update validation limits
    
    Args:
        max_rows: Maximum number of rows
        max_columns: Maximum number of columns
        max_file_size_mb: Maximum file size in MB
        max_memory_usage_mb: Maximum memory usage in MB
    """
    global validator
    if max_rows is not None:
        validator.max_rows = max_rows
    if max_columns is not None:
        validator.max_columns = max_columns
    if max_file_size_mb is not None:
        validator.max_file_size_mb = max_file_size_mb
    if max_memory_usage_mb is not None:
        validator.max_memory_usage_mb = max_memory_usage_mb
    
    logger.info(f"Updated validation limits: rows={validator.max_rows}, cols={validator.max_columns}, "
                f"file_size={validator.max_file_size_mb} MB, memory={validator.max_memory_usage_mb} MB")

# Convenience functions
def check_data_size(df: pd.DataFrame, name: str = "dataframe") -> Tuple[bool, str]:
    """Check if a dataframe exceeds size limits"""
    return validator.validate_dataframe(df, name)

def check_file_size(file_path: str, name: str = "file") -> Tuple[bool, str]:
    """Check if a file exceeds size limits"""
    return validator.validate_file_size(file_path, name)