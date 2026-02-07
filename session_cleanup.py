"""
Session cleanup utilities for AutoAnalyzer
Handles temporary file management and session isolation
"""

import os
import shutil
import tempfile
import atexit
import streamlit as st
from typing import Optional
import uuid
import time
import glob


class SessionManager:
    """Manages session-specific temporary directories and cleanup"""
    
    def __init__(self):
        self.session_id = None
        self.temp_dir = None
        self.cleanup_registered = False
    
    def get_session_id(self) -> str:
        """Get or create a unique session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def get_temp_directory(self) -> str:
        """Get or create session-specific temporary directory"""
        if 'temp_directory' not in st.session_state:
            session_id = self.get_session_id()
            # Create session-specific temp directory
            base_temp = tempfile.gettempdir()
            session_temp = os.path.join(base_temp, f"autoanalyzer_{session_id}")
            os.makedirs(session_temp, exist_ok=True)
            st.session_state.temp_directory = session_temp
            
            # Register cleanup on exit
            if not self.cleanup_registered:
                atexit.register(self.cleanup_session_files, session_temp)
                self.cleanup_registered = True
        
        return st.session_state.temp_directory
    
    def cleanup_session_files(self, temp_dir: Optional[str] = None):
        """Clean up session-specific temporary files"""
        if temp_dir is None:
            temp_dir = st.session_state.get('temp_directory')
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up session directory: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up session directory {temp_dir}: {e}")
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session directories that are older than max_age_hours"""
        base_temp = tempfile.gettempdir()
        pattern = os.path.join(base_temp, "autoanalyzer_*")
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for session_dir in glob.glob(pattern):
            try:
                # Check if directory is older than max_age
                dir_mtime = os.path.getmtime(session_dir)
                if current_time - dir_mtime > max_age_seconds:
                    shutil.rmtree(session_dir)
                    print(f"Cleaned up old session directory: {session_dir}")
            except Exception as e:
                print(f"Error cleaning up old session directory {session_dir}: {e}")
    
    def reset_session(self):
        """Reset current session and clean up files"""
        # Clean up current session files
        self.cleanup_session_files()
        
        # Clear session state variables related to file storage
        keys_to_clear = [
            'temp_directory', 'session_id', 'outputs_path',
            'gpt_analysis_images', 'persistent_gpt_images',
            'data_gov_results', 'gen_csv'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Force creation of new session
        self.session_id = None
        self.temp_dir = None


# Global session manager instance
session_manager = SessionManager()


def get_session_temp_dir() -> str:
    """Get the current session's temporary directory"""
    return session_manager.get_temp_directory()


def cleanup_on_session_start():
    """Clean up old sessions when a new session starts"""
    # Clean up sessions older than 24 hours
    session_manager.cleanup_old_sessions(max_age_hours=24)


def register_session_cleanup():
    """Register cleanup functions for the current session"""
    # This is called automatically by SessionManager
    pass


def safe_file_cleanup(file_path: str):
    """Safely remove a file if it exists"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")


def get_session_file_path(filename: str) -> str:
    """Get a session-specific file path"""
    temp_dir = get_session_temp_dir()
    return os.path.join(temp_dir, filename)


def cleanup_session_images():
    """Clean up all images in the current session directory"""
    temp_dir = get_session_temp_dir()
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']
    
    for pattern in image_patterns:
        for img_path in glob.glob(os.path.join(temp_dir, pattern)):
            safe_file_cleanup(img_path)


def get_session_info() -> dict:
    """Get information about the current session"""
    temp_dir = get_session_temp_dir()
    session_id = session_manager.get_session_id()
    
    # Count files in session directory
    file_count = 0
    total_size = 0
    
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir):
            file_count += len(files)
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                except:
                    pass
    
    return {
        'session_id': session_id,
        'temp_directory': temp_dir,
        'file_count': file_count,
        'total_size_mb': total_size / (1024 * 1024),
        'exists': os.path.exists(temp_dir)
    }
