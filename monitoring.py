import psutil
import os
import logging
import time
import threading
from collections import defaultdict
import tempfile
import streamlit as st
import logging.handlers

# Configure logging with rotation to prevent large log files
log_handler = logging.handlers.RotatingFileHandler(
    'app_monitoring.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=3
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources and application usage"""
    
    def __init__(self, log_frequency=5):
        """
        Initialize the resource monitor
        
        Args:
            log_frequency: How often to log resource usage (in monitoring intervals)
        """
        self.process = psutil.Process(os.getpid())
        self.temp_dirs = set()
        self.data_sizes = defaultdict(int)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.log_frequency = log_frequency
        self.log_counter = 0
        
    def start_monitoring(self, interval=5):
        """Start monitoring resources at specified interval (seconds)"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring resources"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval):
        """Internal monitoring loop"""
        while self.monitoring_active:
            try:
                self.log_counter += 1
                # Only log detailed resource usage based on frequency setting
                if self.log_counter % self.log_frequency == 0:
                    self.log_resource_usage()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def log_resource_usage(self):
        """Log current resource usage"""
        # Memory usage
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        
        # System-wide CPU usage
        system_cpu_percent = psutil.cpu_percent()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        logger.info(f"Process Memory: {memory_info.rss / 1024 / 1024:.2f} MB ({memory_percent:.2f}%)")
        logger.info(f"Process CPU: {cpu_percent:.2f}%")
        logger.info(f"System CPU: {system_cpu_percent:.2f}%")
        logger.info(f"System Memory: {system_memory.percent:.2f}% used of {system_memory.total / 1024 / 1024 / 1024:.2f} GB")
        
        # Log temporary directory usage
        self.log_temp_usage()
    
    def log_temp_usage(self):
        """Log temporary directory usage"""
        temp_dir = tempfile.gettempdir()
        try:
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(temp_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                        file_count += 1
                    except (OSError, FileNotFoundError):
                        pass
            
            logger.info(f"Temp directory: {file_count} files, {total_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            logger.error(f"Error checking temp directory: {e}")
    
    def register_temp_dir(self, temp_dir):
        """Register a temporary directory for tracking"""
        self.temp_dirs.add(temp_dir)
        logger.info(f"Registered temp directory: {temp_dir}")
    
    def unregister_temp_dir(self, temp_dir):
        """Unregister a temporary directory"""
        self.temp_dirs.discard(temp_dir)
        logger.info(f"Unregistered temp directory: {temp_dir}")
    
    def log_data_size(self, data_name, size_bytes):
        """Log data size for tracking"""
        self.data_sizes[data_name] = size_bytes
        size_mb = size_bytes / 1024 / 1024
        logger.info(f"Data '{data_name}': {size_mb:.2f} MB")
    
    def get_resource_summary(self):
        """Get a summary of current resource usage"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        cpu_percent = self.process.cpu_percent()
        system_memory = psutil.virtual_memory()
        
        summary = {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': memory_percent,
            'process_cpu_percent': cpu_percent,
            'system_memory_percent': system_memory.percent,
            'system_memory_total_gb': system_memory.total / 1024 / 1024 / 1024,
            'data_sizes': dict(self.data_sizes)
        }
        return summary

# Global monitor instance
monitor = ResourceMonitor()

def log_dataframe_size(df, name="dataframe"):
    """Log the size of a dataframe"""
    if hasattr(df, 'memory_usage'):
        size_bytes = df.memory_usage(deep=True).sum()
        monitor.log_data_size(name, size_bytes)
        return size_bytes
    return 0

def setup_monitoring():
    """Setup monitoring for the application"""
    # Start monitoring with 10 second intervals, log every 6 intervals (1 minute)
    monitor.start_monitoring(10)
    
    # Log initial resource usage
    monitor.log_resource_usage()
    
    logger.info("Monitoring setup completed")

def cleanup_monitoring():
    """Cleanup monitoring resources"""
    monitor.stop_monitoring()
    logger.info("Monitoring cleanup completed")

def display_resource_usage():
    """Display current resource usage in Streamlit"""
    try:
        summary = monitor.get_resource_summary()
        with st.sidebar:
        
            st.subheader("Resource Usage")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Process Memory", f"{summary['process_memory_mb']:.2f} MB", f"{summary['process_memory_percent']:.2f}%")
                st.metric("Process CPU", f"{summary['process_cpu_percent']:.2f}%")
            
            with col2:
                st.metric("System Memory", f"{summary['system_memory_percent']:.2f}%", f"{summary['system_memory_total_gb']:.2f} GB total")
            
            if summary['data_sizes']:
                st.subheader("Data Sizes")
                for name, size in summary['data_sizes'].items():
                    st.metric(name, f"{size / 1024 / 1024:.2f} MB")
                
    except Exception as e:
        st.warning(f"Could not display resource usage: {e}")

# Context manager for temporary directories
class MonitoredTempDir:
    """Context manager for temporary directories with monitoring"""
    
    def __init__(self, prefix="autoanalyzer_"):
        self.prefix = prefix
        self.temp_dir = None
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix=self.prefix)
        monitor.register_temp_dir(self.temp_dir)
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                monitor.unregister_temp_dir(self.temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temp directory {self.temp_dir}: {e}")