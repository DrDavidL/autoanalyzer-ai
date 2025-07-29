#!/usr/bin/env python3
"""
Local Development Startup Script
Runs AutoAnalyzer AI without Docker for development/testing
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['redis', 'fastapi', 'uvicorn', 'streamlit']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install redis fastapi uvicorn streamlit")
        return False
    
    return True

def check_redis():
    """Check if Redis is available"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return True
    except:
        return False

def start_redis():
    """Start Redis server if not running"""
    if check_redis():
        print("✅ Redis is already running")
        return None
    
    print("🔧 Starting Redis server...")
    try:
        # Try to start Redis
        redis_process = subprocess.Popen(['redis-server'], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
        time.sleep(2)
        
        if check_redis():
            print("✅ Redis started successfully")
            return redis_process
        else:
            print("❌ Failed to start Redis")
            return None
    except FileNotFoundError:
        print("❌ Redis not found. Please install Redis:")
        print("   macOS: brew install redis")
        print("   Ubuntu: sudo apt-get install redis-server")
        return None

def start_backend():
    """Start FastAPI backend"""
    print("🔧 Starting FastAPI backend...")
    
    # Change to backend directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None
    
    try:
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], cwd=backend_dir)
        
        # Wait a moment for startup
        time.sleep(3)
        print("✅ Backend started on http://localhost:8000")
        return backend_process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start Streamlit frontend"""
    print("🔧 Starting Streamlit frontend...")
    
    try:
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "frontend.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
        
        time.sleep(3)
        print("✅ Frontend started on http://localhost:8501")
        return frontend_process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🚀 Starting AutoAnalyzer AI (Local Development Mode)")
    print("=" * 55)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment file
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Creating from .env.example...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("📝 Please edit .env file with your configuration")
            print("   Especially set your OPENAI_API_KEY")
        else:
            print("❌ .env.example not found")
            sys.exit(1)
    
    processes = []
    
    try:
        # Start Redis
        redis_proc = start_redis()
        if redis_proc:
            processes.append(redis_proc)
        elif not check_redis():
            print("❌ Redis is required but not available")
            sys.exit(1)
        
        # Start Backend
        backend_proc = start_backend()
        if backend_proc:
            processes.append(backend_proc)
        else:
            print("❌ Failed to start backend")
            sys.exit(1)
        
        # Start Frontend
        frontend_proc = start_frontend()
        if frontend_proc:
            processes.append(frontend_proc)
        else:
            print("❌ Failed to start frontend")
            sys.exit(1)
        
        print("\n🎉 AutoAnalyzer AI is now running!")
        print("=" * 40)
        print("📊 Frontend: http://localhost:8501")
        print("🔧 Backend API: http://localhost:8000")
        print("📈 API Docs: http://localhost:8000/docs")
        print("\n💡 Press Ctrl+C to stop all services")
        
        # Wait for interrupt
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down services...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up processes
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
