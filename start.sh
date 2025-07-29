#!/bin/bash

# AutoAnalyzer AI Startup Script
# This script starts the multi-service architecture

set -e

echo "🚀 Starting AutoAnalyzer AI with API Backend"
echo "============================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before running again."
    echo "   Especially set your OPENAI_API_KEY"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found."
    echo ""
    echo "🔧 Alternative options:"
    echo "1. Install docker-compose:"
    echo "   macOS: brew install docker-compose"
    echo "   Linux: sudo apt-get install docker-compose"
    echo ""
    echo "2. Use local development mode:"
    echo "   python start_local.py"
    echo ""
    echo "3. Try Docker Compose V2 (if you have Docker Desktop):"
    echo "   docker compose up --build -d"
    echo ""
    exit 1
fi

echo "🔧 Building and starting services..."

# Start the services
docker-compose up --build -d

echo "⏳ Waiting for services to be ready..."

# Wait for Redis
echo "   Waiting for Redis..."
until docker-compose exec redis redis-cli ping > /dev/null 2>&1; do
    sleep 1
done
echo "   ✅ Redis is ready"

# Wait for Backend
echo "   Waiting for Backend API..."
until curl -f http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
done
echo "   ✅ Backend API is ready"

# Wait for Frontend
echo "   Waiting for Frontend..."
until curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; do
    sleep 2
done
echo "   ✅ Frontend is ready"

echo ""
echo "🎉 AutoAnalyzer AI is now running!"
echo ""
echo "📊 Frontend (Streamlit): http://localhost:8501"
echo "🔧 Backend API: http://localhost:8000"
echo "📈 API Documentation: http://localhost:8000/docs"
echo "🌸 Task Monitor (Flower): http://localhost:5555 (run with --profile monitoring)"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart: docker-compose restart"
echo ""
echo "🔍 Check service status:"
docker-compose ps
